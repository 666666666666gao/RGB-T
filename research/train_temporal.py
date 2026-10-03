"""Train coupled A/B/C on completed causal TRAIN caches; no test selection."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .temporal_modules import TemporalModules, temporal_objective, relative_geometry, DECISION_FIELDS


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', nargs='+', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--c1-head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--slots', type=int, default=4)
    p.add_argument('--motion-history', type=int, default=8)
    p.add_argument('--modes', type=int, default=3)
    p.add_argument('--rank-gap', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_data(roots, partition, device):
    arrays, configs, names = [], [], set()
    for root in roots:
        root = Path(root)
        config = json.loads((root / 'config.json').read_text())
        receipt = json.loads((root / 'completion.json').read_text())
        assert config['partition'] == receipt['partition'] == partition
        assert receipt['completed'] and not receipt['decision_input_contains_future']
        with np.load(root / 'samples.npz') as archive:
            row = {key: torch.from_numpy(archive[key].copy()).to(device) for key in archive.files}
        assert len(row['valid']) == receipt['clips'] == config['clips']
        assert row['valid'].any(1).all() and row['history_valid'][:, -1].all()
        assert row['history_descriptors'].shape[1:] == (config['history'], 2, 768)
        assert row['modality_features'].shape[2:] == (2, 768)
        assert all(torch.isfinite(v).all() for v in row.values())
        assert row['valid'].gather(1, row['original_choice'].long()[:, None]).all()
        arrays.append(row)
        configs.append(config)
        names.update(job['sequence'] for job in config['jobs'])
    keys = arrays[0].keys()
    assert all(a.keys() == keys for a in arrays)
    return {key: torch.cat([a[key] for a in arrays]) for key in keys}, configs, names


def forward(model, data):
    return model({key: data[key] for key in DECISION_FIELDS})


@torch.no_grad()
def evaluate(model, data, batch_size, rank_gap):
    model.eval()
    results = []
    motion_nll, total_loss, count = 0., 0., 0
    for start in range(0, len(data['valid']), batch_size):
        batch = {key: value[start:start + batch_size] for key, value in data.items()}
        output = forward(model, batch)
        loss, parts = temporal_objective(model, output, batch, rank_gap)
        valid = batch['valid']
        chosen = output['scores'].masked_fill(~valid, -torch.inf).argmax(1)
        original = batch['original_choice'].long()
        current = batch['current_iou']
        future = batch['future_iou'].mean(-1)
        utility = .7 * current + .3 * future - .1 * batch['wrong_update_fraction']
        selected_current = current.gather(1, chosen[:, None]).squeeze(1)
        c1_current = current.gather(1, original[:, None]).squeeze(1)
        selected_utility = utility.gather(1, chosen[:, None]).squeeze(1)
        c1_utility = utility.gather(1, original[:, None]).squeeze(1)
        target_motion = relative_geometry(batch['motion_targets'].float(), output['distribution']['reference'])
        displacement = torch.linalg.vector_norm(output['distribution']['means'][..., :2] - target_motion[:, None, :, :2], dim=-1)
        best_mode_ade = displacement.mean(-1).min(-1).values
        results.append({'current': selected_current, 'c1_current': c1_current,
                        'utility': selected_utility, 'c1_utility': c1_utility,
                        'regret': utility.masked_fill(~valid, -torch.inf).max(1).values - selected_utility,
                        'reselected': chosen != original,
                        'rescued': (c1_current < .2) & (selected_current >= .5),
                        'harmed': (c1_current >= .5) & (selected_current < .2),
                        'failed': selected_current < .2, 'c1_failed': c1_current < .2,
                        'recall': current.masked_fill(~valid, -torch.inf).max(1).values >= .5,
                        'best_mode_normalized_ade': best_mode_ade})
        size = len(chosen)
        total_loss += float(loss) * size
        motion_nll += parts['motion'] * size
        count += size
    all_values = {key: torch.cat([r[key] for r in results]) for key in results[0]}
    metrics = {'clips': count, 'valid_actions': int(data['valid'].sum()),
               'loss': total_loss / count, 'motion_trajectory_nll': motion_nll / count}
    for key, values in all_values.items():
        metrics[key] = int(values.sum()) if values.dtype == torch.bool else float(values.mean())
    metrics['current_gain_against_c1'] = metrics['current'] - metrics['c1_current']
    metrics['utility_gain_against_c1'] = metrics['utility'] - metrics['c1_utility']
    assert all(np.isfinite(value) for value in metrics.values())
    return metrics


def main():
    args = arguments()
    assert args.epochs > 0 and args.batch_size > 0 and args.motion_history == 8 and args.modes == 3
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    train, train_configs, train_names = load_data(args.train, 'train', device)
    validation, val_configs, val_names = load_data([args.validation], 'validation', device)
    assert not train_names & val_names
    reference = train_configs[0]
    for config in train_configs + val_configs:
        for key in ('split', 'root', 'cache', 'head', 'pretrained', 'history', 'horizon', 'prefix_policy', 'future_policy'):
            assert config[key] == reference[key], key
        assert config['head'] == args.c1_head
    split = json.loads(Path(reference['split']).read_text())
    assert train_names <= set(split['train']) and val_names <= set(split['validation'])
    c1 = torch.load(args.c1_head, map_location='cpu', weights_only=False)
    model = TemporalModules(c1, args.slots, args.motion_history, args.modes, reference['horizon']).to(device)
    groups = {'A': model.memory, 'B': model.motion, 'C': model.selector}
    initial = {name: {key: value.detach().clone() for key, value in module.state_dict().items()}
               for name, module in groups.items()}
    model.eval()
    with torch.no_grad():
        for data in (train, validation):
            for start in range(0, len(data['valid']), args.batch_size):
                batch = {k: v[start:start + args.batch_size] for k, v in data.items()}
                output = forward(model, batch)
                assert torch.equal(output['scores'], output['c1_scores']), 'Zero residual changed C1 scores'
                choices = output['scores'].masked_fill(~batch['valid'], -torch.inf).argmax(1)
                assert torch.equal(choices, batch['original_choice'].long()), 'Zero residual changed C1 choices'
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'module': 'ABC_temporal', 'train_clips': len(train['valid']),
                           'validation_clips': len(validation['valid']),
                           'train_sequences': len(train_names), 'validation_sequences': len(val_names),
                           'source_configs': train_configs + val_configs,
                           'decision_fields': DECISION_FIELDS, 'initial_all_choices_match_c1': True,
                           'base_GOLA': 'full pretrained frozen features; collector has no gradients',
                           'c1_trainable_parameters': sum(p.numel() for p in model.c1.parameters() if p.requires_grad),
                           'new_trainable_parameters': {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()},
                           'checkpoint_selection': 'maximum fixed TRAIN-held-out selected current IoU; strict improvement',
                           'utility_target': '.7 current IoU + .3 future rollout mean IoU - .1 wrong-write fraction',
                           'scope': 'coupled learned ABC cache training, not official tracking accuracy or completed online integration'}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    metrics = evaluate(model, validation, args.batch_size, args.rank_gap)
    records = [{'epoch': 0, **metrics}]
    best = metrics['current']
    def save(name, epoch, metrics):
        torch.save({'module': 'ABC_temporal', 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'args': vars(args), 'horizon': reference['horizon'],
                    'validation': metrics, 'selection_policy': config['checkpoint_selection']}, out / name)
    save('initial.pth', 0, metrics)
    save('best.pth', 0, metrics)
    (out / 'metrics.json').write_text(json.dumps(records, indent=2))
    print('INITIAL', json.dumps(metrics), flush=True)
    started, steps = time.perf_counter(), 0
    max_gradients = {name: 0. for name in groups}
    with (out / 'train.jsonl').open('w') as stream:
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train['valid']), device=device)
            for step, indices in enumerate(order.split(args.batch_size), 1):
                batch = {key: value[indices] for key, value in train.items()}
                output = forward(model, batch)
                loss, parts = temporal_objective(model, output, batch, args.rank_gap)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = {}
                for name, module in groups.items():
                    norm = torch.stack([p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]).sum().sqrt()
                    assert torch.isfinite(norm)
                    gradients[name] = float(norm)
                    max_gradients[name] = max(max_gradients[name], float(norm))
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
                assert torch.isfinite(norm) and all(p.grad is None for p in model.c1.parameters())
                optimizer.step()
                steps += 1
                if step == 1 or step % 10 == 0 or step * args.batch_size >= len(order):
                    row = {'epoch': epoch, 'step': step, 'optimizer_steps': steps,
                           'loss': float(loss.detach()), 'parts': parts, 'module_gradient_norms': gradients,
                           'elapsed_seconds': time.perf_counter() - started,
                           'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20}
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    print('TRAIN', json.dumps(row), flush=True)
            metrics = evaluate(model, validation, args.batch_size, args.rank_gap)
            records.append({'epoch': epoch, **metrics})
            save('last.pth', epoch, metrics)
            if metrics['current'] > best:
                best = metrics['current']
                save('best.pth', epoch, metrics)
            (out / 'metrics.json').write_text(json.dumps(records, indent=2))
            print('VALIDATION', json.dumps(records[-1]), flush=True)
    changed = {name: any(not torch.equal(initial[name][key], value) for key, value in module.state_dict().items())
               for name, module in groups.items()}
    assert all(changed.values()) and all(norm > 0 for norm in max_gradients.values())
    receipt = {'completed': True, 'epochs': args.epochs, 'optimizer_steps': steps,
               'modules_changed': changed, 'max_module_gradient_norms': max_gradients,
               'frozen_c1_gradients_absent': all(p.grad is None for p in model.c1.parameters()),
               'elapsed_seconds': time.perf_counter() - started,
               'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20,
               'last_validation': metrics, 'best_validation_current_iou': best,
               'official_tracking_accuracy': False}
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
