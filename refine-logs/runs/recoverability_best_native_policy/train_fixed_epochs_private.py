"""Train ROI memory, one-extra-region search, and relative action advantage.

Checkpoint selection uses held-out budgeted rollout utility, never test data.
Retain cached best.pth and requested temporary fixed epochs for continuous validation; keep all logs.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .recoverability_modules import (DECISION_FIELDS, RecoverabilityModules,
                                     action_utility, objective, select_actions)

LABEL_FIELDS = ('current_iou', 'future_iou', 'wrong_update_fraction',
                'action_valid', 'history_iou')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', nargs='+', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--c1-head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--init-checkpoint', help='Continue the learned ABC modules from an existing checkpoint; C1 stays frozen.')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--retain-epochs', type=int, nargs='*', default=[],
                   help='Temporarily retain fixed epochs for full continuous developer validation.')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--threshold', type=float, default=.03)
    p.add_argument('--search-supervision', choices=('oracle', 'selector'), default='oracle',
                   help='Region utility upper bound, or detached deployed-selector marginal utility.')
    p.add_argument('--action-ranking', choices=('reference', 'pairwise', 'budgeted'), default='reference',
                   help='Keep sign margin, coexisting pair ordering, or budgeted winner versus current rival.')
    p.add_argument('--write-verification', choices=('identity', 'action'), default='identity')
    p.add_argument('--frozen-modules', nargs='+', choices=('A', 'B', 'C'), default=[],
                   help='Keep selected pretrained ABC modules fixed while retaining the complete deployed method.')
    p.add_argument('--write-pair-calibration', action='store_true',
                   help='Calibrate net pause-versus-write score differences on writable candidates.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prefer-last-prefix', action='store_true',
                   help='For matched-query prefix controls, use the last supplied state for each sequence/query.')
    return p.parse_args()


def load_data(roots, partition, device, prefer_last_prefix=False):
    arrays, configs, names, jobs = [], [], set(), []
    for root in map(Path, roots):
        config = json.loads((root / 'config.json').read_text())
        receipt = json.loads((root / 'completion.json').read_text())
        assert config['partition'] == receipt['partition'] == partition
        assert receipt['completed'] and not receipt['decision_input_contains_future']
        with np.load(root / 'samples.npz') as archive:
            row = {key: torch.from_numpy(archive[key].copy()).to(device)
                   for key in DECISION_FIELDS + LABEL_FIELDS}
        assert len(row['valid']) == receipt['clips'] == config['clips']
        assert row['valid'][:, 0].any(1).all() and row['history_valid'][:, -1].all()
        assert row['instance_features'].shape[1:] == (7, 5, 2, 768)
        assert all(torch.isfinite(value).all() for value in row.values())
        rows = torch.arange(len(row['valid']), device=device)
        assert row['valid'][rows, 0, row['original_choice'].long()].all()
        assert torch.equal(row['action_valid'][..., 0], row['valid'])
        assert torch.equal(row['action_valid'][..., 1], row['valid'] & (row['raw_score'] > .84))
        arrays.append(row)
        configs.append(config)
        names.update(job['sequence'] for job in config['jobs'])
        jobs.extend((job['sequence'], job['query_frame'], config['prefix_policy']) for job in config['jobs'])
    pooled = {key: torch.cat([row[key] for row in arrays]) for key in arrays[0]}
    unique, seen = [], set()
    for index, job in enumerate(jobs):
        if job not in seen:
            unique.append(index)
            seen.add(job)
    if prefer_last_prefix:
        by_query = {(job[0], job[1]): index for index, job in enumerate(jobs)}
        unique = list(by_query.values())
    indices = torch.tensor(unique, device=device)
    return {key: value[indices] for key, value in pooled.items()}, configs, names, [jobs[i] for i in unique]


def forward(model, data):
    return model({key: data[key] for key in DECISION_FIELDS})


@torch.no_grad()
def evaluate(model, data, batch_size, threshold, details=False, search_supervision='oracle', action_ranking='reference',
             write_pair_calibration=False, write_verification='identity'):
    model.eval()
    results, total_loss = [], 0.
    for start in range(0, len(data['valid']), batch_size):
        batch = {key: value[start:start + batch_size] for key, value in data.items()}
        output = forward(model, batch)
        loss, _ = objective(model, output, batch, search_supervision, threshold, action_ranking,
                            write_pair_calibration, write_verification)
        chosen = select_actions(output, batch, threshold, write_verification)
        utility = action_utility(batch).flatten(1)
        current = batch['current_iou'].float()
        rows = torch.arange(len(current), device=current.device)
        keep = batch['original_choice'].long()
        selected_utility = utility.gather(1, chosen['flat_action'][:, None]).squeeze(1)
        keep_utility = utility[rows, keep * 2]
        selected_current = current[rows, chosen['region'], chosen['candidate']]
        keep_current = current[rows, 0, keep]
        original_recall = current[:, 0].masked_fill(~batch['valid'][:, 0], -1).max(1).values >= .5
        available = batch['valid'].clone()
        available[:, 1:] = False
        available[rows, chosen['searched_region']] = batch['valid'][rows, chosen['searched_region']] & chosen['search_triggered'][:, None]
        budget_recall = current.masked_fill(~available, -1).flatten(1).max(1).values >= .5
        results.append({'current': selected_current, 'c1_current': keep_current,
                        'utility_before_search_cost': selected_utility,
                        'utility': selected_utility - .01 * chosen['search_triggered'], 'c1_utility': keep_utility,
                        'flat_action': chosen['flat_action'], 'region': chosen['region'],
                        'candidate': chosen['candidate'], 'pause': chosen['pause'].bool(),
                        'searched_region': chosen['searched_region'], 'search_triggered': chosen['search_triggered'],
                        'changed_location': (chosen['region'] != 0) | (chosen['candidate'] != keep),
                        'rescued': (keep_current < .2) & (selected_current >= .5),
                        'harmed': (keep_current >= .5) & (selected_current < .2),
                        'failed': selected_current < .2, 'c1_failed': keep_current < .2,
                        'original_recall': original_recall, 'budget_recall': budget_recall,
                        'missing_candidate_found': ~original_recall & budget_recall,
                        'wrong_update_fraction': batch['wrong_update_fraction'][rows, chosen['region'], chosen['candidate'], chosen['pause']],
                        'c1_wrong_update_fraction': batch['wrong_update_fraction'][rows, 0, keep, 0]})
        total_loss += float(loss) * len(current)
    values = {key: torch.cat([row[key] for row in results]) for key in results[0]}
    metrics = {'clips': len(data['valid']), 'loss': total_loss / len(data['valid']),
               'valid_actions': int(data['action_valid'].sum())}
    for key, value in values.items():
        if key not in ('flat_action', 'region', 'candidate', 'searched_region'):
            metrics[key] = int(value.sum()) if value.dtype == torch.bool else float(value.mean())
    metrics['current_gain_against_c1'] = metrics['current'] - metrics['c1_current']
    metrics['utility_gain_against_c1'] = metrics['utility'] - metrics['c1_utility']
    assert all(np.isfinite(value) for value in metrics.values())
    if details:
        return metrics, {key: value.cpu().numpy() for key, value in values.items()}
    return metrics


def main():
    args = arguments()
    assert len(set(args.retain_epochs)) == len(args.retain_epochs) and all(1 <= epoch <= args.epochs for epoch in args.retain_epochs)
    assert args.epochs > 0 and args.batch_size > 0
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    train, train_configs, train_names, train_jobs = load_data(args.train, 'train', device, args.prefer_last_prefix)
    validation, val_configs, val_names, val_jobs = load_data([args.validation], 'validation', device)
    assert not train_names & val_names
    reference = train_configs[0]
    for config in train_configs + val_configs:
        for key in ('split', 'root', 'cache', 'head', 'pretrained', 'motion_run', 'max_prefix',
                    'regions', 'future_policy', 'future_horizon'):
            assert config[key] == reference[key], key
        assert config['head'] == args.c1_head
    split = json.loads(Path(reference['split']).read_text())
    assert train_names <= set(split['train']) and val_names <= set(split['validation'])
    c1 = torch.load(args.c1_head, map_location='cpu', weights_only=False)
    model = RecoverabilityModules(c1).to(device)
    initial_checkpoint_epoch = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        assert checkpoint['module'] == 'ABC_recoverability'
        model.load_state_dict(checkpoint['model'], strict=True)
        assert all(torch.equal(value.cpu(), c1['head'][key]) for key, value in model.c1.state_dict().items())
        initial_checkpoint_epoch = checkpoint['epoch']
    groups = {'A': model.memory, 'B': model.search, 'C': model.action}
    for name in args.frozen_modules:
        groups[name].requires_grad_(False)
    initial = {name: {key: value.detach().clone() for key, value in module.state_dict().items()}
               for name, module in groups.items()}
    model.eval()
    initial_matches_c1 = True
    with torch.no_grad():
        for data in (train, validation):
            for start in range(0, len(data['valid']), args.batch_size):
                batch = {key: value[start:start + args.batch_size] for key, value in data.items()}
                chosen = select_actions(forward(model, batch), batch, args.threshold, args.write_verification)
                initial_matches_c1 &= (torch.equal(chosen['flat_action'], batch['original_choice'].long() * 2)
                                       and not bool(chosen['search_triggered'].any()))
    if not args.init_checkpoint:
        assert initial_matches_c1
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'module': 'ABC_recoverability', 'train_clips': len(train_jobs),
                           'validation_clips': len(val_jobs), 'train_jobs': train_jobs, 'validation_jobs': val_jobs,
                           'source_configs': train_configs + val_configs,
                           'decision_fields': DECISION_FIELDS, 'initial_all_choices_match_c1': initial_matches_c1,
                           'initial_checkpoint_epoch': initial_checkpoint_epoch,
                           'base_GOLA': 'full pretrained frozen visual extractor used by collector',
                           'frozen_c1': True, 'trainable_parameters': {name: sum(p.numel() for p in module.parameters() if p.requires_grad) for name, module in groups.items()},
                           'checkpoint_selection': 'maximum held-out selected rollout utility minus .01 per triggered extra search; strict improvement',
                           'checkpoint_retention': 'Cached best.pth plus specified temporary fixed epochs pending full continuous validation; delete unselected weights after scoring',
                           'search_budget': 'original region plus at most one extra region; extra cost applies even if kept original candidate',
                           'scope': 'causal predicted-prefix TRAIN caches; frozen continuation policy recorded in source_configs; not complete online or official accuracy'}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    metrics = evaluate(model, validation, args.batch_size, args.threshold, search_supervision=args.search_supervision,
                       action_ranking=args.action_ranking, write_pair_calibration=args.write_pair_calibration,
                       write_verification=args.write_verification)
    records, best = [{'epoch': 0, **metrics}], metrics['utility']

    def save(epoch, metrics):
        torch.save({'module': 'ABC_recoverability', 'model': model.state_dict(),
                    'epoch': epoch, 'args': vars(args), 'validation': metrics,
                    'selection_policy': config['checkpoint_selection']}, out / 'best.pth')

    save(0, metrics)
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
                loss, parts = objective(model, output, batch, args.search_supervision, args.threshold, args.action_ranking,
                                        args.write_pair_calibration, args.write_verification)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = {}
                for name, module in groups.items():
                    if name in args.frozen_modules:
                        assert all(p.grad is None for p in module.parameters())
                        gradients[name] = 0.
                    else:
                        gradients[name] = float(torch.stack([p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]).sum().sqrt())
                    assert np.isfinite(gradients[name])
                    max_gradients[name] = max(max_gradients[name], gradients[name])
                norm = torch.nn.utils.clip_grad_norm_(parameters, 5.)
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
            metrics = evaluate(model, validation, args.batch_size, args.threshold, search_supervision=args.search_supervision,
                               action_ranking=args.action_ranking, write_pair_calibration=args.write_pair_calibration,
                               write_verification=args.write_verification)
            records.append({'epoch': epoch, **metrics})
            if epoch in args.retain_epochs:
                torch.save({'module': 'ABC_recoverability', 'model': model.state_dict(),
                            'epoch': epoch, 'args': vars(args), 'validation': metrics,
                            'selection_policy': 'Fixed epoch; pending full continuous developer validation, not cached best'},
                           out / f'epoch_{epoch:03d}.pth')
            if metrics['utility'] > best:
                best = metrics['utility']
                save(epoch, metrics)
            (out / 'metrics.json').write_text(json.dumps(records, indent=2))
            print('VALIDATION', json.dumps(records[-1]), flush=True)
    changed = {name: any(not torch.equal(initial[name][key], value) for key, value in module.state_dict().items())
               for name, module in groups.items()}
    assert all(changed[name] == (name not in args.frozen_modules) for name in groups)
    assert all((norm > 0) == (name not in args.frozen_modules) for name, norm in max_gradients.items())
    checkpoint = torch.load(out / 'best.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    best_metrics, values = evaluate(model, validation, args.batch_size, args.threshold, details=True,
                                    search_supervision=args.search_supervision, action_ranking=args.action_ranking,
                                    write_pair_calibration=args.write_pair_calibration,
                                    write_verification=args.write_verification)
    assert best_metrics == checkpoint['validation']
    np.savez_compressed(out / 'best_validation.npz', **values)
    receipt = {'completed': True, 'epochs': args.epochs, 'optimizer_steps': steps,
               'modules_changed': changed, 'max_module_gradient_norms': max_gradients,
               'frozen_modules': args.frozen_modules,
               'frozen_module_weights_unchanged': {name: not changed[name] for name in args.frozen_modules},
               'frozen_c1_gradients_absent': all(p.grad is None for p in model.c1.parameters()),
               'elapsed_seconds': time.perf_counter() - started,
               'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20,
               'best_epoch': checkpoint['epoch'], 'best_validation': best_metrics,
               'strict_reload_metrics_equal': True, 'last_validation': metrics,
               'retained_weights': ['best.pth'] + [f'epoch_{epoch:03d}.pth' for epoch in sorted(args.retain_epochs)], 'official_tracking_accuracy': False}
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
