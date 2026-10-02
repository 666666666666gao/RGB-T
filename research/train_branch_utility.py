"""Fit C3 future utility on cached counterfactual TRAIN/held-out clips.

The frozen GOLA/C1 features are observations; future GT supplies labels only.
This prototype's held-out utility is not official benchmark tracking accuracy.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .branch_utility import BranchUtilityHead, utility_objective, utility_selection_scores


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True, help='Completed collect_rollouts TRAIN folder')
    p.add_argument('--validation', required=True, help='Completed collect_rollouts validation folder')
    p.add_argument('--c1-head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--rank-weight', type=float, default=1.)
    p.add_argument('--rank-gap', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_samples(root, partition, device):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    receipt = json.loads((root / 'completion.json').read_text())
    assert receipt['completed'] and receipt['partition'] == config['partition'] == partition
    assert not receipt['decision_input_contains_future']
    arrays = np.load(root / 'samples.npz')
    data = {key: torch.from_numpy(arrays[key]).to(device) for key in arrays.files}
    n = len(data['valid'])
    assert n == config['clips'] == receipt['clips'] and data['valid'].any(1).all()
    assert data['features'].shape == (n, data['valid'].shape[1], 768)
    assert data['evidence'].shape[-1] == 10 and data['motion'].shape[-1] == 9
    assert all(torch.isfinite(value).all() for value in data.values())
    assert data['valid'].gather(1, data['original_choice'].long().unsqueeze(1)).all()
    return data, config


def predict(head, batch):
    return head(batch['features'].float(), batch['evidence'].float(),
                batch['motion'].float(), batch['c1_quality'].float())


@torch.no_grad()
def evaluate(head, data):
    head.eval()
    prediction = predict(head, data)
    valid, target = data['valid'], data['utility'].float()
    scores = utility_selection_scores(prediction, data['evidence'].float())
    selected = scores.masked_fill(~valid, -torch.inf).argmax(1)
    quality = target.gather(1, selected.unsqueeze(1)).squeeze(1)
    c1 = target.gather(1, data['original_choice'].long().unsqueeze(1)).squeeze(1)
    oracle = target.masked_fill(~valid, -torch.inf).max(1).values
    assert torch.isfinite(prediction).all()
    return {'clips': len(selected), 'valid_actions': int(valid.sum()),
            'utility_mse': float(((prediction - target)[valid] ** 2).mean()),
            'selected_mean_utility': float(quality.mean()), 'c1_mean_utility': float(c1.mean()),
            'oracle_mean_utility': float(oracle.mean()), 'mean_regret': float((oracle - quality).mean()),
            'mean_gain_against_c1': float((quality - c1).mean()),
            'better_than_c1_clips': int((quality > c1 + .01).sum()),
            'worse_than_c1_clips': int((quality < c1 - .01).sum()),
            'reselected_clips': int((selected != data['original_choice']).sum()),
            'selected_current_iou': float(data['current_iou'].gather(1, selected.unsqueeze(1)).mean())}


def main():
    args = arguments()
    assert args.epochs > 0 and args.batch_size > 0
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    train, train_config = load_samples(args.train, 'train', device)
    validation, val_config = load_samples(args.validation, 'validation', device)
    train_names = {job['sequence'] for job in train_config['jobs']}
    val_names = {job['sequence'] for job in val_config['jobs']}
    assert not train_names & val_names
    assert train_config['split'] == val_config['split']
    assert train_config['head'] == val_config['head'] == args.c1_head
    for key in ('horizon', 'pollution_weight', 'pretrained', 'future_policy', 'history'):
        assert train_config[key] == val_config[key], key
    checkpoint = torch.load(args.c1_head, map_location='cpu', weights_only=False)
    assert checkpoint['module'] == 'C1_candidate_quality'
    head = BranchUtilityHead(checkpoint['args']['hidden']).to(device)
    head.projection.load_state_dict({k[len('projection.'):]: v for k, v in checkpoint['head'].items()
                                    if k.startswith('projection.')}, strict=True)
    with torch.no_grad():
        for data in (train, validation):
            scores = utility_selection_scores(predict(head, data), data['evidence'].float())
            choices = scores.masked_fill(~data['valid'], -torch.inf).argmax(1)
            assert torch.equal(choices, data['original_choice'].long()), 'Zero-residual policy differs from C1'
    initial = {key: value.detach().clone() for key, value in head.state_dict().items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'module': 'C3_short_rollout_utility',
                           'scope': 'cached counterfactual TRAIN/held-out utility, not benchmark tracking',
                           'train_sequence_count': len(train_names), 'validation_sequence_count': len(val_names),
                           'train_clips': len(train['valid']), 'validation_clips': len(validation['valid']),
                           'teacher_training_config': train_config, 'teacher_validation_config': val_config,
                           'base_trainable_parameters': 0,
                           'new_trainable_parameters': sum(p.numel() for p in head.parameters()),
                           'selection_for_utility_evaluation': 'C1 Hann policy with learned utility correction',
                           'window_penalty': .45,
                           'initial_selection_matches_c1_all_clips': True,
                           'ranking_loss_uses_selection_scores': True,
                           'checkpoint_selection': 'minimum validation mean regret; official test is excluded'}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    torch.save({key: value.cpu() for key, value in initial.items()}, out / 'initial_head.pth')
    metrics = evaluate(head, validation)
    records = [{'epoch': 0, 'train_loss': None, **metrics}]
    best = metrics['mean_regret']
    def save_checkpoint(epoch, metrics):
        torch.save({'module': config['module'], 'head': head.state_dict(),
                    'optimizer': optimizer.state_dict(), 'epoch': epoch,
                    'selection_policy': config['selection_for_utility_evaluation'],
                    'args': vars(args), 'hidden': checkpoint['args']['hidden'],
                    'validation': metrics, 'teacher_training_config': train_config}, out / 'best.pth')
    save_checkpoint(0, metrics)
    started, optimizer_steps = time.perf_counter(), 0
    with (out / 'train.jsonl').open('w') as stream:
        for epoch in range(1, args.epochs + 1):
            head.train()
            loss_sum = 0.
            order = torch.randperm(len(train['valid']), device=device)
            for indices in order.split(args.batch_size):
                batch = {key: value[indices] for key, value in train.items()}
                prediction = predict(head, batch)
                scores = utility_selection_scores(prediction, batch['evidence'].float())
                loss = utility_objective(prediction, batch['utility'].float(),
                                         batch['valid'], scores, args.rank_weight, args.rank_gap)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 5.))
                assert np.isfinite(gradient_norm)
                optimizer.step()
                optimizer_steps += 1
                loss_sum += float(loss.detach()) * len(indices)
            metrics = evaluate(head, validation)
            record = {'epoch': epoch, 'train_loss': loss_sum / len(order), **metrics}
            records.append(record)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            print('EPOCH', json.dumps(record), flush=True)
            if metrics['mean_regret'] < best:
                best = metrics['mean_regret']
                save_checkpoint(epoch, metrics)
            (out / 'metrics.json').write_text(json.dumps(records, indent=2))
    changed = any(not torch.equal(value, head.state_dict()[key]) for key, value in initial.items())
    assert changed
    with (out / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    receipt = {'completed': True, 'optimizer_steps': optimizer_steps, 'epochs': args.epochs,
               'new_parameters_changed': changed, 'base_trainable_parameters': 0,
               'best_validation_mean_regret': best, 'last_validation': metrics,
               'elapsed_seconds': time.perf_counter() - started,
               'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20,
               'official_tracking_accuracy': False}
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
