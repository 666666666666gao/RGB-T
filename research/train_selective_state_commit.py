"""Train C action advantages on event-weighted real matched consequences."""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .selective_state_commit import FEATURES, SelectiveStateCommitHead, choose_action


def load_partition(root, partition, horizon, penalty, device, reward='mean_all'):
    assert reward in ('mean_all', 'current_future')
    records, values = [], []
    for shard in sorted((root / 'full').glob('gpu*')):
        report = json.loads((shard / 'summary.json').read_text())
        assert (shard / 'COMPLETE').is_file() and report['status'] == 'COMPLETE_MATCHED_STATE_COMMIT_LABELS'
        for index, record in enumerate(report['results']):
            if record['partition'] != partition:
                continue
            if record['actual_rollout_horizon'] < horizon:
                continue  # Real 27/32-frame videos have H3 labels only.
            with np.load(shard / f'query_{index:05d}.npz') as data:
                known = data['gt_known'][:horizon + 1]
                if not known[1:].any():
                    continue  # Unknown future is not a zero-IoU training label.
                overlaps = data['iou'][:, :horizon + 1][:, known]
                mean = overlaps.mean(1)
                current = data['iou'][:, 0]
                future = data['iou'][:, 1:horizon + 1][:, known[1:]]
                utility = (current + future.mean(1) - penalty * (future < .2).mean(1)
                           if reward == 'current_future' else mean - penalty * (overlaps < .2).mean(1))
                legal = data['legal'].reshape(-1)
                assert np.isfinite(utility[legal]).all()
                values.append((data['features'].copy(), data['legal'].copy(), data['keep'].copy(), utility, mean,
                               current, future.mean(1)))
                records.append(record)
    assert records
    counts = Counter(r['actual_event_id'] for r in records)
    weights = np.array([1 / counts[r['actual_event_id']] for r in records], dtype=np.float32)
    weights /= weights.mean()
    fields = ('features', 'legal', 'keep', 'utility', 'mean_iou', 'current_iou', 'future_iou')
    data = {key: torch.as_tensor(np.stack([v[i] for v in values]), device=device)
            for i, key in enumerate(fields)}
    data['features'] = data['features'].float()
    data['utility'], data['mean_iou'] = data['utility'].float(), data['mean_iou'].float()
    data['weights'] = torch.as_tensor(weights, device=device)
    return data, records


@torch.no_grad()
def evaluate(head, data, components=False):
    scores = head(data['features'])
    keep, rows = data['keep'].long(), torch.arange(len(scores), device=scores.device)
    choices = choose_action(scores, data['legal'], torch.zeros_like(keep), keep.bool())
    selected = data['utility'][rows, choices]
    baseline = data['utility'][rows, keep]
    overlap = data['mean_iou'][rows, choices]
    base_overlap = data['mean_iou'][rows, keep]
    w = data['weights']
    metrics = {'event_weighted_utility': float((selected * w).sum() / w.sum()),
            'event_weighted_utility_advantage': float(((selected - baseline) * w).sum() / w.sum()),
            'event_weighted_mean_iou': float((overlap * w).sum() / w.sum()),
            'changed_queries': int((choices != keep).sum()), 'harmful_future_queries': int((overlap < base_overlap - 1e-8).sum()),
            'beneficial_future_queries': int((overlap > base_overlap + 1e-8).sum()),
            'geometry_holds': int(((choices != keep) & (choices % 3 == 2)).sum())}
    if components:
        for key in ('current_iou', 'future_iou'):
            advantage = data[key][rows, choices] - data[key][rows, keep]
            metrics['event_weighted_' + key + '_advantage'] = float((advantage * w).sum() / w.sum())
    return metrics, choices


def loss(head, batch):
    scores = head(batch['features']).flatten(1)
    rows, keep = torch.arange(len(scores), device=scores.device), batch['keep'].long()
    prediction = scores - scores[rows, keep][:, None]
    target = batch['utility'] - batch['utility'][rows, keep][:, None]
    valid = batch['legal'].flatten(1)
    mse = ((prediction - target).square() * valid).sum(1) / valid.sum(1)
    better = valid[:, :, None] & valid[:, None, :] & ((target[:, :, None] - target[:, None, :]) > .03)
    gap = (target[:, :, None] - target[:, None, :]).clamp(min=0)
    ranking = F.relu(.03 - (prediction[:, :, None] - prediction[:, None, :])) * better * gap
    rank = ranking.sum((1, 2)) / better.sum((1, 2)).clamp(min=1)
    return ((mse + rank) * batch['weights']).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--horizon', type=int, choices=(3, 32), required=True)
    p.add_argument('--lost-penalty', type=float, choices=(0., .1), required=True)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--reward', choices=('mean_all', 'current_future'), default='mean_all')
    p.add_argument('--head-architecture', choices=('mlp', 'linear'), default='mlp')
    p.add_argument('--sanity', action='store_true')
    args = p.parse_args()
    root = Path(args.collection)
    assert json.loads((root / 'collection_complete.json').read_text())['complete']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    train, train_records = load_partition(root, 'train', args.horizon, args.lost_penalty, device, args.reward)
    valid, val_records = load_partition(root, 'validation', args.horizon, args.lost_penalty, device, args.reward)
    corpus_train_queries = len(train['keep'])
    if args.sanity:
        # Real collected states can have only the keep action, hence zero loss.
        # The backward sanity must exercise actual differing TRAIN consequences.
        legal = train['legal'].flatten(1)
        spread = (train['utility'].masked_fill(~legal, -torch.inf).max(1).values
                  - train['utility'].masked_fill(~legal, torch.inf).min(1).values)
        indices = torch.nonzero(spread > 1e-6).flatten()[:8]
        assert len(indices) == 8, 'Need eight TRAIN states with distinct legal utility labels for backward sanity'
        train = {key: value[indices] for key, value in train.items()}
        train_records = [train_records[i] for i in indices.cpu().tolist()]
    train_names, val_names = {r['sequence'] for r in train_records}, {r['sequence'] for r in val_records}
    assert not train_names & val_names
    head = SelectiveStateCommitHead(args.head_architecture).to(device)
    original = {key: value.detach().clone() for key, value in head.state_dict().items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.0001)
    components = args.reward == 'current_future'
    m0, choices = evaluate(head, valid, components)
    assert torch.equal(choices, valid['keep']) and m0['changed_queries'] == 0
    module = ('selective_state_commit' if args.reward == 'mean_all' and args.head_architecture == 'mlp'
              else 'selective_state_commit_refined')
    config = vars(args) | {'module': module, 'features': FEATURES, 'threshold': .03,
        'train_queries_with_known_future': len(train['keep']), 'train_sequences_with_known_future': len(train_names),
        'full_corpus_train_queries_with_known_future': corpus_train_queries,
        'validation_queries': len(valid['keep']), 'validation_sequences': len(val_names),
        'event_weighting': 'Inverse query count within actual event, normalized to global mean1; equal event mass',
        'trained_parameters': sum(p.numel() for p in head.parameters()), 'frozen_A_B_C_parent': True,
        'scope': 'C extension trained on causal protected-old4/gross TRAIN states; future continuation frozen parent',
        'required_optimizer_updates': args.epochs * math.ceil(len(train['keep']) / args.batch)}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    best, updates, grads, started = m0['event_weighted_utility'], 0, set(), time.perf_counter()
    parent = json.loads((root / 'full/gpu0/config.json').read_text())['model']

    def save(epoch, metrics, filename='best.pth'):
        torch.save({'module': module, 'architecture': args.head_architecture,
                    'reward': args.reward, 'features': FEATURES, 'threshold': .03,
                    'parent_model': parent, 'search_value': 'gross', 'epoch': epoch,
                    'args': vars(args), 'head': head.state_dict(), 'validation': metrics}, out / filename)

    save(0, m0)
    history = [{'epoch': 0, 'optimizer_updates': 0, 'validation': m0}]
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train['keep']), device=device)
        losses = []
        for indices in order.split(args.batch):
            optimizer.zero_grad(set_to_none=True)
            objective = loss(head, {key: value[indices] for key, value in train.items()})
            assert torch.isfinite(objective)
            objective.backward()
            for name, parameter in head.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                if parameter.grad.abs().sum() > 0:
                    grads.add(name)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
            optimizer.step()
            updates += 1
            losses.append(float(objective))
        metrics, _ = evaluate(head, valid, components)
        if metrics['event_weighted_utility'] > best:
            best = metrics['event_weighted_utility']
            save(epoch, metrics)
        record = {'epoch': epoch, 'optimizer_updates': updates, 'loss': sum(losses) / len(losses),
                  'validation': metrics, 'elapsed_seconds': time.perf_counter() - started}
        history.append(record)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
    changed = [key for key, value in head.state_dict().items() if not torch.equal(value, original[key])]
    assert updates == config['required_optimizer_updates'] and changed and grads
    if not args.sanity:
        save(args.epochs, metrics, 'last.pth')
        endpoint = torch.load(out / 'last.pth', map_location=device, weights_only=False)
        endpoint_head = SelectiveStateCommitHead(args.head_architecture).to(device)
        endpoint_head.load_state_dict(endpoint['head'], strict=True)
        endpoint_metrics, _ = evaluate(endpoint_head, valid, components)
        assert endpoint['epoch'] == args.epochs and endpoint_metrics == metrics
    saved = torch.load(out / 'best.pth', map_location=device, weights_only=False)
    reloaded = SelectiveStateCommitHead(args.head_architecture).to(device)
    reloaded.load_state_dict(saved['head'], strict=True)
    reload_metrics, _ = evaluate(reloaded, valid, components)
    assert reload_metrics == saved['validation']
    result = {'status': 'COMPLETE_FULL_C_EXTENSION_TRAINING', 'epochs': args.epochs, 'optimizer_updates': updates,
              'best_epoch': saved['epoch'], 'best_validation': saved['validation'], 'M0_parent_exact': True,
              'actual_gradient_parameters': sorted(grads), 'actual_changed_parameters': changed, 'strict_reload_pass': True,
              'last_epoch': args.epochs if not args.sanity else None, 'last_strict_reload_pass': not args.sanity,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'elapsed_seconds': time.perf_counter() - started}
    (out / 'training_complete.json').write_text(json.dumps(result, indent=2))
    (out / 'COMPLETE').write_text('COMPLETE\n')


if __name__ == '__main__':
    main()
