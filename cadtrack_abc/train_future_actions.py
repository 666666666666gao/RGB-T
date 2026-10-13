"""Full matched future-C fitting; scheduled candidates use full-video selection later."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .counterfactual import ACTION_MODES
from .future_actions import FutureActionHead, objective
from .future_data import FuturePackets, collate_packets


def batch_loss(head, batch, device):
    losses = []
    for features, baseline, utility in batch['contexts']:
        features, baseline, utility = features.to(device), baseline.to(device), utility.to(device)
        losses.append(objective(head(features, baseline), utility, baseline)[0])
    return torch.stack(losses).mean()


@torch.no_grad()
def cached_validation(head, loader, device, threshold):
    head.eval()
    gain, queries, interventions = 0., 0, 0
    for batch in loader:
        outcomes = []
        changed = []
        for features, baseline, utility in batch['contexts']:
            features, baseline, utility = features.to(device), baseline.to(device), utility.to(device)
            scores = head(features, baseline)['scores'].flatten(1)
            value, best = scores.max(1)
            action = torch.where(value > threshold, best, baseline * len(ACTION_MODES))
            outcomes.append(utility.flatten(1).gather(1, action[:, None]).squeeze(1))
            changed.append(action != baseline * len(ACTION_MODES))
        rows = torch.arange(len(outcomes[0]), device=device)
        observed = batch['observed_context'].to(device)
        chosen = torch.stack(outcomes, 1)[rows, observed]
        gain += float((chosen - batch['reference_utility'].to(device)).sum())
        interventions += int(torch.stack(changed, 1)[rows, observed].sum())
        queries += len(rows)
    return {'cached_mean_utility_gain_vs_teacher': gain / queries, 'queries': queries,
            'interventions': interventions, 'formal_metrics': None,
            'scope': 'Frozen prefix/continuation labels only; not continuous-video or final checkpoint selection.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--teacher', required=True, help='Selected immutable native or future teacher checkpoint.')
    parser.add_argument('--teacher-policy', choices=['native', 'future_actions'], default='native')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--train-index', required=True)
    parser.add_argument('--validation-index')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--save-epochs', nargs='+', type=int, default=[0, 1, 2, 5, 10, 20])
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    torch.set_num_threads(4)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    teacher_sha = hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest()
    teacher = torch.load(args.teacher, map_location='cpu', weights_only=False)
    assert teacher['config']['pretrained'] == config['pretrained'] and teacher['config']['threshold'] == config['threshold']
    train_data = FuturePackets(args.train_index, manifest, 'train', teacher_sha, config['threshold'], args.teacher_policy)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              collate_fn=collate_packets)
    val_loader = None
    if not args.sanity:
        assert train_data.names == set(manifest['train_sequences']) and len(train_data.names) == 881
        assert args.validation_index
        val_data = FuturePackets(args.validation_index, manifest, 'validation', teacher_sha, config['threshold'], args.teacher_policy)
        assert val_data.names == set(manifest['validation_sequences']) and len(val_data.names) == 98
        val_loader = DataLoader(val_data, batch_size=args.batch, num_workers=args.workers, collate_fn=collate_packets)
    device = torch.device(args.device)
    head = FutureActionHead().to(device)
    # A future teacher already has this head: continue its actual policy rather
    # than changing the initialization back to the native zero head.
    if args.teacher_policy == 'future_actions':
        assert teacher['policy_family'] == 'future_actions'
        head.load_state_dict(teacher['future_action'], strict=True)
    else:
        assert 'future_action' not in teacher
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'fit_complete.json').exists()
    before = {name: p.detach().clone() for name, p in head.named_parameters()}
    epochs, updates, candidates, history = (2 if args.sanity else args.epochs), 0, [], []
    started = time.perf_counter()

    def save_candidate(epoch):
        path = output / ('candidate_epoch' + str(epoch).zfill(2) + '.pth')
        payload = {'policy_family': 'future_actions', 'abc': teacher['abc'], 'future_action': head.state_dict(),
                   'epoch': epoch, 'updates': updates, 'teacher_epoch': teacher['epoch'], 'teacher_sha256': teacher_sha,
                   'config': config, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                   'selection': 'Pending complete98-development-video comparison; cache is diagnostic only.'}
        torch.save(payload, path)
        candidates.append({'epoch': epoch, 'checkpoint': str(path)})

    save_candidate(0)
    for epoch in range(1, epochs + 1):
        if args.sanity and updates >= 2:
            break
        head.train()
        total, examples = 0., 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(head, batch, device)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            updates += 1
            size = len(batch['reference_utility'])
            total += float(loss.detach()) * size; examples += size
            if args.sanity and updates == 2:
                break
        scheduler.step()
        record = {'epoch': epoch, 'updates': updates, 'train_loss': total / examples, 'formal_metrics': None}
        if val_loader is not None:
            record['cached_validation'] = cached_validation(head, val_loader, device, config['threshold'])
        history.append(record)
        print(json.dumps(record), flush=True)
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        if epoch in args.save_epochs or epoch == epochs or (args.sanity and updates == 2):
            save_candidate(epoch)
    changed = [name for name, p in head.named_parameters() if not torch.equal(before[name], p.detach())]
    assert changed
    loaded = torch.load(candidates[-1]['checkpoint'], map_location=device, weights_only=False)
    reloaded = FutureActionHead().to(device)
    reloaded.load_state_dict(loaded['future_action'], strict=True)
    assert all(torch.equal(head.state_dict()[name], reloaded.state_dict()[name]) for name in head.state_dict())
    terminal = {'status': 'FUTURE_HEAD_SANITY_COMPLETE' if args.sanity else 'FUTURE_HEAD_FIT_COMPLETE',
                'epochs': len(history), 'updates': updates, 'teacher_sha256': teacher_sha, 'train_queries': len(train_data),
                'train_sequences': len(train_data.names), 'candidates': candidates, 'changed_head_tensors': changed,
                'strict_head_reload': True, 'full_video_selection_complete': False, 'formal_metrics_complete': False,
                'seconds': time.perf_counter() - started}
    (output / 'fit_complete.json').write_text(json.dumps(terminal, indent=2))
    print(json.dumps(terminal), flush=True)


if __name__ == '__main__':
    main()
