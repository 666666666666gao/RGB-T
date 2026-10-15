"""Fit R3 future/search decisions on fresh, matched native consequences.

Visual/A/motion/candidate representations remain fixed. Four observed contexts
are separate selector examples; search sees only the local causal summary.
Cached validation is diagnostic. Final selection requires full98 video runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dense_submission import DenseDecisionHeads, consequence_objective, search_consequence_objective
from .search_actions import selected_action


class Consequences(Dataset):
    def __init__(self, root, split, teacher):
        self.root = Path(root).resolve()
        self.receipt = json.loads((self.root / 'collection_complete.json').read_text())
        assert self.receipt['completed'] and self.receipt['split'] == split
        assert self.receipt['model_family'] == 'CADTrackDenseABC-R3'
        assert self.receipt['checkpoint_sha256'] == teacher
        self.records = self.receipt['records']
        assert len(self.records) == self.receipt['queries']

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = torch.load(self.root / self.records[index]['file'], map_location='cpu', weights_only=False)
        assert record['metadata']['checkpoint_sha256'] == self.receipt['checkpoint_sha256']
        assert record['actual_native_replay_equal'] and record['actions'] == 210
        assert [context['features'].shape for context in record['contexts']] == [
            torch.Size([1, 5, 275]), *[torch.Size([1, 10, 275])] * 3]
        return record


def collate_records(records):
    return records


def evaluate_batch(model, records, device):
    """Selector-realized search targets, with future harm kept separate."""
    loss = torch.zeros((), device=device)
    realized = {name: [] for name in ('utility', 'future_mean_iou', 'future_lost_fraction', 'current_iou')}
    for index in range(4):
        context = {name: torch.stack([record['contexts'][index][name] for record in records]).to(device)
            for name in ('utility', 'future_mean_iou', 'future_lost_fraction', 'current_iou')}
        features = torch.cat([record['contexts'][index]['features'] for record in records]).to(device)
        baseline = torch.zeros(len(records), dtype=torch.long, device=device)
        prediction = model.future_action(features, baseline)
        loss = loss + consequence_objective(prediction, context['utility'], context['future_mean_iou'],
            context['future_lost_fraction'], baseline) / 4
        selected = selected_action(prediction, baseline, model.threshold)
        for name, value in context.items():
            realized[name].append(value.flatten(1).gather(1, selected[:, None]).squeeze(1))
    realized = {name: torch.stack(values, 1) for name, values in realized.items()}
    summary = torch.cat([record['search_input'] for record in records]).to(device)
    search = model.search_action(summary)
    loss = loss + search_consequence_objective(search, realized['utility'].detach(),
        realized['future_mean_iou'].detach(), realized['future_lost_fraction'].detach())
    score, rank = search['scores'].max(1)
    region = torch.where(score > model.threshold + .01, rank + 1, 0)
    rows = torch.arange(len(records), device=device)
    reference = torch.stack([record['contexts'][0]['utility'][0, 0] for record in records]).to(device)
    current = torch.stack([record['contexts'][0]['current_iou'][0, 0] for record in records]).to(device)
    actual = {name: value[rows, region] for name, value in realized.items()}
    diagnostics = dict(utility=actual['utility'], delta_utility=actual['utility'] - reference,
        current_iou=actual['current_iou'], delta_current_iou=actual['current_iou'] - current,
        future_mean_iou=actual['future_mean_iou'], future_lost_fraction=actual['future_lost_fraction'],
        extra_search=region.gt(0).float())
    return loss, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    parent_path = Path(args.parent).resolve()
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    assert parent['model_family'] == 'CADTrackDenseABC-R3'
    train = Consequences(args.train, 'train', parent_sha)
    validation = Consequences(args.validation, 'validation', parent_sha)
    train_sequences = {record['sequence'] for record in train.records}
    validation_sequences = {record['sequence'] for record in validation.records}
    assert train_sequences and validation_sequences and not train_sequences & validation_sequences
    model = DenseDecisionHeads(parent['config']['intervention_threshold']).cuda()
    model.load_state_dict(parent['dense_abc']['decisions'], strict=True)
    before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epochs = 1 if args.probe else args.epochs
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(parent['config'], stage='consequence_policy',
        consequence_parent=str(parent_path), consequence_parent_sha256=parent_sha,
        consequence_epochs=epochs, consequence_batch=args.batch, consequence_lr=args.lr,
        consequence_seed=42, consequence_probe=args.probe)
    generator = torch.Generator().manual_seed(42)
    training_loader = DataLoader(train, batch_size=args.batch, shuffle=True,
        generator=generator, collate_fn=collate_records, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=args.batch,
        collate_fn=collate_records, num_workers=0)
    manifest = dict(config=config, parent_checkpoint_sha256=parent_sha,
        train_queries=len(train), validation_queries=len(validation), train_sequences=sorted(train_sequences),
        validation_sequences=sorted(validation_sequences),
        trainable_parameters={name: value.numel() for name, value in model.named_parameters()},
        frozen='All parent localizer/A/motion/grid/candidate components; cached features carry no gradient.',
        selection='Finish prescribed training, then full98 continuous videos. Cache validation is diagnostic only.',
        context_rule='One local context or one local+extra context; search uses only local summary.',
        full_ABC_training_complete=False, formal_metrics=None)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    history, candidates = [], []
    updates = 0

    def save(epoch):
        path = output / f'candidate_epoch{epoch:02d}.pth'
        weights = dict(parent['dense_abc'], decisions=model.state_dict())
        checkpoint_config = dict(config,
            future_actions_trained=config['future_actions_trained'] or updates > 0,
            search_actions_trained=config['search_actions_trained'] or updates > 0)
        torch.save(dict(model_family=parent['model_family'], stage='consequence_policy',
            dense_abc=weights, epoch=epoch, updates=updates, config=checkpoint_config), path)
        candidates.append(dict(epoch=epoch, updates=updates, checkpoint=str(path)))

    save(0)
    for epoch in range(1, epochs + 1):
        model.train()
        count = 0
        train_loss = 0.
        for records in training_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = evaluate_batch(model, records, torch.device('cuda'))
            assert bool(torch.isfinite(loss)), (epoch, updates)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm))
            optimizer.step()
            updates += 1
            count += len(records)
            train_loss += float(loss.detach()) * len(records)
            if args.probe and updates == 2:
                break
        totals = {}
        evaluated = 0
        model.eval()
        with torch.no_grad():
            for records in validation_loader:
                _, diagnostics = evaluate_batch(model, records, torch.device('cuda'))
                for name, values in diagnostics.items():
                    totals[name] = totals.get(name, 0.) + float(values.sum())
                evaluated += len(records)
        record = dict(epoch=epoch, updates=updates, train_queries_seen=count, mean_loss=train_loss / count,
            cache_validation={name: value / evaluated for name, value in totals.items()},
            cache_validation_queries=evaluated, full_video_validation=False, formal_metrics=None)
        history.append(record)
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        if epoch in (1, 2, 5, 10, 20):
            save(epoch)
    changed = {group: sum(not torch.equal(before[name], value.detach().cpu())
        for name, value in model.named_parameters() if name.startswith(group))
        for group in ('future_action.', 'search_action.')}
    assert all(changed.values()), changed
    if candidates[-1]['epoch'] != epochs:
        save(epochs)
    reloaded = torch.load(candidates[-1]['checkpoint'], map_location='cpu', weights_only=False)
    clone = DenseDecisionHeads(model.threshold)
    clone.load_state_dict(reloaded['dense_abc']['decisions'], strict=True)
    assert all(torch.equal(value.cpu(), clone.state_dict()[name]) for name, value in model.state_dict().items())
    terminal = dict(status='DENSE_CONSEQUENCE_PROBE_COMPLETE' if args.probe else 'DENSE_CONSEQUENCE_TRAINING_COMPLETE',
        epochs=epochs, updates=updates, candidates=candidates, changed_parameter_tensors=changed,
        strict_decision_reload=True, actual_native_consequence_labels=True,
        future_actions_trained=True, search_actions_trained=True,
        full_ABC_training_complete=False, full98_selection_complete=False,
        scope='Decision stage completed; full main-method acceptance still needs raw-stage qualification and complete video evaluation.',
        formal_metrics=None, goal_achieved=False)
    (output / 'fit_complete.json').write_text(json.dumps(terminal, indent=2))
    print(json.dumps(terminal), flush=True)


if __name__ == '__main__':
    main()
