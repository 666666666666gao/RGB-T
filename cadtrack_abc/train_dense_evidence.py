"""Calibrate R3 candidate evidence without changing A/B/visual coordinates.

Each relation context contains local5 or local5+one extra5. Candidate evidence
is fitted before collecting fresh consequence labels and final policy features.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .dense_submission import DenseCandidateEvidence, proxy_objective


class Evidence(Dataset):
    def __init__(self, root, split, teacher):
        self.root = Path(root)
        self.receipt = json.loads((self.root / 'collection_complete.json').read_text())
        assert self.receipt['completed'] and self.receipt['split'] == split
        assert self.receipt['checkpoint_sha256'] == teacher
        self.records = self.receipt['records']
        assert len(self.records) == self.receipt['queries'] and len(self.records) > 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        value = torch.load(self.root / self.records[index]['file'], map_location='cpu', weights_only=False)
        assert value['metadata']['checkpoint_sha256'] == self.receipt['checkpoint_sha256']
        return value


def collate_records(records):
    return records


def score_context(model, candidate, appearance, distribution):
    count = candidate['raw'].shape[1]
    assert count in (5, 10)
    encoded = []
    for start in range(0, count, 5):
        local = {key: value[:, start:start + 5] for key, value in candidate.items()}
        descriptors = {key: value[:, start:start + 5] for key, value in appearance.items()}
        encoded.append(model.encode(local, descriptors, distribution, float(start > 0)))
    combined = {key: torch.cat([item[key] for item in encoded], 1) for key in encoded[0]}
    return model.contextualize(candidate, combined)


def batch_objective(model, records, device):
    distribution = {key: torch.cat([r['distribution'][key] for r in records]).to(device)
        for key in records[0]['distribution']}
    loss = torch.zeros((), device=device)
    squared_error = torch.zeros((), device=device)
    candidates = 0
    for index in range(4):
        contexts = [record['contexts'][index] for record in records]
        candidate = {key: torch.cat([c['candidate'][key] for c in contexts]).to(device)
            for key in contexts[0]['candidate']}
        appearance = {key: torch.cat([c['appearance'][key] for c in contexts]).to(device)
            for key in contexts[0]['appearance']}
        quality = torch.stack([c['quality'] for c in contexts]).to(device)
        context = score_context(model, candidate, appearance, distribution)
        valid = torch.ones(len(records), dtype=torch.bool, device=device)
        loss = loss + proxy_objective(context, quality, valid) / 4
        squared_error = squared_error + (context['quality'].sigmoid() - quality).square().sum()
        candidates += quality.numel()
    return loss, squared_error.detach(), candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    parent_path = Path(args.parent).resolve()
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    assert parent['model_family'] == 'CADTrackDenseABC-R3' and parent['stage'] == 'dense_motion'
    assert not parent['config']['future_actions_trained'] and not parent['config']['search_actions_trained']
    training = Evidence(args.train, 'train', digest)
    validation = Evidence(args.validation, 'validation', digest)
    assert not {r['sequence'] for r in training.records} & {r['sequence'] for r in validation.records}
    model = DenseCandidateEvidence().cuda()
    model.load_state_dict(parent['dense_abc']['candidates'], strict=True)
    before = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(parent['config'], stage='evidence_calibration', evidence_parent=str(parent_path),
        evidence_parent_sha256=digest, evidence_epochs=args.epochs, evidence_batch=args.batch, evidence_lr=args.lr)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(training, batch_size=args.batch, shuffle=True,
        generator=generator, collate_fn=collate_records, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=args.batch, collate_fn=collate_records, num_workers=0)
    candidates, history = [], []
    updates = 0

    def save(epoch):
        path = output / f'candidate_epoch{epoch:02d}.pth'
        torch.save(dict(model_family=parent['model_family'], stage='evidence_calibration',
            dense_abc=dict(parent['dense_abc'], candidates=model.state_dict()),
            epoch=epoch, updates=updates, config=config), path)
        candidates.append(dict(epoch=epoch, checkpoint=str(path), updates=updates))

    def validate():
        model.eval()
        error, count = 0., 0
        with torch.no_grad():
            for records in validation_loader:
                _, squared, number = batch_objective(model, records, torch.device('cuda'))
                error += float(squared)
                count += number
        return error / count

    save(0)
    history.append(dict(epoch=0, cache_quality_mse=validate()))
    for epoch in range(1, args.epochs + 1):
        model.train()
        count, total = 0, 0.
        for records in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = batch_objective(model, records, torch.device('cuda'))
            assert bool(torch.isfinite(loss))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm))
            optimizer.step()
            updates += 1
            count += len(records)
            total += float(loss.detach()) * len(records)
        score = validate()
        history.append(dict(epoch=epoch, updates=updates, queries=count, mean_loss=total / count,
            cache_quality_mse=score, full_video_validation=False))
        save(epoch)
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1]), flush=True)
    assert any(not torch.equal(before[key], value.cpu()) for key, value in model.state_dict().items())
    selected = min(history, key=lambda record: record['cache_quality_mse'])
    selected_path = Path(candidates[selected['epoch']]['checkpoint'])
    payload = torch.load(selected_path, map_location='cpu', weights_only=False)
    clone = DenseCandidateEvidence()
    clone.load_state_dict(payload['dense_abc']['candidates'], strict=True)
    assert all(torch.equal(value, clone.state_dict()[key]) for key, value in payload['dense_abc']['candidates'].items())
    terminal = dict(status='DENSE_EVIDENCE_CALIBRATION_COMPLETE', epochs=args.epochs, updates=updates,
        selected_epoch=selected['epoch'], selected_checkpoint=str(selected_path),
        selected_checkpoint_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        candidates=candidates, cache_selection_only=True, initialization_is_training_gain=False,
        strict_candidate_reload=True, fresh_consequence_labels_required=True,
        full_ABC_training_complete=False, formal_metrics=None, goal_achieved=False)
    (output / 'fit_complete.json').write_text(json.dumps(terminal, indent=2))


if __name__ == '__main__':
    main()
