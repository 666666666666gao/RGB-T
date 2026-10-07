"""Paired identity-feature probe; not a native tracking benchmark.

Complete pretrained GOLA and C1 stay frozen. Each query has identical crops,
five candidates and C1 scores for the two descriptor sources. The encoded
RGB/TIR streams have interacted through joint attention, so they are not
independent sensor-reliability measurements. Sampling uses supervised frame
pairs, including the existing template perturbation, not deployed prefixes.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .candidate_learning import CandidateTrainingData, CandidateQualityHead, box_iou, selection_scores
from .collect_recoverability import InstanceExtractor, instance_pool
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


class PairedInstanceExtractor(InstanceExtractor):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.normalized_tokens = None
        self.norm_hook = self.base.norm.register_forward_hook(self.capture_tokens)

    def capture_tokens(self, module, inputs, output):
        self.normalized_tokens = output.detach()

    @torch.no_grad()
    def extract_with_output(self, batch):
        candidates, output = super().extract_with_output(batch)
        z_len, x_len = 64, 256
        tokens = self.normalized_tokens
        assert tokens.shape[1:] == (768, 768)
        candidates['encoded_instance_features'] = torch.stack([
            instance_pool(tokens[:, start:start + x_len], candidates['boxes'])
            for start in (z_len, 2 * z_len + x_len)], dim=2)
        weights = batch['z_feat_mask'].flatten(1).float()
        candidates['encoded_template_features'] = torch.stack([
            (tokens[:, start:start + z_len].float() * weights[..., None]).sum(1)
            / weights.sum(1, keepdim=True).clamp(min=1)
            for start in (0, z_len + x_len)], dim=1)
        return candidates, output


class FixedSequencePairs(Dataset):
    def __init__(self, root, cache, indices, views, seed, shard, shards):
        self.base = CandidateTrainingData(root, cache, indices, len(indices) * views,
                                          seed, 100, .3, 1.)
        assert set(self.base.sequence_indices) == set(indices)
        self.jobs = [(index, view, order * views + view)
                     for order, index in enumerate(indices) if order % shards == shard
                     for view in range(views)]

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, item):
        index, view, sample = self.jobs[item]
        # DataLoader workers each own their dataset; this fixes sequence coverage
        # while reusing the existing crop/valid-frame/perturbation sampler.
        self.base.sequence_indices = [index]
        batch = self.base[sample]
        batch.update(sequence_index=index, view=view, sample_index=sample)
        return batch


def projection_from_parent(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU()).to(device)
    prefix = 'memory.projection.'
    state = {key[len(prefix):]: value for key, value in checkpoint['model'].items() if key.startswith(prefix)}
    projection.load_state_dict(state, strict=True)
    return projection, checkpoint['epoch']


def first_context(z, mask):
    return dict(z=z, d=z, x=F.interpolate(z, size=(224, 224), mode='bilinear', align_corners=False),
                z_feat_mask=mask, d_feat_mask=mask)


@torch.inference_mode()
def check(args):
    device = torch.device('cuda:0')
    paired = PairedInstanceExtractor(args.pretrained).to(device)
    baseline = InstanceExtractor(args.pretrained).to(device)
    assert all(torch.equal(value, baseline.base.state_dict()[key]) for key, value in paired.base.state_dict().items())
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    wanted = set(json.loads(Path(args.split).read_text())['train'])
    index = next(i for i in range(len(dataset)) if dataset[i].get_name() in wanted)
    pairs = FixedSequencePairs(args.root, args.cache, [index], 2, args.seed, 0, 1)
    batch = next(iter(DataLoader(pairs, batch_size=2)))
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.autocast('cuda', dtype=torch.float16):
        original, old_output = baseline.extract_with_output(batch)
        augmented, new_output = paired.extract_with_output(batch)
        changed_gt = dict(batch, gt=torch.zeros_like(batch['gt']))
        label_changed, _ = paired.extract_with_output(changed_gt)
        initial, _ = paired.extract_with_output(first_context(batch['z'], batch['z_feat_mask']))
    assert all(torch.equal(value, augmented[key]) for key, value in original.items())
    assert all(torch.equal(value, new_output[key]) for key, value in old_output.items())
    assert all(torch.equal(value, label_changed[key]) for key, value in augmented.items())
    assert augmented['encoded_instance_features'].shape == (2, 5, 2, 768)
    assert initial['encoded_template_features'].shape == (2, 2, 768)
    assert all(torch.isfinite(value).all() for value in augmented.values())
    assert all(p.grad is None for p in paired.base.parameters())
    output = Path(args.output)
    assert not output.exists()
    output.mkdir(parents=True)
    receipt = dict(completed=True, original_candidate_fields_exact=True, original_localization_output_exact=True,
                   GT_label_perturbation_does_not_change_features=True, encoded_shapes_and_finite=True,
                   pretrained_base_state_exact=True, frozen_gradients_absent=True, optimizer_updates=0,
                   identity_only_probe=True, native_tracking_accuracy=False,
                   peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20)
    (output / 'completion.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


def similarity(projection, anchor, instances):
    a = F.normalize(projection(anchor.float()), dim=-1)
    c = F.normalize(projection(instances.float()), dim=-1)
    return (c * a[:, None]).sum(-1).mean(-1)


@torch.inference_mode()
def extract(args):
    device = torch.device('cuda:0')
    extractor = PairedInstanceExtractor(args.pretrained).to(device).eval()
    c1 = torch.load(args.c1, map_location='cpu', weights_only=False)
    head = CandidateQualityHead(c1['args']['hidden']).to(device).eval()
    head.load_state_dict(c1['head'], strict=True)
    head.requires_grad_(False)
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    split = json.loads(Path(args.split).read_text())
    assert not set(split['train']) & set(split['validation'])
    wanted = set(split[args.partition])
    indices = [i for i in range(len(dataset)) if dataset[i].get_name() in wanted]
    assert {dataset[i].get_name() for i in indices} == wanted
    if args.limit_sequences:
        indices = indices[:args.limit_sequences]
    pairs = FixedSequencePairs(args.root, args.cache, indices, args.views, args.seed, args.shard, args.shards)
    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    output = Path(args.output)
    assert not output.exists()
    output.mkdir(parents=True)
    names = {i: dataset[i].get_name() for i in indices}
    config = dict(vars(args), expected_samples=len(pairs), sequences=len({job[0] for job in pairs.jobs}),
                  sequence_names=names, scope=__doc__, pretrained_load=extractor.load_receipt,
                  identity_sources=['pre_attention_roi', 'jointly_attended_roi'],
                  candidate_visual_forwards_per_query=1, optimizer_updates=0,
                  anchor_initialization='First-template self-context: d=z, x=bilinearly resized z; cached once per sequence',
                  first_context_not_current_query=True, native_tracking_accuracy=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    anchors, rows, anchor_forward_calls = {}, [], 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        identifiers = batch['sequence_index'].tolist()
        missing = sorted(set(identifiers) - set(anchors))
        if missing:
            positions = torch.tensor([identifiers.index(index) for index in missing], device=device)
            z, mask = batch['z'][positions], batch['z_feat_mask'][positions]
            initial = first_context(z, mask)
            with torch.autocast('cuda', dtype=torch.float16):
                first, _ = extractor.extract_with_output(initial)
            for row, identifier in enumerate(missing):
                anchors[identifier] = (first['anchor_features'][row].half(),
                                       first['encoded_template_features'][row].half())
            anchor_forward_calls += 1
        with torch.autocast('cuda', dtype=torch.float16):
            candidates, _ = extractor.extract_with_output(batch)
            scores = selection_scores(head(candidates).float(), candidates, extractor.window_penalty)
        quality = box_iou(candidates['boxes'], batch['gt'][:, None])
        arrays = dict(sequence_index=batch['sequence_index'], sample_index=batch['sample_index'],
                      contaminated=batch['contaminated'], valid=candidates['valid'],
                      boxes=candidates['boxes'], quality=quality, c1_score=scores,
                      pre_anchor=torch.stack([anchors[i][0] for i in identifiers]),
                      encoded_anchor=torch.stack([anchors[i][1] for i in identifiers]),
                      pre_instances=candidates['instance_features'].half(),
                      encoded_instances=candidates['encoded_instance_features'].half())
        assert all(torch.isfinite(value).all() for value in arrays.values())
        rows.append({key: value.cpu().numpy() for key, value in arrays.items()})
        progress = dict(completed_queries=sum(len(row['valid']) for row in rows), expected_queries=len(pairs),
                        elapsed_seconds=time.perf_counter() - started,
                        peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20)
        (output / 'progress.json').write_text(json.dumps(progress))
        if batch_index % 10 == 0:
            print('PROGRESS', json.dumps(progress), flush=True)
    merged = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    assert len(merged['valid']) == len(pairs)
    np.savez_compressed(output / 'samples.npz', **merged)
    done = dict(progress, completed=True, optimizer_updates=0, actual_sequences=len(anchors),
                actual_anchor_initialization_forward_calls=anchor_forward_calls,
                actual_anchor_initialization_images=len(anchors), native_tracking_accuracy=False,
                frozen_base_gradients_absent=all(p.grad is None for p in extractor.base.parameters()))
    (output / 'completion.json').write_text(json.dumps(done, indent=2))
    print('COMPLETED', json.dumps(done), flush=True)


def paired_loss(score, quality, valid):
    positive = valid & (quality >= .5)
    negative = valid & (quality < .2)
    pairs = positive[:, :, None] & negative[:, None, :]
    delta = score[:, :, None] - score[:, None, :]
    return F.softplus(-delta[pairs] / .1).mean(), int(pairs.sum())


@torch.no_grad()
def evaluate_projection(projection, data, source, batch_size, alpha):
    output = []
    for start in range(0, len(data['valid']), batch_size):
        batch = {key: value[start:start + batch_size] for key, value in data.items()}
        score = similarity(projection, batch[source + '_anchor'], batch[source + '_instances'])
        keep = batch['c1_score'].masked_fill(~batch['valid'], -torch.inf).argmax(-1)
        residual = score - score.gather(1, keep[:, None])
        chosen = (batch['c1_score'] + alpha * residual).masked_fill(~batch['valid'], -torch.inf).argmax(-1)
        output.append(batch['quality'].gather(1, chosen[:, None]).squeeze(1))
    selected = torch.cat(output)
    keep = data['c1_score'].masked_fill(~data['valid'], -torch.inf).argmax(-1)
    c1 = data['quality'].gather(1, keep[:, None]).squeeze(1)
    return dict(selected_query_mean_iou=float(selected.mean()), c1_query_mean_iou=float(c1.mean()),
                rescued=int(((c1 < .2) & (selected >= .5)).sum()),
                harmed=int(((c1 >= .5) & (selected < .2)).sum())), selected.cpu().numpy()


def fit(args):
    device = torch.device('cuda:0')
    data = {}
    for partition in ('train', 'validation'):
        rows = []
        for path in args.inputs:
            root = Path(path) / partition
            assert json.loads((root / 'completion.json').read_text())['completed']
            with np.load(root / 'samples.npz') as archive:
                rows.append({key: archive[key].copy() for key in archive.files})
        merged = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
        assert len(np.unique(merged['sample_index'])) == len(merged['sample_index'])
        data[partition] = {key: torch.from_numpy(value).to(device) for key, value in merged.items()}
    assert not set(data['train']['sequence_index'].tolist()) & set(data['validation']['sequence_index'].tolist())
    projection, initial_epoch = projection_from_parent(args.parent, device)
    initial = {key: value.detach().clone() for key, value in projection.state_dict().items()}
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.output)
    assert not output.exists()
    output.mkdir(parents=True)
    (output / 'config.json').write_text(json.dumps(dict(vars(args), source_parent_epoch=initial_epoch,
        identity_only_pretraining=True, native_tracking_accuracy=False, train_queries=len(data['train']['valid']),
        validation_queries=len(data['validation']['valid'])), indent=2))
    history, best, steps, max_gradient = [], -float('inf'), 0, 0.
    started = time.perf_counter()
    for epoch in range(args.epochs + 1):
        epoch_loss, epoch_pairs = 0., 0
        if epoch:
            order = torch.randperm(len(data['train']['valid']), device=device)
            for indices in order.split(args.batch_size):
                batch = {key: value[indices] for key, value in data['train'].items()}
                has_pair = ((batch['quality'] >= .5) & batch['valid']).any(1) & ((batch['quality'] < .2) & batch['valid']).any(1)
                # This condition is defined by actual positive/negative supervision,
                # not an online target-presence decision.
                if not bool(has_pair.any()):
                    continue
                score = similarity(projection, batch[args.source + '_anchor'], batch[args.source + '_instances'])
                loss, pairs = paired_loss(score, batch['quality'], batch['valid'])
                assert pairs > 0 and torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.))
                assert np.isfinite(norm)
                max_gradient = max(max_gradient, norm)
                optimizer.step()
                steps += 1
                epoch_loss += float(loss.detach()) * pairs
                epoch_pairs += pairs
        metrics, selected = evaluate_projection(projection, data['validation'], args.source, args.batch_size, args.alpha)
        row = dict(epoch=epoch, optimizer_steps=steps, elapsed_seconds=time.perf_counter() - started,
                   training_pair_loss=epoch_loss / epoch_pairs if epoch_pairs else None,
                   training_positive_negative_pairs=epoch_pairs, **metrics)
        history.append(row)
        if metrics['selected_query_mean_iou'] > best:
            best = metrics['selected_query_mean_iou']
            torch.save(dict(projection=projection.state_dict(), epoch=epoch, source=args.source, args=vars(args),
                            validation_metrics=metrics), output / 'best.pth')
            np.save(output / 'best_selected_iou.npy', selected)
        print('EPOCH', json.dumps(row), flush=True)
        (output / 'metrics.json').write_text(json.dumps(history, indent=2))
    assert steps > 0 and max_gradient > 0
    changed = any(not torch.equal(value, initial[key]) for key, value in projection.state_dict().items())
    assert changed
    checkpoint = torch.load(output / 'best.pth', map_location=device, weights_only=False)
    projection.load_state_dict(checkpoint['projection'], strict=True)
    loaded, selected = evaluate_projection(projection, data['validation'], args.source, args.batch_size, args.alpha)
    assert loaded == checkpoint['validation_metrics']
    assert np.array_equal(selected, np.load(output / 'best_selected_iou.npy'))
    done = dict(completed=True, epochs=args.epochs, optimizer_steps=steps, max_gradient_norm=max_gradient,
                parameters_changed=True, best_epoch=checkpoint['epoch'], strict_reload_metrics_equal=True,
                validation_metrics=loaded, identity_only_pretraining=True, native_tracking_accuracy=False,
                elapsed_seconds=time.perf_counter() - started,
                peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20)
    (output / 'completion.json').write_text(json.dumps(done, indent=2))
    print('COMPLETED', json.dumps(done), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('check', 'extract', 'fit'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    parser.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    parser.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json')
    parser.add_argument('--pretrained', default='/data/gb/GOLA/pretrained_models/gola_b224.bin')
    parser.add_argument('--c1', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    parser.add_argument('--parent', default='/data/gb/outputs/post_search_relation_control_20261007/fit_full/one_way_lr5/best.pth')
    parser.add_argument('--partition', choices=('train', 'validation'), default='train')
    parser.add_argument('--views', type=int, default=16)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--limit-sequences', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--inputs', nargs='*')
    parser.add_argument('--source', choices=('pre', 'encoded'), default='pre')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    if args.stage == 'check':
        check(args)
    elif args.stage == 'extract':
        extract(args)
    else:
        fit(args)


if __name__ == '__main__':
    main()
