"""Train only C1 candidate quality/ranking on LasHeR training sequences.

Run with: python -m research.train_candidate --output /data/gb/outputs/c1_seed42
This is a supervised candidate diagnostic, not official end-to-end PR/SR.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .candidate_learning import (CandidateTrainingData, FrozenCandidateExtractor,
                                 CandidateQualityHead, candidate_objective, selection_scores)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    p.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    p.add_argument('--pretrained', default='pretrained_models/gola_b224.bin')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--steps-per-epoch', type=int, default=512)
    p.add_argument('--validation-steps', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--candidates', type=int, default=5)
    p.add_argument('--window-penalty', type=float, default=.45)
    p.add_argument('--nms-iou', type=float, default=.7)
    p.add_argument('--max-gap', type=int, default=10, help='Maximum gap in valid historical observations.')
    p.add_argument('--contamination-probability', type=float, default=.3)
    p.add_argument('--translation-jitter', type=float, default=1.)
    p.add_argument('--rank-weight', type=float, default=1.)
    p.add_argument('--rank-gap', type=float, default=.1)
    p.add_argument('--validation-fraction', type=float, default=.1)
    p.add_argument('--recall-threshold', type=float, default=.5)
    p.add_argument('--failure-threshold', type=float, default=.2)
    p.add_argument('--log-every', type=int, default=20)
    return p.parse_args()


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(extractor, head, loader, device, args, optimizer=None, log_file=None, epoch=0):
    training = optimizer is not None
    head.train(training)
    sums = dict(loss=0., candidate_count=0., baseline_iou=0., learned_iou=0., oracle_iou=0.,
                recall=0., rejected_correct=0., rescued=0., false_switches=0., baseline_correct=0.,
                failed=0., samples=0.)
    started = time.perf_counter()
    gradient_norm = None
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader, 1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                candidates = extractor(batch)
                logits = head(candidates)
            loss, quality = candidate_objective(logits.float(), candidates, batch['gt'], args.rank_weight, args.rank_gap)
            assert torch.isfinite(loss), f'Non-finite loss at epoch {epoch}, step {step}'
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 5.))
                assert np.isfinite(gradient_norm), gradient_norm
                assert all(p.grad is None for p in extractor.parameters()), 'Frozen tracker received gradients'
                optimizer.step()
            with torch.no_grad():
                scores = selection_scores(logits.float(), candidates, args.window_penalty)
                winner = scores.masked_fill(~candidates['valid'], -torch.inf).argmax(1)
                learned = quality.gather(1, winner.unsqueeze(1)).squeeze(1)
                oracle = quality.masked_fill(~candidates['valid'], -torch.inf).max(1).values
                baseline = quality[:, 0]
                failed = baseline < args.failure_threshold
                correct = baseline >= args.recall_threshold
                rejected_correct = failed & (oracle >= args.recall_threshold)
                size = len(baseline)
                sums['loss'] += float(loss.detach()) * size
                sums['candidate_count'] += float(candidates['valid'].sum())
                sums['baseline_iou'] += float(baseline.sum())
                sums['learned_iou'] += float(learned.sum())
                sums['oracle_iou'] += float(oracle.sum())
                sums['recall'] += int((oracle >= args.recall_threshold).sum())
                sums['failed'] += int(failed.sum())
                sums['rejected_correct'] += int(rejected_correct.sum())
                sums['rescued'] += int((rejected_correct & (learned >= args.recall_threshold)).sum())
                sums['false_switches'] += int((correct & (learned < args.failure_threshold)).sum())
                sums['baseline_correct'] += int(correct.sum())
                sums['samples'] += size
            if training and (step == 1 or step % args.log_every == 0 or step == len(loader)):
                record = {'epoch': epoch, 'step': step, 'steps': len(loader),
                          'loss': float(loss.detach()), 'gradient_norm': gradient_norm,
                          'elapsed_seconds': time.perf_counter() - started,
                          'samples': sums['samples'], 'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20}
                print('TRAIN', json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + '\n')
                log_file.flush()
    n = sums['samples']
    assert n > 0
    metrics = {k: sums[k] / n for k in ('loss', 'candidate_count', 'baseline_iou', 'learned_iou', 'oracle_iou', 'recall')}
    metrics.update({k: sums[k] for k in ('samples', 'failed', 'rejected_correct', 'rescued', 'false_switches', 'baseline_correct')})
    metrics['rejected_recovery_rate'] = sums['rescued'] / sums['rejected_correct'] if sums['rejected_correct'] else None
    metrics['false_switch_rate'] = sums['false_switches'] / sums['baseline_correct'] if sums['baseline_correct'] else None
    metrics['elapsed_seconds'] = time.perf_counter() - started
    metrics['last_gradient_norm'] = gradient_norm
    return metrics


def main():
    args = arguments()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extractor = FrozenCandidateExtractor(args.pretrained, args.candidates, args.window_penalty, args.nms_iou).to(device)
    head = CandidateQualityHead(args.hidden).to(device)
    assert all(not p.requires_grad for p in extractor.parameters())
    assert all(p.requires_grad for p in head.parameters())
    initial_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
    initial_head_cpu = {k: v.cpu() for k, v in initial_head.items()}
    torch.save(initial_head_cpu, out / 'initial_head.pth')
    from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped as Dataset
    data = Dataset.load(args.root, args.cache)
    indices = np.random.default_rng(args.seed).permutation(len(data)).tolist()
    val_count = max(1, round(len(indices) * args.validation_fraction))
    val_indices, train_indices = indices[:val_count], indices[val_count:]
    split = {'seed': args.seed, 'train': [data[i].get_name() for i in train_indices],
             'validation': [data[i].get_name() for i in val_indices],
             'source': 'LasHeR training split ONLY; official test and RGBT234 are excluded'}
    assert not set(split['train']) & set(split['validation'])
    (out / 'split.json').write_text(json.dumps(split, indent=2))
    common = (args.root, args.cache)
    train = CandidateTrainingData(*common, train_indices, args.steps_per_epoch * args.batch_size,
                                 args.seed, args.max_gap, args.contamination_probability, args.translation_jitter)
    val = CandidateTrainingData(*common, val_indices, args.validation_steps * args.batch_size,
                               args.seed + 100000, args.max_gap, args.contamination_probability, args.translation_jitter)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                          shuffle=False, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train, **loader_options)
    val_loader = torch.utils.data.DataLoader(val, **loader_options)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = vars(args) | {'scope': 'C1 current candidate quality/ranking ONLY; no A/B, rollout value or online recovery claims',
                           'pretrained_load': extractor.load_receipt,
                           'base_trainable_parameters': sum(p.numel() for p in extractor.parameters() if p.requires_grad),
                           'new_trainable_parameters': sum(p.numel() for p in head.parameters() if p.requires_grad),
                           'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device),
                           'eligible_train_sequences': len(train.sequence_indices),
                           'eligible_validation_sequences': len(val.sequence_indices)}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    print('INITIALIZED', json.dumps({k: config[k] for k in ('scope', 'base_trainable_parameters', 'new_trainable_parameters', 'gpu', 'eligible_train_sequences', 'eligible_validation_sequences')}), flush=True)
    records = []
    # Fixed held-out crops and candidates permit a meaningful before/after check.
    metrics = run_epoch(extractor, head, val_loader, device, args)
    records.append({'epoch': 0, 'stage': 'before_training', **metrics})
    print('VALIDATION_BEFORE', json.dumps(metrics), flush=True)
    (out / 'metrics.json').write_text(json.dumps(records, indent=2))
    best = metrics['learned_iou']
    torch.save({'module': 'C1_candidate_quality', 'head': head.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': 0, 'args': vars(args), 'validation': metrics}, out / 'best.pth')
    with (out / 'train.jsonl').open('w') as log_file:
        for epoch in range(1, args.epochs + 1):
            train.epoch = epoch
            train_metrics = run_epoch(extractor, head, train_loader, device, args, optimizer, log_file, epoch)
            records.append({'epoch': epoch, 'stage': 'train', **train_metrics})
            metrics = run_epoch(extractor, head, val_loader, device, args)
            records.append({'epoch': epoch, 'stage': 'validation', **metrics})
            print('VALIDATION', json.dumps({'epoch': epoch, **metrics}), flush=True)
            checkpoint = {'module': 'C1_candidate_quality', 'head': head.state_dict(),
                          'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': vars(args), 'validation': metrics}
            torch.save(checkpoint, out / f'epoch_{epoch:02d}.pth')
            if metrics['learned_iou'] > best:
                best = metrics['learned_iou']
                torch.save(checkpoint, out / 'best.pth')
            (out / 'metrics.json').write_text(json.dumps(records, indent=2))
    changed = any(not torch.equal(initial_head[k], v) for k, v in head.state_dict().items())
    assert changed, 'New module parameters did not update'
    receipt = {'completed': True, 'new_parameters_changed': changed,
               'base_requires_grad': False, 'base_gradients_absent': all(p.grad is None for p in extractor.parameters()),
               'epochs': args.epochs, 'last_validation': metrics,
               'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20,
               'official_tracking_accuracy_measured': False}
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2))
    with (out / 'metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print('COMPLETED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
