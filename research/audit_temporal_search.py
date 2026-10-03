"""TRAIN-only candidate failures and causal motion-proposal geometry, no new tracker."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .temporal_modules import TemporalModules
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider


def crop_bounds(boxes):
    output = np.array((224, 224))
    bounds = []
    for box in boxes:
        provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
        provider.initialize(box)
        scale, translation = provider.get(output)
        bounds.append(np.concatenate((-translation / scale, (output - translation) / scale)))
    return np.asarray(bounds)


def center_inside(centers, bounds):
    return ((centers >= bounds[..., :2]) & (centers <= bounds[..., 2:])).all(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    training = Path(args.training_run)
    assert (training / 'training_and_audit_completed.txt').is_file()
    config = json.loads((training / 'config.json').read_text())
    checkpoint = torch.load(training / 'last.pth', map_location='cpu', weights_only=False)
    c1 = torch.load(config['c1_head'], map_location='cpu', weights_only=False)
    model = TemporalModules(c1, config['slots'], config['motion_history'], config['modes'], checkpoint['horizon'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    counts = {name: sum(p.numel() for p in module.parameters())
              for name, module in (('frozen_C1', model.c1), ('A', model.memory), ('B', model.motion), ('C', model.selector))}
    report = {'completed': True, 'official_tracking_accuracy': False, 'checkpoint_epoch': checkpoint['epoch'],
              'training_parameters': counts,
              'protocol': {
                  'data': 'completed original TRAIN/validation caches; original disjoint sequence split',
                  'proposal': 'query-step three learned means; translated original-size crop; no visual inference',
                  'crop': 'native factor4/minsize10 provider parameters before image resampling adjustment',
                  'coverage': 'GT center only; not bbox coverage, candidate recall or tracking accuracy',
                  'budget': 'baseline plus all three proposals uses four regions; summed region areas count overlap',
                  'causal': 'A/B receive only anchor and past predicted observations; GT labels used after forward',
                  'execution': 'CPU-only read-only inference; no optimizer or tracker/output changes'}, 'caches': []}
    fields = ('anchor_features', 'history_descriptors', 'history_evidence', 'history_quality',
              'history_boxes', 'history_frames', 'history_write', 'history_valid')
    roots = config['train'] + [config['validation']]
    for root in map(Path, roots):
        source = json.loads((root / 'config.json').read_text())
        receipt = json.loads((root / 'completion.json').read_text())
        assert receipt['completed']
        split = json.loads(Path(source['split']).read_text())
        assert not set(split['train']) & set(split['validation'])
        pairs = [(job['sequence'], job['query_frame']) for job in source['jobs']]
        assert all(name in split[source['partition']] for name, _ in pairs)
        with np.load(root / 'samples.npz') as archive:
            data = {key: archive[key] for key in fields}
            labels = {key: archive[key] for key in ('current_iou', 'motion_targets', 'valid', 'original_choice', 'history_iou')}
        n = len(pairs)
        assert n == receipt['clips'] and all(len(array) == n and np.isfinite(array).all() for array in data.values())
        assert data['history_valid'][:, -1].all()
        references, means, mode_choices = [], [], []
        with torch.inference_mode():
            for start in range(0, n, args.batch_size):
                batch = {key: torch.from_numpy(array[start:start + args.batch_size]) for key, array in data.items()}
                anchor, memory, _ = model.history_memory(batch)
                distribution = model.motion(batch['history_boxes'].float(), batch['history_frames'],
                                            batch['history_valid'], batch['history_quality'].float(), anchor, memory)
                references.append(distribution['reference'].numpy())
                means.append(distribution['means'][:, :, 0].numpy())
                mode_choices.append(distribution['log_weights'].argmax(-1).numpy())
        reference, means, selected_modes = map(np.concatenate, (references, means, mode_choices))
        assert np.isfinite(means).all() and np.array_equal(reference, data['history_boxes'][:, -1])
        reference, means = reference.astype(np.float64), means.astype(np.float64)
        center = (reference[:, :2] + reference[:, 2:]) / 2
        size = np.maximum(reference[:, 2:] - reference[:, :2], 10)
        proposed_center = center[:, None] + means[..., :2] * size[:, None]
        # Keep physical crop size unchanged: only use the predicted motion center.
        proposal_boxes = np.concatenate((proposed_center - size[:, None] / 2,
                                         proposed_center + size[:, None] / 2), -1)
        original_bounds = crop_bounds(reference)
        proposal_bounds = crop_bounds(proposal_boxes.reshape(-1, 4)).reshape(n, config['modes'], 4)
        query_gt = labels['motion_targets'][:, 0]
        assert np.isfinite(query_gt).all() and (query_gt[:, 2:] > query_gt[:, :2]).all()
        gt_center = (query_gt[:, :2] + query_gt[:, 2:]) / 2
        original_inside = center_inside(gt_center, original_bounds)
        proposal_inside = center_inside(gt_center[:, None], proposal_bounds)
        selected_inside = proposal_inside[np.arange(n), selected_modes]
        union_inside = original_inside | proposal_inside.any(1)
        valid = labels['valid']
        selected_iou = labels['current_iou'][np.arange(n), labels['original_choice']]
        oracle = np.where(valid, labels['current_iou'], -1).max(1)
        failed = selected_iou < .2
        missing = failed & (oracle < .5)
        known = data['history_valid'] & (labels['history_iou'] >= 0)
        written = data['history_write'] & known
        original_area = (original_bounds[:, 2:] - original_bounds[:, :2]).prod(-1)
        proposal_area = (proposal_bounds[..., 2:] - proposal_bounds[..., :2]).prod(-1).sum(1)
        assert np.allclose(proposal_area, original_area * config['modes'], rtol=1e-4)
        row = {'cache': str(root), 'partition': source['partition'], 'sampled_clips': n,
               'unique_sequence_query_pairs': len(set(pairs)),
               'c1_failed_queries': int(failed.sum()), 'failed_with_correct_candidate': int((failed & ~missing).sum()),
               'failed_without_correct_candidate': int(missing.sum()),
               'target_center_outside_native_original_crop': int((~original_inside).sum()),
               'missing_candidate_and_center_outside_original': int((missing & ~original_inside).sum()),
               'missing_and_outside_original_but_in_top_probability_motion_crop': int((missing & ~original_inside & selected_inside).sum()),
               'missing_and_outside_original_but_in_any_motion_crop': int((missing & ~original_inside & proposal_inside.any(1)).sum()),
               'mean_original_crop_area_pixels': float(original_area.mean()),
               'mean_original_plus_three_crop_area_pixels': float((original_area + proposal_area).mean()),
               'original_center_coverage': float(original_inside.mean()),
               'original_plus_three_center_coverage': float(union_inside.mean()),
               'current_iou': float(selected_iou.mean()), 'oracle_iou': float(oracle.mean()),
               'history_known_observations': int(known.sum()), 'history_known_updates': int(written.sum()),
               'history_wrong_updates': int((written & (labels['history_iou'] < .2)).sum())}
        report['caches'].append(row)
        print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
