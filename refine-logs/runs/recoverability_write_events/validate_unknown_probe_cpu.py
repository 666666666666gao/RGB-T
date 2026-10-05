"""Audit actual masked arrays and aggregate each TRAIN event exactly once."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.collect_rollouts import iou
from research.probe_geometry_commit import event_summary, outcome


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--phases', nargs='+', choices=['sanity', 'remaining'], required=True)
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    args = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert len(set(args.phases)) == len(args.phases)
    assert len(args.gpus) in (1, 4) and len(set(args.gpus)) == len(args.gpus)
    assert all(gpu in range(4) for gpu in args.gpus)
    rows = []
    for phase in args.phases:
        for gpu in args.gpus:
            folder = args.base / phase / f'gpu{gpu}'
            assert (folder / 'COMPLETE').is_file()
            config = json.loads((folder / 'config.json').read_text())
            saved = json.loads((folder / 'events.json').read_text())
            assert saved['status'] == 'COMPLETE_MATCHED_QUERY_COMMIT_ROLLOUTS'
            expected = 1 if phase == 'sanity' else (20 - len(args.gpus)) // len(args.gpus)
            assert saved['queries'] == len(saved['results']) == len(config['jobs']) == expected
            for index, (row, job) in enumerate(zip(saved['results'], config['jobs'])):
                assert (row['sequence'], row['query_frame'], row['event_id']) == (job['sequence'], job['query_frame'], job['event_id'])
                assert row['geometry_reference'] == 'C1_components' and row['query_visual_work_shared_once']
                arrays = np.load(folder / f'event_{index:04d}.npz')
                gt = arrays['gt_xyxy']
                known = np.isfinite(gt).all(1) & (gt[:, 2:] > gt[:, :2]).all(1)
                assert len(known) == 33 and np.array_equal(known, arrays['gt_known'])
                assert not known[:4].any() and known[1:].sum() >= 3
                assert [c['name'] for c in row['controls']] == ['raw', 'search_C1', 'motion_C1', 'search_motion_C1']
                for control in row['controls']:
                    name = control['name']
                    overlap, writes = arrays[name + '_iou'], arrays[name + '_writes']
                    assert np.array_equal(overlap >= 0, known) and (overlap[~known] == -1).all()
                    boxes = arrays[name + '_boxes_xyxy']
                    assert np.array_equal(overlap[known], np.asarray([iou(box, target) for box, target in zip(boxes[known], gt[known])]))
                    assert np.array_equal(arrays[name + '_boxes_xyxy'][0], arrays['raw_boxes_xyxy'][0])
                    assert control['query_output_iou'] is None and control['query_geometry_iou'] is None
                    assert control['query_pause'] == row['actual_policy_query_pause']
                    assert control['query_template_updated'] == row['controls'][0]['query_template_updated']
                    for horizon in (3, 32):
                        assert outcome(overlap[:horizon + 1], writes[:horizon + 1]) == control['horizons'][str(horizon)]
                    assert control['horizons']['3']['mean_future_iou'] is None
                    assert control['horizons']['3']['wrong_localization_writes'] == 0
                assert np.array_equal(arrays['motion_C1_search_xyxy'][0], arrays['raw_search_xyxy'][0])
                assert np.array_equal(arrays['search_C1_motion_xyxy'][0], arrays['raw_motion_xyxy'][0])
                rows.append(row)
    assert len({row['event_id'] for row in rows}) == len(rows)
    summary = event_summary(rows, [3, 32])
    assert summary['events'] == summary['query_states'] == len(rows)
    deltas = {}
    for name in ('search_C1', 'motion_C1', 'search_motion_C1'):
        values = np.asarray([next(c for c in row['controls'] if c['name'] == name)['horizons']['32']['mean_future_iou']
                             - row['controls'][0]['horizons']['32']['mean_future_iou'] for row in rows])
        rng = np.random.default_rng(42)
        sampled = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(1)
        deltas[name] = {'mean_future_iou_delta_percentage_points': float(values.mean() * 100),
                        'paired_event_95_ci': (np.quantile(sampled, [.025, .975]) * 100).tolist(),
                        'improved_events': int((values > 1e-12).sum()), 'worsened_events': int((values < -1e-12).sum()),
                        'tied_events': int((np.abs(values) <= 1e-12).sum())}
    result = {'status': 'PASS', 'phases': args.phases, 'gpus': args.gpus, 'events': len(rows), 'event_summary': summary,
              'H32_vs_raw': deltas, 'results': rows, 'new_optimizer_updates': 0,
              'actual_arrays_known_GT_masks_and_same_query_outputs_passed': True,
              'scope': 'Finite TRAIN event counterfactuals, unknown GT excluded; not trusted identity or native performance.'}
    target = args.base / ('_'.join(args.phases) + '_CPU.json')
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'events': len(rows), 'H32_vs_raw': deltas, 'report': str(target)}), flush=True)


if __name__ == '__main__':
    main()
