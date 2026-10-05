"""Check the new TRAIN transition-state caches against the actual dataset GT."""
import argparse
import ast
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


def read(path):
    return json.loads(Path(path).read_text())


def pairs(jobs):
    return {(job['sequence'], job['query_frame']) for job in jobs}


def overlaps(boxes, target):
    intersection = np.maximum(np.minimum(boxes[..., 2:], target[..., 2:]) -
                              np.maximum(boxes[..., :2], target[..., :2]), 0).prod(-1)
    areas = np.maximum(boxes[..., 2:] - boxes[..., :2], 0).prod(-1)
    return intersection / (areas + (target[..., 2:] - target[..., :2]).prod(-1) - intersection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('sanity', 'full'), required=True)
    args = parser.parse_args()
    root = Path('/data/gb/GOLA')
    assert Path.cwd() == root and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    folder = root / 'refine-logs/runs/recoverability_train_events'
    plan = read(folder / 'plan.json')
    base = read(Path(plan['base_train']) / 'config.json')
    schema = read(Path(plan['base_train']) / 'partition_cpu_acceptance.json')
    validation = read(Path(plan['validation']) / 'config.json')
    split = read(base['split'])
    assert schema['status'] == 'PASS' and base['clips'] == 1762 and validation['clips'] == 196
    assert len(split['train']) == 881 and len(split['validation']) == 98
    assert not set(split['train']) & set(split['validation'])
    assert {job['sequence'] for job in base['jobs']} == set(split['train'])
    assert {job['sequence'] for job in validation['jobs']} == set(split['validation'])
    source_jobs = read(root / plan['source_eligibility'])['new_event_jobs']
    full_jobs = [job for entry in plan['full'].values() for job in read(entry['jobs_file'])['jobs']]
    assert len(full_jobs) == len(pairs(full_jobs)) == plan['new_queries'] == 716
    assert pairs(full_jobs) == pairs(source_jobs) and not pairs(full_jobs) & pairs(base['jobs'])
    assert {job['sequence'] for job in full_jobs} <= set(split['train'])
    assert len(base['jobs']) + len(full_jobs) == plan['combined_train_queries'] == 2478

    tree = ast.parse((root / 'research/recoverability_modules.py').read_text())
    fields = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                  and any(isinstance(target, ast.Name) and target.id == 'DECISION_FIELDS'
                                          for target in node.targets)))
    assert not set(fields) & {'current_iou', 'future_iou', 'history_iou', 'wrong_update_fraction', 'action_valid'}
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(base['root'], base['cache'])
    sequences = {dataset[index].get_name(): dataset[index] for index in range(len(dataset))}
    entries, jobs = [], []
    common = ('root', 'cache', 'head', 'pretrained', 'motion_run', 'split', 'regions',
              'prefix_policy', 'prefix_execution', 'future_execution', 'history_descriptor_storage')
    for gpu in range(4):
        item = plan[args.phase][str(gpu)]
        output = Path(item['output'])
        assert (output / 'job_completed.txt').is_file()
        config, completed = read(output / 'config.json'), read(output / 'completion.json')
        expected = read(item['jobs_file'])['jobs']
        count = item['clips']
        assert config['jobs'] == expected and config['clips'] == completed['clips'] == len(expected) == count
        assert pairs(expected) <= pairs(read(plan['full'][str(gpu)]['jobs_file'])['jobs'])
        assert config['partition'] == completed['partition'] == 'train'
        assert completed['completed'] and not completed['decision_input_contains_future']
        assert not completed['official_tracking_accuracy']
        assert config['prefix_model'] == plan['teacher']
        assert config['prefix_checkpoint_epoch'] == config['future_checkpoint_epoch'] == plan['teacher_epoch'] == 4
        assert config['future_policy_mode'] == 'own' and config['prefix_write_verification'] == 'action'
        assert config['decision_inputs_copied_before_future_decode'] and config['history_capacity_covers_prefix']
        assert config['future_horizon'] == 3
        assert config['sampling'] == 'explicit TRAIN partition jobs; GT may choose supervision states, never online decisions'
        for key in common:
            assert config[key] == base[key], key
        for key in ('max_prefix', 'batch_clips', 'forward_batch', 'seed'):
            assert config[key] == plan[key] == base[key], key
        with np.load(output / 'samples.npz', allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        assert set(data) == set(schema['schema']) and set(fields) <= set(data)
        for key, value in data.items():
            expected_schema = schema['schema'][key]
            assert list(value.shape) == [count] + expected_schema['shape'][1:], key
            assert value.dtype.str == expected_schema['dtype'] and np.isfinite(value).all(), key
        assert np.array_equal(data['action_valid'][..., 0], data['valid'])
        assert np.array_equal(data['action_valid'][..., 1], data['valid'] & (data['raw_score'] > .84))
        assert data['valid'][np.arange(count), 0, data['original_choice']].all()
        assert int(data['action_valid'].sum()) == completed['valid_actions']
        for key, mask in [('current_iou', data['valid']), ('future_iou', data['action_valid']),
                          ('wrong_update_fraction', data['action_valid'])]:
            assert ((data[key][mask] >= 0) & (data[key][mask] <= 1)).all(), key
        current_error, history_error = 0., 0.
        for index, job in enumerate(expected):
            assert job['sequence'] in split['train'] and job['sequence'] not in split['validation']
            sequence, query = sequences[job['sequence']], job['query_frame']
            targets = np.asarray([sequence[t].get_bounding_box() for t in range(query, query + 4)])
            assert np.isfinite(targets).all() and (targets[:, 2:] > targets[:, :2]).all()
            measured = overlaps(data['image_boxes'][index].astype(np.float64), targets[0])
            error = float(np.abs(measured - data['current_iou'][index])[data['valid'][index]].max())
            assert error <= 1e-4, (job, error)
            current_error = max(current_error, error)
            past = data['history_valid'][index].astype(bool)
            frames = data['history_frames'][index][past]
            assert np.array_equal(frames, np.arange(max(0, query - plan['max_prefix']), query)), job
            past_targets = np.asarray([sequence[int(frame)].get_bounding_box() for frame in frames])
            gt_valid = np.isfinite(past_targets).all(-1) & (past_targets[:, 2:] > past_targets[:, :2]).all(-1)
            labels = data['history_iou'][index][past]
            assert (labels[~gt_valid] == -1).all()
            measured = overlaps(data['history_boxes'][index][past][gt_valid].astype(np.float64), past_targets[gt_valid])
            error = float(np.abs(measured - labels[gt_valid]).max())
            assert error <= 1e-4, (job, error)
            history_error = max(history_error, error)
        entries.append({'gpu': gpu, 'output': str(output), 'clips': count,
                        'current_GT_max_error': current_error, 'history_GT_max_error': history_error,
                        'completion': completed})
        jobs.extend(expected)
        print('SHARD_PASS', gpu, count, flush=True)
        del data
    expected_count = 4 if args.phase == 'sanity' else 716
    assert len(jobs) == len(pairs(jobs)) == expected_count
    if args.phase == 'full':
        assert jobs == full_jobs
    receipt = {'status': 'PASS', 'phase': args.phase, 'new_clips': expected_count, 'entries': entries,
               'all_queries_TRAIN_only': True, 'base1762_and_validation196_unchanged': True,
               'full_combined_train_queries': 2478, 'GT_current_and_all_valid_history_verified': True,
               'decision_inputs_copied_before_future': True, 'official_metrics_completed': False,
               'future_label_scope': 'Actual frozen own-policy outputs: source-linked, finite/range checked; no independent future neural rerun.',
               'execution': {'new_neural_forward_calls': 0, 'optimizer_steps': 0, 'native_test_reads': False},
               'completed_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
    (Path('/data/gb/setup') / ('train_events_' + args.phase + '_cpu_acceptance_20261005.json')).write_text(
        json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'phase': args.phase, 'new_clips': expected_count}), flush=True)


if __name__ == '__main__':
    main()
