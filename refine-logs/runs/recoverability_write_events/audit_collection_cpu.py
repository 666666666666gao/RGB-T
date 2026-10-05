"""Verify 98 event representatives against TRAIN GT and count actual write pairs."""
import ast
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_write_events'
OUTPUT = Path('/data/gb/outputs/recoverability_write_events_collection_20261005')


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
    assert Path.cwd() == ROOT and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    plan = read(FOLDER / 'prepared_write_episode_queries.json')
    prior = read(ROOT / 'refine-logs/runs/recoverability_train_events/plan.json')
    base = read(Path(plan['base_train']) / 'config.json')
    schema = read(Path(plan['base_train']) / 'partition_cpu_acceptance.json')
    validation = read(Path(prior['validation']) / 'config.json')
    split = read(base['split'])
    assert schema['status'] == 'PASS' and base['clips'] == 1762 and validation['clips'] == 196
    assert len(split['train']) == 881 and len(split['validation']) == 98
    assert not set(split['train']) & set(split['validation'])
    assert {job['sequence'] for job in base['jobs']} == set(split['train'])
    assert {job['sequence'] for job in validation['jobs']} == set(split['validation'])
    full_jobs = [job for gpu in range(4) for job in read(FOLDER / f'gpu{gpu}_jobs.json')['jobs']]
    prior_jobs = base['jobs'] + [job for entry in prior['full'].values() for job in read(entry['jobs_file'])['jobs']]
    assert len(prior_jobs) == len(pairs(prior_jobs)) == 2478
    assert len(full_jobs) == len(pairs(full_jobs)) == plan['new_episode_queries'] == 98
    assert pairs(full_jobs) == pairs(plan['eligible']) and not pairs(full_jobs) & pairs(prior_jobs)
    assert len({job['event_id'] for job in full_jobs}) == 98
    assert {job['sequence'] for job in full_jobs} <= set(split['train'])
    tree = ast.parse((ROOT / 'research/recoverability_modules.py').read_text())
    fields = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                  and any(isinstance(target, ast.Name) and target.id == 'DECISION_FIELDS'
                                          for target in node.targets)))
    assert not set(fields) & {'current_iou', 'future_iou', 'history_iou', 'wrong_update_fraction', 'action_valid'}
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(base['root'], base['cache'])
    sequences = {dataset[index].get_name(): dataset[index] for index in range(len(dataset))}
    entries, jobs, event_counts = [], [], []
    totals = {'queries_with_write_pair': 0, 'candidate_write_pairs': 0,
              'original_candidate_write_pairs': 0, 'pause_beneficial_pairs': 0,
              'pause_harmful_pairs': 0, 'pause_equal_pairs': 0,
              'original_candidate_missing_queries': 0, 'extra_region_correct_candidate_queries': 0}
    common = ('root', 'cache', 'head', 'pretrained', 'motion_run', 'split', 'regions',
              'prefix_policy', 'prefix_execution', 'future_execution', 'history_descriptor_storage',
              'max_prefix', 'batch_clips', 'forward_batch', 'seed', 'future_horizon')
    for gpu in range(4):
        output = OUTPUT / f'full_gpu{gpu}'
        config, completed = read(output / 'config.json'), read(output / 'completion.json')
        expected = read(FOLDER / f'gpu{gpu}_jobs.json')['jobs']
        count = len(expected)
        assert config['jobs'] == [{'sequence': job['sequence'], 'query_frame': job['query_frame']}
                                  for job in expected]
        assert config['clips'] == completed['clips'] == count
        assert config['partition'] == completed['partition'] == 'train'
        assert completed['completed'] and not completed['decision_input_contains_future']
        assert not completed['official_tracking_accuracy']
        assert config['prefix_model'] == plan['teacher'] == base['prefix_model']
        assert config['prefix_checkpoint_epoch'] == config['future_checkpoint_epoch'] == 4
        assert config['future_policy_mode'] == 'own' and config['prefix_write_verification'] == 'action'
        assert config['decision_inputs_copied_before_future_decode'] and config['history_capacity_covers_prefix']
        assert config['future_horizon'] == 3 and not plan['collector_source_changed']
        for key in common:
            assert config[key] == base[key], key
        with np.load(output / 'samples.npz', allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        assert set(data) == set(schema['schema']) and set(fields) <= set(data)
        for key, value in data.items():
            expected_schema = schema['schema'][key]
            assert list(value.shape) == [count] + expected_schema['shape'][1:], key
            assert value.dtype.str == expected_schema['dtype'] and np.isfinite(value).all(), key
        assert np.array_equal(data['action_valid'][..., 0], data['valid'])
        writable = data['action_valid'][..., 1]
        assert np.array_equal(writable, data['valid'] & (data['raw_score'] > .84))
        rows = np.arange(count)
        assert data['valid'][rows, 0, data['original_choice']].all()
        assert int(data['action_valid'].sum()) == completed['valid_actions']
        for key, mask in [('current_iou', data['valid']), ('future_iou', data['action_valid']),
                          ('wrong_update_fraction', data['action_valid'])]:
            assert ((data[key][mask] >= 0) & (data[key][mask] <= 1)).all(), key
        current_error, history_error = 0., 0.
        utility = .7 * data['current_iou'][..., None] + .3 * data['future_iou'].mean(-1) - .1 * data['wrong_update_fraction']
        pause_delta = utility[..., 1] - utility[..., 0]
        original_present = np.where(data['valid'][:, 0], data['current_iou'][:, 0], -1).max(-1) >= .5
        extra_present = np.where(data['valid'][:, 1:], data['current_iou'][:, 1:], -1).reshape(count, -1).max(-1) >= .5
        for index, job in enumerate(expected):
            sequence, query = sequences[job['sequence']], job['query_frame']
            assert job['sequence'] in split['train'] and job['sequence'] not in split['validation']
            targets = np.asarray([sequence[t].get_bounding_box() for t in range(query, query + 4)])
            assert np.isfinite(targets).all() and (targets[:, 2:] > targets[:, :2]).all()
            measured = overlaps(data['image_boxes'][index].astype(np.float64), targets[0])
            error = float(np.abs(measured - data['current_iou'][index])[data['valid'][index]].max())
            assert error <= 1e-4, (job, error)
            current_error = max(current_error, error)
            past = data['history_valid'][index].astype(bool)
            frames = data['history_frames'][index][past]
            assert np.array_equal(frames, np.arange(max(0, query - base['max_prefix']), query)), job
            past_targets = np.asarray([sequence[int(frame)].get_bounding_box() for frame in frames])
            gt_valid = np.isfinite(past_targets).all(-1) & (past_targets[:, 2:] > past_targets[:, :2]).all(-1)
            labels = data['history_iou'][index][past]
            assert (labels[~gt_valid] == -1).all()
            measured = overlaps(data['history_boxes'][index][past][gt_valid].astype(np.float64), past_targets[gt_valid])
            error = float(np.abs(measured - labels[gt_valid]).max())
            assert error <= 1e-4, (job, error)
            history_error = max(history_error, error)
            mask = writable[index]
            event_counts.append({'event_id': job['event_id'], 'sequence': job['sequence'], 'query_frame': query,
                                 'candidate_write_pairs': int(mask.sum()),
                                 'pause_beneficial_pairs': int((pause_delta[index][mask] > 1e-6).sum()),
                                 'pause_harmful_pairs': int((pause_delta[index][mask] < -1e-6).sum())})
        totals['queries_with_write_pair'] += int(writable.reshape(count, -1).any(-1).sum())
        totals['candidate_write_pairs'] += int(writable.sum())
        totals['original_candidate_write_pairs'] += int(writable[rows, 0, data['original_choice']].sum())
        totals['pause_beneficial_pairs'] += int((pause_delta[writable] > 1e-6).sum())
        totals['pause_harmful_pairs'] += int((pause_delta[writable] < -1e-6).sum())
        totals['pause_equal_pairs'] += int((np.abs(pause_delta[writable]) <= 1e-6).sum())
        totals['original_candidate_missing_queries'] += int((~original_present).sum())
        totals['extra_region_correct_candidate_queries'] += int((~original_present & extra_present).sum())
        entries.append({'gpu': gpu, 'output': str(output), 'clips': count,
                        'current_GT_max_error': current_error, 'history_GT_max_error': history_error,
                        'completion': completed})
        jobs.extend(expected)
        print('SHARD_PASS', gpu, count, flush=True)
        del data
    assert jobs == full_jobs and len(pairs(jobs)) == 98
    assert totals['candidate_write_pairs'] == sum(totals[k] for k in ('pause_beneficial_pairs', 'pause_harmful_pairs', 'pause_equal_pairs'))
    receipt = {'status': 'PASS', 'new_queries': 98, 'combined_train_queries': 2576,
               'all_queries_TRAIN_only': True, 'prior2478_and_validation196_unchanged': True,
               'GT_current_and_all_valid_history_verified': True, 'entries': entries,
               'write_and_search_opportunities': totals, 'event_level_write_counts': event_counts,
               'event_unit': 'One previously unqueried representative per actual wrong-localization write episode; candidate pairs are not independent events.',
               'future_label_scope': 'Actual frozen own-policy H3 outputs, source-linked and range checked; no independent future neural rerun.',
               'scope': 'TRAIN supervision coverage, not native accuracy or proof of causal write damage.',
               'execution': {'new_neural_forward_calls': 0, 'optimizer_steps': 0, 'native_test_reads': False},
               'completed_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
    target = OUTPUT / 'collection_cpu_acceptance.json'
    assert not target.exists()
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('entries', 'event_level_write_counts')}), flush=True)


if __name__ == '__main__':
    main()
