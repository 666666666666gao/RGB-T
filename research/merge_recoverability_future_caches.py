"""Verify and concatenate the two matched frozen-future-teacher collections.

CPU/NumPy only. A completed collection is required before any output is written.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FUTURE_FIELDS = {'future_iou', 'wrong_update_fraction'}
COMMON_CONFIG = ('prefix_model', 'prefix_checkpoint_epoch', 'prefix_write_verification',
                 'split', 'root', 'cache', 'regions', 'pretrained', 'head', 'motion_run', 'max_prefix', 'batch_clips',
                 'forward_batch', 'seed', 'future_horizon')


def read_partition(base, policy, partition, expected_jobs):
    arrays, configs, receipts, roots, jobs = [], [], [], [], []
    for shard in range(4):
        root = base / f'recoverability_future_{policy}_{partition}_shard{shard}_20261004'
        assert (root / 'job_completed.txt').is_file(), str(root)
        config = json.loads((root / 'config.json').read_text())
        receipt = json.loads((root / 'completion.json').read_text())
        assert config['partition'] == receipt['partition'] == partition
        assert config['future_policy_mode'] == policy
        assert receipt['completed'] and not receipt['decision_input_contains_future']
        assert not receipt['official_tracking_accuracy']
        with np.load(root / 'samples.npz') as archive:
            data = {key: archive[key].copy() for key in archive.files}
        count = len(config['jobs'])
        assert count == config['clips'] == receipt['clips'] > 0
        assert all(len(value) == count and np.isfinite(value).all() for value in data.values())
        assert data['instance_features'].shape[1:] == (7, 5, 2, 768)
        assert data['future_iou'].shape[1:] == (7, 5, 2, 3)
        assert np.array_equal(data['action_valid'][..., 0], data['valid'])
        assert np.array_equal(data['action_valid'][..., 1], data['valid'] & (data['raw_score'] > .84))
        rows = np.arange(count)
        assert data['valid'][rows, 0, data['original_choice']].all()
        assert data['history_valid'][:, -1].all()
        query = np.asarray([job['query_frame'] for job in config['jobs']])[:, None]
        assert ((data['history_frames'] < query) | ~data['history_valid']).all()
        assert int(data['action_valid'].sum()) == receipt['valid_actions']
        if configs:
            assert data.keys() == arrays[0].keys()
            assert all(config[key] == configs[0][key] for key in COMMON_CONFIG)
            assert config['future_policy'] == configs[0]['future_policy']
            assert all(data[key].dtype == arrays[0][key].dtype and
                       data[key].shape[1:] == arrays[0][key].shape[1:] for key in data)
        arrays.append(data)
        configs.append(config)
        receipts.append(receipt)
        roots.append(str(root))
        jobs.extend(config['jobs'])
    assert jobs == expected_jobs
    assert len({(job['sequence'], job['query_frame']) for job in jobs}) == len(jobs)
    data = {key: np.concatenate([part[key] for part in arrays]) for key in arrays[0]}
    return data, configs, receipts, roots


def verify_pair(reference, own):
    assert reference.keys() == own.keys()
    assert FUTURE_FIELDS <= reference.keys()
    for key in reference:
        assert reference[key].dtype == own[key].dtype and reference[key].shape == own[key].shape, key
        if key not in FUTURE_FIELDS:
            assert np.array_equal(reference[key], own[key]), key
    return {key: {'different_elements': int(np.count_nonzero(reference[key] != own[key])),
                  'max_absolute_difference': float(np.abs(reference[key] - own[key]).max())}
            for key in sorted(FUTURE_FIELDS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='/data/gb/outputs', type=Path)
    parser.add_argument('--jobs', default='/data/gb/setup/recoverability_future_policy_jobs_20261004', type=Path)
    parser.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    jobs = {part: json.loads((args.jobs / (part + '.json')).read_text()) for part in ('train', 'validation')}
    assert {part: len(value) for part, value in jobs.items()} == {'train': 902, 'validation': 128}
    split = json.loads(args.split.read_text())
    assert not set(split['train']) & set(split['validation'])
    for part in jobs:
        assert {job['sequence'] for job in jobs[part]} <= set(split[part])
    caches, comparisons = {}, {}
    for part in jobs:
        for policy in ('c1', 'own'):
            caches[policy, part] = read_partition(args.base, policy, part, jobs[part])
        comparisons[part] = verify_pair(caches['c1', part][0], caches['own', part][0])
        assert all(caches['c1', part][1][0][key] == caches['own', part][1][0][key] for key in COMMON_CONFIG)
    first = caches['c1', 'train'][1][0]
    assert all(config[key] == first[key] for _, configs, _, _ in caches.values()
               for config in configs for key in COMMON_CONFIG)
    assert first['split'] == str(args.split)
    for policy in ('c1', 'own'):
        assert caches[policy, 'train'][1][0]['future_policy'] == caches[policy, 'validation'][1][0]['future_policy']
    assert first['prefix_checkpoint_epoch'] == 4 and first['prefix_write_verification'] == 'action'
    assert first['batch_clips'] == first['forward_batch'] == 1
    assert first['max_prefix'] == 1024 and first['future_horizon'] == 3 and first['seed'] == 42
    model_sha = hashlib.sha256(Path(first['prefix_model']).read_bytes()).hexdigest()
    assert model_sha == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
    args.output.mkdir(parents=True, exist_ok=False)
    for (policy, part), (data, configs, receipts, roots) in caches.items():
        out = args.output / policy / part
        out.mkdir(parents=True)
        (out / 'jobs.json').write_text(json.dumps({'jobs': jobs[part]}, indent=2))
        config = configs[0] | {'output': str(out), 'clips': len(jobs[part]), 'jobs': jobs[part],
                               'jobs_file': str(out / 'jobs.json'), 'merged_from': roots,
                               'source_configs': configs, 'merge_method': 'CPU exact ordered concatenation; no NN rerun'}
        np.savez_compressed(out / 'samples.npz', **data)
        with np.load(out / 'samples.npz') as archive:
            assert all(np.array_equal(archive[key], data[key]) for key in data)
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        receipt = {'completed': True, 'partition': part, 'clips': len(jobs[part]),
                   'valid_actions': int(data['action_valid'].sum()), 'source_receipts': receipts,
                   'decision_input_contains_future': False, 'official_tracking_accuracy': False,
                   'cpu_merge_only': True, 'strict_npz_reload_equal': True}
        (out / 'completion.json').write_text(json.dumps(receipt, indent=2))
        (out / 'job_completed.txt').write_text('CPU merge complete; no training or benchmark claim\n')
    result = {'status': 'PASS', 'matched_partitions': {p: len(jobs[p]) for p in jobs},
              'all_non_future_arrays_exact': True, 'future_labels': comparisons,
              'teacher_checkpoint_sha256': model_sha, 'training_started': False,
              'official_goal_completed': False, 'next_gate': 'matched B384 training sanity before full fit'}
    (args.output / 'paired_cache_gate.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
