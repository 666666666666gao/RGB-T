"""Prepare two disjoint-partition temporal queries per video from retained old4."""
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped

assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and Path.cwd() == Path('/data/gb/GOLA')
setup = Path('/data/gb/setup')
read = lambda path: json.loads(Path(path).read_text())
reference = read('/data/gb/outputs/recoverability_full_coverage_merged_20261004/own/train/config.json')
split = read(reference['split'])
assert len(split['train']) == 881 and len(split['validation']) == 98 and not set(split['train']) & set(split['validation'])
dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(reference['root'], reference['cache'])
sequences = {dataset[i].get_name(): dataset[i] for i in range(len(dataset))}
teacher = '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
assert Path(teacher).is_file()
jobs, shards = {}, {}
for partition in ('train', 'validation'):
    rows = []
    for name in sorted(split[partition]):
        boxes = sequences[name].get_all_bounding_boxes()
        valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
        eligible = [q for q in range(1, min(1025, len(boxes) - 3)) if valid[q:q + 4].all()]
        assert valid[0] and len(eligible) >= 2, (partition, name, len(eligible))
        for index in (len(eligible) // 4, 3 * len(eligible) // 4):
            rows.append({'sequence': name, 'query_frame': eligible[index]})
    assert Counter(row['sequence'] for row in rows) == Counter({name: 2 for name in split[partition]})
    assert len(rows) == len({(row['sequence'], row['query_frame']) for row in rows})
    bins, costs = [[] for _ in range(4)], [0] * 4
    for row in sorted(rows, key=lambda row: (-row['query_frame'], row['sequence'])):
        gpu = min(range(4), key=lambda gpu: (costs[gpu], gpu))
        bins[gpu].append(row)
        costs[gpu] += row['query_frame'] + 128  # Workload proxy, not wall-clock seconds.
    jobs[partition] = []
    for gpu, assigned in enumerate(bins):
        assigned.sort(key=lambda row: (row['query_frame'], row['sequence']))
        jobs[partition].extend(assigned)
        path = setup / f'best_native_policy_jobs_{partition}_gpu{gpu}_20261005.json'
        assert not path.exists()
        path.write_text(json.dumps({'jobs': assigned}, indent=2) + '\n')
        shards[f'{partition}_gpu{gpu}'] = {'gpu': gpu, 'clips': len(assigned), 'jobs_file': str(path),
                                           'prefix_plus_rollout_workload_proxy': costs[gpu]}
        if partition == 'train':
            sanity = [assigned[0], assigned[-1]]
            assert sanity[0] != sanity[1]
            path = setup / f'best_native_policy_jobs_sanity_gpu{gpu}_20261005.json'
            assert not path.exists()
            path.write_text(json.dumps({'jobs': sanity}, indent=2) + '\n')
            shards[f'sanity_gpu{gpu}'] = {'gpu': gpu, 'clips': 2, 'jobs_file': str(path)}
schema = read('/data/gb/outputs/recoverability_full_coverage_merged_20261004/own/train/partition_cpu_acceptance.json')['schema']
bytes_per_query = sum(int(np.prod(row['shape'][1:])) * np.dtype(row['dtype']).itemsize for row in schema.values())
required = 2 * bytes_per_query * sum(map(len, jobs.values())) + 2 * 2**30
free = shutil.disk_usage('/data/gb').free
assert free >= required, (free, required)
record = {'status': 'PASS', 'teacher': teacher, 'teacher_epoch': 4, 'reference': reference,
          'jobs': jobs, 'shards': shards, 'all881_98_names_twice_and_disjoint': True,
          'max_prefix': 1024, 'future_policy': 'own', 'write_verification': 'action',
          'disk': {'free_bytes': free, 'required_uncompressed_shards_plus_merge_and_margin_bytes': required,
                   'estimated_bytes_per_query_from_actual_prior_schema': bytes_per_query},
          'scope': '1762TRAIN/196developerVAL queries, two/video; no all-frame claim',
          'GPU_queries': 0, 'neural_forward_calls': 0, 'benchmark_test_reads': False}
out = setup / 'best_native_policy_jobs_cpu_acceptance_20261005.json'
assert not out.exists()
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'counts': {p: len(rows) for p, rows in jobs.items()}, 'disk': record['disk']}))
