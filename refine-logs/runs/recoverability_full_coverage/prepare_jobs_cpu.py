"""One eligible training query from every TRAIN/VAL video, never benchmark test data."""
import json
import os
from pathlib import Path

import numpy as np
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped

assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and Path.cwd() == Path('/data/gb/GOLA')
setup = Path('/data/gb/setup')
read = lambda path: json.loads(Path(path).read_text())
accepted = read(setup / 'best_policy_fit_actual_full_fit_cpu_audit_20261004.json')
assert accepted['status'] == 'PASS' and accepted['all_four_lr_ranking_arms_passed'] and accepted['epochs_per_arm'] == 60
arms = ('pairwise_lr4', 'pairwise_lr5', 'budgeted_lr4', 'budgeted_lr5')
chosen = max(arms, key=lambda arm: accepted['arms'][arm]['best_validation']['utility'])
parent = accepted['arms'][chosen]
reference = read('/data/gb/outputs/recoverability_best_policy_merged_20261004/own/train/config.json')
split = read(reference['split'])
assert len(split['train']) == 881 and len(split['validation']) == 98 and not set(split['train']) & set(split['validation'])
dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(reference['root'], reference['cache'])
sequences = {dataset[index].get_name(): dataset[index] for index in range(len(dataset))}
jobs, shards = {}, {}
for part, counts in (('train', (221, 220, 220, 220)), ('validation', (25, 25, 24, 24))):
    rows = []
    for name in sorted(split[part]):
        boxes = sequences[name].get_all_bounding_boxes()
        valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
        eligible = [frame for frame in range(1, min(1025, len(boxes) - 3)) if valid[frame:frame + 4].all()]
        assert valid[0] and eligible, (part, name)
        rows.append({'sequence': name, 'query_frame': eligible[len(eligible) // 2]})
    assert {row['sequence'] for row in rows} == set(split[part]) and len(rows) == len(split[part])
    jobs[part] = rows
    offset = 0
    for gpu, count in enumerate(counts):
        selected = rows[offset:offset + count]
        assert len(selected) == count
        path = setup / ('full_coverage_jobs_' + part + '_gpu' + str(gpu) + '_20261004.json')
        assert not path.exists()
        path.write_text(json.dumps({'jobs': selected}, indent=2) + '\n')
        shards[part + '_gpu' + str(gpu)] = {'gpu': gpu, 'clips': count, 'jobs_file': str(path)}
        offset += count
    assert offset == len(rows)
record = {'status': 'PASS', 'teacher': str(Path(parent['root']) / 'best.pth'),
          'teacher_epoch': parent['best_epoch'], 'teacher_sha256': parent['artifact_sha256']['best.pth'],
          'teacher_selection': 'Best completed current-round developer VAL utility; no native selection.',
          'teacher_is_not_yet_native_best': True, 'init_same_as_prefix_and_future_teacher': True,
          'eligible_train_videos': 881, 'eligible_validation_videos': 98,
          'all_train_video_names_exact': True, 'all_validation_video_names_exact': True,
          'jobs': jobs, 'shards': shards, 'reference': reference,
          'scope': 'All881 TRAIN and98 developer VAL videos, one fixed eligible mid-prefix query/video; backbone frozen, no all-frame/end-to-end claim.',
          'maximum_query_frame_and_history_window': 1024, 'future_horizon': 3,
          'GPU_queries': 0, 'neural_forward_calls': 0, 'native_test_reads': False}
path = setup / 'full_coverage_jobs_cpu_acceptance_20261004.json'
assert not path.exists()
path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: record[key] for key in ('status', 'teacher_epoch', 'eligible_train_videos', 'eligible_validation_videos', 'scope')}))
