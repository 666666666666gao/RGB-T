"""Promote only the exact completed four-shard union to full native inference coverage."""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', choices=('lasher', 'rgbt234'), required=True)
args = p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
setup = Path('/data/gb/setup')
selection = json.loads((setup / 'train_events_native_selection_20261005.json').read_text())
assert selection['status'] == 'PASS' and selection['all_four_full_fits_CPU_passed']
review = json.loads((Path('/data/gb/GOLA/refine-logs/runs/recoverability_train_events/evaluation_source_review.json')).read_text())
assert review['status'] == 'PASS'
dataset = args.dataset
data = Path('/data/wangwj/dataset/LasHeR/testingset' if dataset == 'lasher' else '/data/zhouy/DATASET/RGB-T234')
expected_count, expected_frames = (245, 220703) if dataset == 'lasher' else (234, 116649)
offsets, counts = ((0, 62, 123, 184), (62, 61, 61, 61)) if dataset == 'lasher' else ((0, 59, 118, 176), (59, 59, 58, 58))
sequences = sorted(path for path in data.iterdir() if path.is_dir())
assert len(sequences) == expected_count
root = Path('/data/gb/outputs/recoverability_train_events_native_full_20261005') / dataset
out = root / 'predictions'
assert not out.exists()
configs, completions, records, latency, files = [], [], [], [], []
common = ('dataset', 'root', 'model', 'pretrained', 'c1_head', 'motion_run', 'max_frames', 'zero_init',
          'parity_check', 'policy', 'disable_search', 'unsafe_writes', 'write_verification', 'seed', 'validation_split',
          'threshold', 'head_epoch', 'model_training_seed', 'pretrained_load', 'amp_dtype', 'initialization')
for gpu, (offset, count) in enumerate(zip(offsets, counts)):
    shard = root / 'shards' / ('gpu' + str(gpu))
    pred = shard / 'predictions'
    assert (shard / 'inference_completed.txt').is_file()
    cfg = json.loads((pred / 'inference_config.json').read_text())
    done = json.loads((pred / 'inference_completion.json').read_text())
    assert done['completed'] and done['smoke_only'] and cfg['smoke_only']
    assert cfg['sequence_offset'] == offset and cfg['limit_sequences'] == done['sequences'] == count
    assert cfg['model'] == selection['checkpoint'] and cfg['head_epoch'] == selection['selected_best_epoch']
    assert cfg['root'] == str(data) and cfg['dataset'] == dataset and cfg['write_verification'] == 'action'
    assert cfg['max_frames'] == 0 and cfg['validation_split'] is None
    assert not any(cfg[key] for key in ('zero_init', 'parity_check', 'disable_search', 'unsafe_writes')) and cfg['policy'] == 'learned'
    names = [path.name for path in sequences[offset:offset + count]]
    assert [row['sequence'] for row in done['records']] == names
    assert sum(row['frames'] for row in done['records']) == done['frames']
    assert sorted(path.stem for path in pred.glob('*.txt')) == sorted(names)
    for sequence, record in zip(sequences[offset:offset + count], done['records']):
        visible = sorted(path for path in (sequence / 'visible').iterdir() if path.is_file())
        infrared = sorted(path for path in (sequence / 'infrared').iterdir() if path.is_file())
        prediction = np.loadtxt(pred / (sequence.name + '.txt'), delimiter='\t')
        times = np.load(pred / (sequence.name + '_latency.npy'), allow_pickle=False)
        assert len(visible) == len(infrared) == record['frames'] == len(prediction) and prediction.shape[1] == 4
        assert times.shape == (len(prediction) - 1,) and np.isfinite(prediction).all() and np.isfinite(times).all() and (times > 0).all()
        latency.append(times)
        for suffix in ('.txt', '_latency.npy', '_recoverability_decisions.npz'):
            path = pred / (sequence.name + suffix)
            assert path.is_file()
            files.append(path)
    configs.append(cfg)
    completions.append(done)
    records.extend(done['records'])
assert all(all(cfg[key] == configs[0][key] for key in common) for cfg in configs)
assert [row['sequence'] for row in records] == [path.name for path in sequences]
assert len(records) == expected_count and sum(row['frames'] for row in records) == expected_frames
assert len({path.name for path in files}) == len(files) == expected_count * 3
out.mkdir()
for path in files:
    os.link(path, out / path.name)
config = dict(configs[0], output=str(out), sequence_offset=0, limit_sequences=0, smoke_only=False,
              full_shard_union_verified=True, source_shards=[str(root / 'shards' / ('gpu' + str(gpu))) for gpu in range(4)])
(out / 'inference_config.json').write_text(json.dumps(config, indent=2) + '\n')
times = np.concatenate(latency)
for row in records:
    row['sequence_index'] = [path.name for path in sequences].index(row['sequence']) + 1
    row['sequences'] = expected_count
receipt = {'completed': True, 'sequences': expected_count, 'frames': expected_frames, 'smoke_only': False,
           'records': records, 'gt_scoring_completed': False, 'full_shard_union_verified': True,
           'fps_including_decode_crop_update': len(times) / times.sum(),
           'latency_p50_ms': float(np.percentile(times, 50) * 1000), 'latency_p95_ms': float(np.percentile(times, 95) * 1000),
           'peak_cuda_allocated_mib': max(row['peak_cuda_allocated_mib'] for row in completions),
           'peak_cuda_reserved_mib': max(row['peak_cuda_reserved_mib'] for row in completions),
           'wall_seconds_including_initialization_and_diagnostic_serialization': max(row['wall_seconds_including_initialization_and_diagnostic_serialization'] for row in completions),
           'timing_note': 'Per-frame instrumented latency under four concurrent workers; not a fair isolated-speed comparison.',
           'official_accuracy': 'Native actual-GT collector still required.'}
(out / 'inference_completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
with (out / 'progress.jsonl').open('w') as stream:
    for row in records:
        stream.write(json.dumps(row) + '\n')
proof = {'status': 'PASS', 'dataset': dataset, 'sequences': expected_count, 'frames': expected_frames,
         'same_fixed_checkpoint': selection['checkpoint'], 'all_sequences_full_frames': True,
         'all_four_actual_shards_completed': True, 'native_metrics_computed': False,
         'merged_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
(root / 'full_inference_merge_acceptance.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof))
