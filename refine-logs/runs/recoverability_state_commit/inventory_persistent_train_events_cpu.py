"""Inspect closed old4 scalar prefix records for persistent/recovering TRAIN events."""
import datetime
import json
from pathlib import Path
import numpy as np

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_state_commit'
SOURCE = Path('/data/gb/outputs/recoverability_best_native_policy_merged_20261005/own/train')
config = json.loads((SOURCE / 'config.json').read_text())
assert config['prefix_model'] == '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
assert config['prefix_checkpoint_epoch'] == 4 and config['prefix_write_verification'] == 'action'
assert config['partition'] == 'train' and config['prefix_execution'] == 'per-sequence'
split = json.loads(Path(config['split']).read_text())
assert len(split['train']) == 881 and not set(split['train']) & set(split['validation'])
jobs = config['jobs']
with np.load(SOURCE / 'samples.npz', allow_pickle=False) as a:
    overlaps, frames, valid, writes = [a[key].copy() for key in ['history_iou', 'history_frames', 'history_valid', 'history_write']]
assert overlaps.shape == frames.shape == valid.shape == writes.shape == (1762, 1024)
latest = {}
for index, job in enumerate(jobs):
    assert job['sequence'] in split['train']
    name = job['sequence']
    if name not in latest or job['query_frame'] > jobs[latest[name]]['query_frame']:
        latest[name] = index
assert len(latest) == 881
events, trace_frames = [], 0
for name, index in sorted(latest.items()):
    slots = np.flatnonzero(valid[index])
    times = frames[index, slots].tolist()
    assert times == list(range(jobs[index]['query_frame']))
    trace = {int(t): (float(overlaps[index, slot]), bool(writes[index, slot])) for t, slot in zip(times, slots)}
    trace_frames += len(trace)
    t = 1
    while t < len(trace):
        if not 0 <= trace[t][0] < .2:
            t += 1
            continue
        start = t
        while t + 1 < len(trace) and 0 <= trace[t + 1][0] < .2:
            t += 1
        end = t
        recover = next((offset for offset in range(1, 31)
                        if all(start + offset + h in trace and trace[start + offset + h][0] >= .5 for h in range(3))), None)
        role = 'persistent33' if end - start + 1 >= 33 else 'recovered32' if recover is not None else 'unclassified'
        if role != 'unclassified' and start >= 3:
            events.append({'sequence': name, 'query_frame': start, 'event_id': name + ':old4failure:' + str(start),
                           'cached_role': role, 'cached_failed_run_frames': end - start + 1,
                           'cached_recovery3_correct_offset': recover, 'source_query_frame': jobs[index]['query_frame'],
                           'cached_previous_iou': trace[start - 1][0],
                           'cached_failure_start_written': trace[start][1]})
        t += 1
record = {'status': 'ACTUAL_CLOSED_OLD4_TRAIN_EVENT_INVENTORY',
          'computed_at_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'source': str(SOURCE), 'source_scalar_arrays': ['history_iou', 'history_frames', 'history_valid', 'history_write'],
          'one_latest_prefix_per_TRAIN_sequence': True, 'TRAIN_sequences': len(latest), 'prefix_frames': trace_frames,
          'persistent33_events': sum(e['cached_role'] == 'persistent33' for e in events),
          'recovering32_events': sum(e['cached_role'] == 'recovered32' for e in events),
          'persistent_sequences': len({e['sequence'] for e in events if e['cached_role'] == 'persistent33'}),
          'recovering_sequences': len({e['sequence'] for e in events if e['cached_role'] == 'recovered32'}),
          'event_candidates': events, 'GT_role': 'Previously cached TRAIN IoU only for event sampling; no model inputs',
          'NN_calls': 0, 'GPU_queries': 0, 'images_decoded': 0, 'native_TEST_read': False}
(FOLDER / 'actual_persistent_train_event_inventory.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k != 'event_candidates'}))
