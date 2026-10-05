"""Inventory GT-unknown TRAIN episodes in sealed old4 prefixes, without any inference."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

base = Path('/data/gb/outputs/recoverability_best_native_policy_merged_20261005/own/train')
config = json.loads((base / 'config.json').read_text())
jobs = config['jobs']
split = json.loads(Path(config['split']).read_text())
assert config['partition'] == 'train' and len(jobs) == 1762
assert {job['sequence'] for job in jobs} == set(split['train']) and len(split['train']) == 881
assert not set(split['train']) & set(split['validation'])
states = {}
with np.load(base / 'samples.npz', allow_pickle=False) as data:
    valid, frames, quality = data['history_valid'], data['history_frames'], data['history_iou']
    writes, evidence = data['history_write'], data['history_evidence']
    assert valid.shape == frames.shape == quality.shape == writes.shape == (1762, 1024)
    for index in sorted(range(len(jobs)), key=lambda i: jobs[i]['query_frame']):
        timeline = states.setdefault(jobs[index]['sequence'], {})
        for slot in np.flatnonzero(valid[index]):
            timeline[int(frames[index, slot])] = {'iou': float(quality[index, slot]),
                                                 'raw_score': float(evidence[index, slot, 0]),
                                                 'template_written': bool(writes[index, slot])}
unknown_frames = sum(frame > 0 and row['iou'] < 0 for timeline in states.values() for frame, row in timeline.items())
assert unknown_frames == 1080
events = []
for name, timeline in sorted(states.items()):
    unknown = sorted(frame for frame, row in timeline.items() if frame > 0 and row['iou'] < 0)
    episodes = []
    for frame in unknown:
        if not episodes or frame != episodes[-1][-1] + 1:
            episodes.append([])
        episodes[-1].append(frame)
    for episode in episodes:
        start, end = episode[0], episode[-1]
        before, after = timeline.get(start - 1), timeline.get(end + 1)
        observed = [timeline.get(frame) for frame in range(end + 1, end + 33)]
        recovered = any(all(row is not None and row['iou'] >= .5 for row in observed[offset:offset + 3]) for offset in range(30))
        complete_known = all(row is not None and row['iou'] >= 0 for row in observed)
        role = 'recovered_within32_after_gap' if recovered else 'not_recovered_in_complete_known32' if complete_known else 'incomplete_or_GT_unknown_future'
        events.append({'sequence': name, 'event_id': f'{name}:GT_unknown:{start}',
                       'unknown_start_frame': start, 'unknown_end_frame': end, 'unknown_frames': len(episode),
                       'preceding_predicted_known_iou': before['iou'] if before is not None and start > 1 else None,
                       'next_known_iou': after['iou'] if after is not None and after['iou'] >= 0 else None,
                       'role': role, 'H3_from_start_has_zero_known_frames': all(timeline[frame]['iou'] < 0 for frame in range(start, min(end + 1, start + 4))) and len(episode) >= 4,
                       'actual_template_writes_during_unknown': sum(timeline[frame]['template_written'] for frame in episode),
                       'label_limitation': 'GT-unknown is not an incorrect candidate or semantic distractor. Role refers only to recorded old4 predictions and finite observed future.'})
result = {'completed': True, 'TRAIN_sequences': 881, 'unknown_noninitial_frames': unknown_frames,
          'GT_unknown_episodes': len(events), 'episodes_at_least4_frames': sum(row['unknown_frames'] >= 4 for row in events),
          'roles': dict(Counter(row['role'] for row in events)),
          'actual_template_writes_during_unknown': sum(row['actual_template_writes_during_unknown'] for row in events),
          'source': str(base), 'records': events,
          'existing_collector_eligibility': 'collect_recoverability.py requires current and next3 GT boxes all valid, so these unknown query frames have no current action label in the existing caches.',
          'scope': 'Sealed TRAIN prefixes only, not full raw-video coverage. One event per contiguous GT-unknown episode; no new candidates or matched-action consequences inferred.',
          'neural_forward_calls': 0, 'optimizer_steps': 0, 'GPU_queries': 0, 'native_TEST_data_read': False}
print(json.dumps(result, allow_nan=False))
