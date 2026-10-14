from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import json

private = Path(__file__).parent
out = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/cadtrack_abc/results/full_sampling_20261014')
launch = json.loads((out / 'action_launch.json').read_bytes())
sampling = json.loads((out / 'sampling_complete_verified.json').read_bytes())
rows = []
for shard in range(2):
    total_queries = 0
    total_seconds = 0.
    parts = []
    for split in ['train', 'validation']:
        queries = json.loads((out / f'queries_{split}_shard{shard}.json').read_bytes())
        maximum = defaultdict(int)
        count = defaultdict(int)
        for item in queries:
            maximum[item['sequence']] = max(maximum[item['sequence']], item['frame'])
            count[item['sequence']] += 1
        receipt = json.loads((out / f'{split}_shard{shard}.json').read_bytes())
        print(json.dumps({'receipt_keys': list(receipt), 'split': split, 'shard': shard}))
        full_frames = sum(s['frames'] for s in receipt['records'])
        full_seconds = sampling['splits'][split]['worker_seconds'][shard]
        prefix_frames = sum(maximum.values()) - len(queries)
        prefix_seconds = prefix_frames / (full_frames / full_seconds)
        # Actual two-query batch64 witness: subsequent query 8.0038 seconds.
        label_seconds = len(queries) * 8.0038
        parts.append(dict(split=split, queries=len(queries), prefix_frames=prefix_frames,
                          expected_prefix_seconds=prefix_seconds, label_seconds=label_seconds))
        total_queries += len(queries)
        total_seconds += prefix_seconds + label_seconds
    rows.append(dict(shard=shard, physical_gpu=2 + shard, queries=total_queries,
                     expected_seconds=total_seconds, parts=parts))
start = datetime.fromisoformat(launch['started_utc'])
estimate = dict(started_utc=start.isoformat(), workers=rows,
                expected_finish_utc=(start + timedelta(seconds=max(r['expected_seconds'] for r in rows))).isoformat(),
                first_observation_not_before_utc=(start + timedelta(seconds=max(r['expected_seconds'] for r in rows) - 300)).isoformat(),
                basis='Actual complete sampling prefix throughput + actual 8.0038-second batch64 query witness; estimate, not completion evidence.',
                observe_interval_after_deadline_seconds=240)
(out / 'action_eta.json').write_bytes((json.dumps(estimate, indent=2) + '\n').encode())
print(json.dumps(estimate, indent=2))
