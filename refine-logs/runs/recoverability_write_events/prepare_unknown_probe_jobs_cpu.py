"""Select one actually GT-unknown TRAIN query per long episode; no model execution."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research.collect_candidate_metrics import ground_truth

folder = Path('/data/gb/GOLA/refine-logs/runs/recoverability_write_events')
inventory = json.loads((folder / 'actual_TRAIN_unknown_events_cpu.json').read_text())
split = json.loads(Path('/data/gb/outputs/c1_initial_seed42/split.json').read_text())
assert len(split['train']) == 881 and not set(split['train']) & set(split['validation'])
jobs, excluded = [], []
for event in inventory['records']:
    if event['unknown_frames'] < 4:
        continue
    name, query = event['sequence'], event['unknown_start_frame']
    assert name in split['train'] and query > 0
    gt = ground_truth('/data/wangwj/dataset/LasHeR/traingset', name, 'lasher')
    known = np.isfinite(gt).all(1) & (gt[:, 2:] > 0).all(1)
    assert known[0] and not known[query:query + 4].any()
    if query + 32 >= len(gt) or known[query + 1:query + 33].sum() < 3:
        excluded.append({'event_id': event['event_id'], 'reason': 'Fewer than3future known labels or incomplete32frame video horizon.'})
        continue
    jobs.append({'sequence': name, 'query_frame': query, 'event_id': event['event_id'],
                 'unknown_frames_in_episode': event['unknown_frames'],
                 'future32_known_GT_frames': int(known[query + 1:query + 33].sum()),
                 'source_prefix_role': event['role']})
assert len(jobs) >= 4 and len({(row['sequence'], row['query_frame']) for row in jobs}) == len(jobs)
result = {'status': 'ACTUAL_TRAIN_GT_ELIGIBLE_JOBS_PREPARED_NO_NN', 'jobs': jobs, 'excluded': excluded,
          'long_unknown_episodes_considered': 30, 'eligible_queries': len(jobs),
          'source_roles': dict(Counter(row['source_prefix_role'] for row in jobs)),
          'scope': 'One query at actual first unknown frame per long TRAIN GT-gap event. H3 has no GT; H32 has at least3known future labels. Role is past-prefix observation, not a new action outcome.',
          'model': '/data/gb/outputs/recoverability_write_events_budgeted_lr5_b336_full_20261005/epoch_004.pth',
          'controls': 'Existing C1_components: raw/search-only/motion-only/both. Same query output, actual query pause, template/identity/motion-memory tensors and observation time/quality; only reference coordinates differ.',
          'eligibility_uses_GT_offline_only': True, 'neural_forward_calls': 0, 'optimizer_steps': 0,
          'native_TEST_data_read': False, 'normal_main_tracker_code_changed': False,
          'launch_dependency': 'Current2914365 must actually finish both fullnative reports and GT audits before any new GPU diagnostic; jobs/source alone are not experiments.'}
print(json.dumps(result, allow_nan=False))
