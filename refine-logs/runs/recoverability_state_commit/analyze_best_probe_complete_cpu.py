"""Use the reviewed event analysis for the same48 states under fixed old4."""
import json
from pathlib import Path
import runpy
import numpy as np

FOLDER = Path('/data/gb/GOLA/refine-logs/runs/recoverability_state_commit')
BASE = Path('/data/gb/outputs/recoverability_geometry_commit_best_train_20261005')
ACTOR = '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
expected = json.loads((FOLDER / 'prepared_geometry_commit_train_jobs_cpu.json').read_text())['jobs']
rows = []
for part, count in [('sanity', 1), ('gpu0', 12), ('gpu1', 12), ('gpu2', 12), ('gpu3', 11)]:
    out = BASE / part
    assert (out / 'COMPLETE').is_file()
    config = json.loads((out / 'config.json').read_text())
    assert config['model'] == ACTOR and config['checkpoint_epoch'] == 4
    d = json.loads((out / 'events.json').read_text())
    assert d['queries'] == count and d['status'] == 'COMPLETE_MATCHED_QUERY_COMMIT_ROLLOUTS'
    for index, q in enumerate(d['results']):
        with np.load(out / f'event_{index:04d}.npz') as arrays:
            names = ['raw', 'pause', 'geometry_C1', 'pause_geometry_C1']
            assert all(np.array_equal(arrays['raw_boxes_xyxy'][0], arrays[n + '_boxes_xyxy'][0]) for n in names)
            assert all(len(arrays[n + '_iou']) == 33 for n in names)
        rows.append(q)
assert len(rows) == len({(q['sequence'], q['query_frame']) for q in rows}) == 48
assert {(q['sequence'], q['query_frame']) for q in rows} == {(q['sequence'], q['query_frame']) for q in expected}
result = runpy.run_path(str(FOLDER / 'analyze_probe_complete_cpu.py'))['analyze'](rows)
assert result['events'] == 16
target = BASE / 'merged48_event_analysis.json'
assert not target.exists()
target.write_text(json.dumps(result | {'status': 'COMPLETE_BEST_OLD4_TRAIN48_QUERIES16_EVENTS',
                                      'actor': ACTOR, 'results': rows}, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ['event_rows', 'eligible_write_pause_future_deltas']}))
