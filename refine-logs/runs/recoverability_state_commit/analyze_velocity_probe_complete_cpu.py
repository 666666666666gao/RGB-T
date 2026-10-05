"""Use the reviewed event analysis for the TRAIN88 states under fixed old4 and velocity-search reference."""
import json
from pathlib import Path
import runpy
import numpy as np

FOLDER = Path('/data/gb/GOLA/refine-logs/runs/recoverability_state_commit')
BASE = Path('/data/gb/outputs/recoverability_velocity_search_commit_train88_20261005')
ACTOR = '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
expected = json.loads((FOLDER / 'prepared_velocity_train88_jobs.json').read_text())['jobs']
rows = []
for part, count in [('sanity', 1), ('gpu0', 22), ('gpu1', 22), ('gpu2', 22), ('gpu3', 21)]:
    out = BASE / part
    assert (out / 'COMPLETE').is_file()
    config = json.loads((out / 'config.json').read_text())
    assert config['model'] == ACTOR and config['checkpoint_epoch'] == 4
    d = json.loads((out / 'events.json').read_text())
    assert d['geometry_reference'] == 'velocity_search' and d['motion_reference_changed_queries'] == 0
    assert d['queries'] == count and d['status'] == 'COMPLETE_MATCHED_QUERY_COMMIT_ROLLOUTS'
    for index, q in enumerate(d['results']):
        with np.load(out / f'event_{index:04d}.npz') as arrays:
            names = ['raw', 'pause', 'search_velocity', 'pause_search_velocity']
            assert all(np.array_equal(arrays['raw_boxes_xyxy'][0], arrays[n + '_boxes_xyxy'][0]) for n in names)
            assert all(len(arrays[n + '_iou']) == 33 for n in names)
        rows.append(q)
assert len(rows) == len({(q['sequence'], q['query_frame']) for q in rows}) == 88
assert {(q['sequence'], q['query_frame']) for q in rows} == {(q['sequence'], q['query_frame']) for q in expected}
assert all(not q['query_motion_reference_different'] for q in rows)
result = runpy.run_path(str(FOLDER / 'analyze_probe_complete_cpu.py'))['analyze'](rows, ('search_velocity', 'pause_search_velocity'))
assert result['events'] == 24
result['query_search_center_shift_pixels'] = [q['query_search_reference_center_shift_pixels'] for q in rows]
result['requested_reference_not_exact_queries'] = sum(not q['query_requested_reference_matched_exactly'] for q in rows)
target = BASE / 'merged88_event_analysis.json'
assert not target.exists()
target.write_text(json.dumps(result | {'status': 'COMPLETE_VELOCITY_SEARCH_TRAIN88_QUERIES24_EVENTS',
                                      'actor': ACTOR, 'results': rows}, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ['event_rows', 'eligible_write_pause_future_deltas']}))
