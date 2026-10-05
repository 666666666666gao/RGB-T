"""Merge48 matched TRAIN queries once and give each of16 events equal weight."""
import json
from pathlib import Path
import numpy as np

BASE = Path('/data/gb/outputs/recoverability_geometry_commit_train_20261005')
FOLDER = Path('/data/gb/GOLA/refine-logs/runs/recoverability_state_commit')


def analyze(rows, geometry_names=('geometry_C1', 'pause_geometry_C1')):
    groups = {}
    for row in rows:
        groups.setdefault((row['sequence'], row['event_id']), []).append(row)
    assert geometry_names in [('geometry_C1', 'pause_geometry_C1'), ('search_velocity', 'pause_search_velocity')]
    comparisons = [(geometry_names[0], 'raw'), (geometry_names[1], 'pause'), ('pause', 'raw')]
    events = []
    for (sequence, event), queries in groups.items():
        item = {'sequence': sequence, 'event_id': event, 'correlated_queries': len(queries),
                'actual_write_eligible_queries': sum(q['query_raw_write_eligible'] for q in queries),
                'geometry_different_queries': sum((not q['selected_matches_C1_keep']) if geometry_names[0] == 'geometry_C1' else q['query_search_reference_different'] for q in queries),
                'controls': {}, 'paired': {}}
        roles = {'normal': 0, 'ambiguous_localization': 0, 'failed_then_recovered32': 0, 'failed_unrecovered32': 0}
        for q in queries:
            raw = next(c for c in q['controls'] if c['name'] == 'raw')
            if raw['query_output_iou'] >= .5:
                role = 'normal'
            elif raw['query_output_iou'] >= .2:
                role = 'ambiguous_localization'
            elif raw['horizons']['32']['first_future_three_correct_run_offset'] is not None:
                role = 'failed_then_recovered32'
            else:
                role = 'failed_unrecovered32'
            roles[role] += 1
        item['raw_reference_query_roles'] = roles
        for h in ['3', '32']:
            for name in ['raw', 'pause', *geometry_names]:
                values = [next(c for c in q['controls'] if c['name'] == name) for q in queries]
                failed = [v for v in values if v['query_output_iou'] < .2]
                item['controls'].setdefault(name, {})[h] = {
                    'mean_future_iou': float(np.mean([v['horizons'][h]['mean_future_iou'] for v in values])),
                    'mean_failure_frames': float(np.mean([v['horizons'][h]['failure_frames_including_query'] for v in values])),
                    'mean_longest_failure': float(np.mean([v['horizons'][h]['longest_failure_run'] for v in values])),
                    'failed_query_states': len(failed),
                    'recovered_query_states': sum(v['horizons'][h]['first_future_three_correct_run_offset'] is not None for v in failed),
                    'unrecovered_query_states': sum(v['horizons'][h]['first_future_three_correct_run_offset'] is None for v in failed)}
            for alternative, reference in comparisons:
                diffs = []
                for q in queries:
                    control = {v['name']: v for v in q['controls']}
                    diffs.append(control[alternative]['horizons'][h]['mean_future_iou'] - control[reference]['horizons'][h]['mean_future_iou'])
                item['paired'].setdefault(alternative + '_vs_' + reference, {})[h] = {
                    'mean_future_iou_delta': float(np.mean(diffs)),
                    'query_higher': sum(d > 1e-8 for d in diffs),
                    'query_lower': sum(d < -1e-8 for d in diffs),
                    'query_equal': sum(abs(d) <= 1e-8 for d in diffs)}
        events.append(item)
    summary = {}
    for alternative, reference in comparisons:
        key = alternative + '_vs_' + reference
        summary[key] = {}
        for h in ['3', '32']:
            diffs = [e['paired'][key][h]['mean_future_iou_delta'] for e in events]
            summary[key][h] = {'equal_event_mean_future_iou_delta_percentage_points': float(np.mean(diffs) * 100),
                               'events_higher': sum(d > 1e-8 for d in diffs),
                               'events_lower': sum(d < -1e-8 for d in diffs),
                               'events_equal': sum(abs(d) <= 1e-8 for d in diffs)}
    eligible = [q for q in rows if q['query_raw_write_eligible']]
    return {'events': len(events), 'queries': len(rows), 'event_rows': events, 'equal_event_comparisons': summary,
            'equal_event_mean_raw_role_fractions': {role: float(np.mean([e['raw_reference_query_roles'][role] / e['correlated_queries'] for e in events])) for role in roles},
            'events_containing_raw_role': {role: sum(e['raw_reference_query_roles'][role] > 0 for e in events) for role in roles},
            'actual_write_eligible_queries': len(eligible),
            'eligible_write_pause_future_deltas': [{
                'sequence': q['sequence'], 'event_id': q['event_id'], 'query_frame': q['query_frame'],
                'query_raw_iou': q['controls'][0]['query_output_iou'],
                'pause_minus_raw_future_iou': {h: next(c for c in q['controls'] if c['name'] == 'pause')['horizons'][h]['mean_future_iou'] -
                                              next(c for c in q['controls'] if c['name'] == 'raw')['horizons'][h]['mean_future_iou'] for h in ['3', '32']}}
                for q in eligible],
            'interpretation': 'TRAIN causal diagnostic only, no benchmark scores or learned trusted gate. Raw query commit is a control; actual policy may pause. Recovered requires queryIoU<.2 and three future frames IoU>=.5 within horizon; unrecovered is horizon-limited.'}


def main():
    expected = json.loads((FOLDER / 'prepared_geometry_commit_train_jobs_cpu.json').read_text())['jobs']
    expected_keys = {(j['sequence'], j['query_frame']) for j in expected}
    rows = []
    for part, count in [('sanity', 1), ('gpu0', 24), ('gpu3', 23)]:
        out = BASE / part
        assert (out / 'COMPLETE').is_file()
        d = json.loads((out / 'events.json').read_text())
        assert d['queries'] == count and d['status'] == 'COMPLETE_MATCHED_QUERY_COMMIT_ROLLOUTS'
        for index, row in enumerate(d['results']):
            with np.load(out / f'event_{index:04d}.npz') as arrays:
                query = arrays['raw_boxes_xyxy'][0]
                assert all(np.array_equal(query, arrays[name + '_boxes_xyxy'][0]) for name in ['pause', 'geometry_C1', 'pause_geometry_C1'])
                assert all(len(arrays[name + '_iou']) == 33 for name in ['raw', 'pause', 'geometry_C1', 'pause_geometry_C1'])
            rows.append(row)
    assert len(rows) == 48 and len({(q['sequence'], q['query_frame']) for q in rows}) == 48
    assert {(q['sequence'], q['query_frame']) for q in rows} == expected_keys
    result = analyze(rows)
    assert result['events'] == 16
    target = BASE / 'merged48_event_analysis.json'
    assert not target.exists()
    target.write_text(json.dumps(result | {'status': 'COMPLETE_TRAIN48_QUERIES16_EVENTS', 'results': rows}, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['event_rows', 'eligible_write_pause_future_deltas']}))


if __name__ == '__main__':
    main()
