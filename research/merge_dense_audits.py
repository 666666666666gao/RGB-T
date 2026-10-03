"""Aggregate completed TRAIN visual audits; these are GT-selected diagnostics."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def summarize(rows):
    local = ('replayed_original5', 'dense256', 'local_peak', 'raw5', 'hann16',
             'dense_hann5', 'dense_raw5')
    policies = ('anchor_local', 'anchor_widened', 'anchor_fixed_offset',
                'anchor_motion_0', 'anchor_motion_1', 'anchor_motion_2')
    result = {'queries': len(rows), 'local_correct_candidate_queries': {
        key: sum(row[key + '_oracle_iou'] >= .5 for row in rows) for key in local},
        'extra_search': {}}
    for key in policies:
        entries = [row['extra_search'][key] for row in rows]
        areas = [row['original_actual_crop_area_pixels'] + entry['actual_crop_area_pixels']
                 for row, entry in zip(rows, entries) if entry['usable_image_crop']]
        result['extra_search'][key] = {
            'requested_extra_visual_crops': len(rows),
            'executed_extra_visual_crops': sum(e['extra_visual_forwards_per_query'] for e in entries),
            'union_original5_plus_extra5_correct_queries': sum(e['union_original5_plus_extra5_oracle_iou'] >= .5 for e in entries),
            'union_original5_plus_dense_hann5_correct_queries': sum(e['union_original5_plus_dense_hann5_oracle_iou'] >= .5 for e in entries),
            'union_original5_plus_dense_raw5_correct_queries': sum(e['union_original5_plus_dense_raw5_oracle_iou'] >= .5 for e in entries),
            'extra_dense256_correct_queries': sum(e['extra_dense256_oracle_iou'] >= .5 for e in entries),
            'union_c1_score_selection_correct_queries': sum(e['union_score_selected_iou'] >= .5 for e in entries),
            'target_center_inside_extra_crop_queries': sum(e['target_center_inside_actual_adjusted_crop'] for e in entries),
            'mean_original_plus_extra_actual_crop_area_pixels': sum(areas) / len(areas) if areas else None,
            'queries_with_defined_actual_crop_area': len(areas),
            'maximum_original_plus_extra_candidates': 10}
    result['top_probability_motion_union_correct_queries'] = sum(
        r['extra_search'][f"anchor_motion_{r['motion_top_probability_mode']}"]['union_original5_plus_extra5_oracle_iou'] >= .5 for r in rows)
    result['any_three_motion_union_correct_queries'] = sum(
        max(r['extra_search'][f'anchor_motion_{k}']['union_original5_plus_extra5_oracle_iou'] for k in range(3)) >= .5 for r in rows)
    result['any_three_motion_budget'] = {'regions_including_original': 4, 'maximum_candidates': 20}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in args.reports]
    rows = []
    for report in reports:
        assert report['completed'] and not report['m0_only'] and not report['official_tracking_accuracy']
        assert report['protocol']['partition'] == 'train'
        assert report['protocol'] == reports[0]['protocol']
        assert len(report['rows']) == report['sampled_queries'] == report['summary']['queries']
        for row in report['rows']:
            assert row['replayed_missing_candidate'] and row['cache_choice_slot_equal']
            assert row['cache_features_max_abs_difference'] == row['cache_original5_slots_max_abs_difference_pixels'] == 0
            assert row['cache_valid_candidates'] == row['replayed_valid_candidates']
        rows.extend(report['rows'])
    identities = [(r['cache'], r['cache_row']) for r in rows]
    assert len(identities) == len(set(identities))
    counts = Counter((r['sequence'], r['query_frame']) for r in rows)
    inside = [r for r in rows if r['target_center_inside_actual_adjusted_crop']]
    outside = [r for r in rows if not r['target_center_inside_actual_adjusted_crop']]
    result = {'completed': True, 'official_tracking_accuracy': False,
              'source_reports': args.reports, 'protocol': reports[0]['protocol'],
              'weighting': 'sampled replay rows; repeated sequence-query pairs retained with multiplicities; not independent training seeds',
              'queries': len(rows), 'unique_sequence_query_pairs': len(counts),
              'repeat_rows': len(rows) - len(counts),
              'replayed_original_groups': sum(r['replayed_original_groups'] for r in reports),
              'replayed_prefix_contexts': sum(r['replayed_prefix_contexts'] for r in reports),
              'sum_gpu_job_elapsed_seconds': sum(r['elapsed_seconds'] for r in reports),
              'peak_cuda_allocated_mib': max(r['peak_cuda_allocated_mib'] for r in reports),
              'all': summarize(rows), 'inside_original_actual_crop': summarize(inside),
              'outside_original_actual_crop': summarize(outside),
              'repeated_pairs': [{'sequence': s, 'query_frame': t, 'multiplicity': n}
                                 for (s, t), n in sorted(counts.items()) if n > 1]}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train_visual_audit_merged.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    flat = []
    for row in rows:
        values = {k: v for k, v in row.items() if k != 'extra_search'}
        for policy, entry in row['extra_search'].items():
            values.update({f'{policy}.{k}': v for k, v in entry.items()})
        flat.append(values)
    columns = sorted({key for row in flat for key in row})
    with (out / 'per_query.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    (out / 'train_visual_audit_merge_completed.txt').write_text('Completed TRAIN diagnostic only; not deployed tracking accuracy.\n')
    print(json.dumps({k: result[k] for k in ('queries', 'unique_sequence_query_pairs', 'repeat_rows')}))


if __name__ == '__main__':
    main()
