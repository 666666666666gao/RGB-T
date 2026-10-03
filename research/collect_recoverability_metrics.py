"""Full-video recoverability diagnostics against GT read only after inference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .collect_abc_metrics import aggregate_motion, memory_labels, motion_statistics, quality_at, xywh
from .collect_candidate_metrics import ground_truth
from .collect_core_metrics import EXPECTED, failure_events, localization_quality


def summarize(timeline, prediction, gt, dataset):
    timeline = {key: timeline[key] for key in timeline.files}
    n = len(prediction) - 1
    boxes = xywh(timeline['boxes_xyxy']).reshape(n, 35, 4)
    valid = timeline['valid'].reshape(n, 35)
    choice, keep = timeline['choice'], timeline['original_choice']
    assert boxes.shape == (n, 35, 4) and valid[:, :5].any(1).all()
    assert valid[np.arange(n), choice].all() and (valid.sum(1) <= 10).all()
    assert all(np.isfinite(value).all() for value in timeline.values())
    assert np.allclose(boxes[np.arange(n), choice], prediction[1:], atol=.000501, rtol=0)
    quality = []
    for candidate in range(35):
        q, known_gt = localization_quality(np.concatenate((prediction[:1], boxes[:, candidate])), gt, dataset)
        quality.append(q[1:])
    quality, known_gt = np.stack(quality, 1), known_gt[1:]
    selected = quality[np.arange(n), choice]
    original = quality[np.arange(n), keep]
    local_recalled = np.where(valid[:, :5], quality[:, :5], -1).max(1) >= .5
    recalled = np.where(valid, quality, -1).max(1) >= .5
    changed = choice != keep
    failed = known_gt & (selected < .2)
    writes, paused = timeline['template_updated'], timeline['pause']
    raw = timeline['raw_score'].reshape(n, 35)[np.arange(n), choice]
    assert np.array_equal(writes, (raw > .84) & ~paused)
    assert np.array_equal(timeline['extra_executed'], valid[:, 5:].any(1))
    assert (timeline['extra_executed'] <= timeline['search_requested']).all()
    counts = {'tracking_frames': n, 'valid_tracking_frames': int(known_gt.sum()),
              'candidate_evaluations': int(valid.sum()),
              'local_candidate_recalled_frames': int((local_recalled & known_gt).sum()),
              'budget_candidate_recalled_frames': int((recalled & known_gt).sum()),
              'local_missing_frames': int((~local_recalled & known_gt).sum()),
              'missing_correct_candidates_reintroduced': int((~local_recalled & recalled & known_gt).sum()),
              'reintroduced_candidates_selected_correctly': int((~local_recalled & recalled & known_gt & (selected >= .5)).sum()),
              'c1_same_state_failed_frames': int((known_gt & (original < .2)).sum()),
              'c1_same_state_correct_frames': int((known_gt & (original >= .5)).sum()),
              'selected_failed_frames': int(failed.sum()), 'changed_candidate_indices': int(changed.sum()),
              'valid_changed_candidate_indices': int((changed & known_gt).sum()),
              'same_state_rescues': int((known_gt & (original < .2) & (selected >= .5)).sum()),
              'same_state_harms': int((known_gt & (original >= .5) & (selected < .2)).sum()),
              'same_state_iou_improvements': int((known_gt & (selected > original + 1e-8)).sum()),
              'same_state_iou_degradations': int((known_gt & (selected < original - 1e-8)).sum()),
              'requested_extra_searches': int(timeline['search_requested'].sum()),
              'extra_visual_forwards': int(timeline['extra_executed'].sum()),
              'searches_without_new_correct_candidate': int((timeline['extra_executed'] & known_gt & ~(~local_recalled & recalled)).sum()),
              'requested_extra_area_sum_pixels_squared': float(timeline['requested_extra_area'].sum()),
              'template_updates': int(writes.sum()), 'known_template_updates': int((writes & known_gt).sum()),
              'wrong_template_updates_localization_proxy': int((writes & failed).sum()),
              'paused_query_writes': int(paused.sum()), 'known_paused_query_writes': int((paused & known_gt).sum()),
              'wrong_query_writes_prevented_localization_proxy': int((paused & failed).sum()),
              'correct_query_writes_prevented': int((paused & known_gt & (selected >= .5)).sum()),
              'known_active_template_frames': 0, 'wrong_active_template_frames_localization_proxy': 0}
    for modality in ('rgb', 'tir'):
        for kind in ('target_write', 'target_read'):
            for label in ('known', 'wrong'):
                counts[f'memory_{modality}_{kind}_{label}_mass'] = 0.
    # Identity target slots initialize from the trusted first-frame anchor; context slots start empty.
    known_mass, wrong_mass = np.zeros((4, 2)), np.zeros((4, 2))
    known_mass[:2] = 1
    interventions = []
    for index in range(n):
        frame = index + 1
        source_frame = int(timeline['prior_template_frame'][index])
        assert 0 <= source_frame < frame
        source_quality, source_known = quality_at(timeline['prior_template_box'][index], source_frame, gt, dataset)
        if known_gt[index] and source_known:
            counts['known_active_template_frames'] += 1
            counts['wrong_active_template_frames_localization_proxy'] += int(source_quality < .2)
        region, candidate = divmod(int(choice[index]), 5)
        selected_box = timeline['boxes_xyxy'][index, region, candidate]
        known, wrong = memory_labels(selected_box, frame, gt, dataset)
        read_sources = timeline['target_source_indices'][index, region, candidate]
        assert (read_sources >= 0).all() and (read_sources <= 2).all()
        for modality, name in enumerate(('rgb', 'tir')):
            source = int(read_sources[modality])
            if known[modality]:
                counts[f'memory_{name}_target_read_known_mass'] += 1. if source == 0 else float(known_mass[source-1, modality])
                counts[f'memory_{name}_target_read_wrong_mass'] += 0. if source == 0 else float(wrong_mass[source-1, modality])
        rate = timeline['memory_write_rates'][index]
        assert rate.shape == (4, 2) and (rate >= 0).all() and (rate <= 1).all()
        assert np.array_equal(rate[:2].sum(0) > 0, timeline['memory_target_commit'][index])
        if not writes[index]:
            assert not rate[:2].any()
        for modality, name in enumerate(('rgb', 'tir')):
            mass = float(rate[:2, modality].sum())
            counts[f'memory_{name}_target_write_known_mass'] += mass * known[modality]
            counts[f'memory_{name}_target_write_wrong_mass'] += mass * wrong[modality]
        known_mass = (1-rate) * known_mass + rate * known[None]
        wrong_mass = (1-rate) * wrong_mass + rate * wrong[None]
        if changed[index] or paused[index] or timeline['search_requested'][index]:
            interventions.append({'frame_zero_based': frame, 'region': region, 'candidate': candidate,
                                  'searched_region': int(timeline['searched_region'][index]),
                                  'extra_executed': bool(timeline['extra_executed'][index]),
                                  'pause': bool(paused[index]), 'gt_known': bool(known_gt[index]),
                                  'same_state_c1_iou': float(original[index]) if known_gt[index] else None,
                                  'selected_iou': float(selected[index]) if known_gt[index] else None})
    forecast = {'lasher': motion_statistics(timeline, gt)} if dataset == 'lasher' else {
        modality: motion_statistics(timeline, target) for modality, target in gt.items()}
    prediction_quality = timeline['current_quality'].reshape(n, 35)[valid & known_gt[:, None]]
    target_quality = quality[valid & known_gt[:, None]]
    bins = []
    for index in range(10):
        mask = np.minimum((prediction_quality * 10).astype(int), 9) == index
        bins.append({'count': int(mask.sum()), 'prediction_sum': float(prediction_quality[mask].sum()),
                     'actual_iou_sum': float(target_quality[mask].sum())})
    counts['candidate_quality_squared_error_sum'] = float(((prediction_quality-target_quality)**2).sum())
    counts['known_candidate_evaluations'] = len(prediction_quality)
    return counts, forecast, bins, interventions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['lasher', 'rgbt234'], required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--split', help='Original98 TRAIN-held-out split; without it native full dataset is required.')
    parser.add_argument('--runs', nargs='+', required=True, help='New method direct prediction directories.')
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--references', nargs='+', required=True, help='Already completed baseline/C1 direct prediction directories.')
    parser.add_argument('--reference-labels', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert len(args.runs) == len(args.labels) and len(args.references) == len(args.reference_labels)
    assert len(set(args.labels + args.reference_labels)) == len(args.labels + args.reference_labels)
    names = sorted(p.name for p in Path(args.root).iterdir() if p.is_dir())
    if args.split:
        split = json.loads(Path(args.split).read_text())
        assert args.dataset == 'lasher' and not set(split['train']) & set(split['validation'])
        names = sorted(split['validation'])
        assert len(names) == 98
    else:
        assert len(names) == EXPECTED[args.dataset][0]
    report = {'completed': True, 'native_accuracy_completed': False, 'sequences': len(names),
              'scope': 'full original98 internal videos' if args.split else 'full native dataset mechanism diagnostics; native metrics collected separately',
              'definitions': {'failure': '3 consecutive known frames IoU<.2; recovery3 known frames IoU>=.5',
                              'candidate_recall': 'IoU>=.5 among valid actual proposals; original5 vs original plus executed extra5, same pre-action state',
                              'harm_rescue': 'same-state frozen C1 selection versus chosen current box; not independent future-rollout causality',
                              'memory_mass': 'target slot EMA localization provenance and selected maximum-support source; normalized feature purity not asserted',
                              'template_pollution': 'actual pre-frame template source localization; unknownGT excluded; no semantic distractor labels',
                              'motion': 'frozen bootstrap motion forecast, sensor-specific labels; not new motion model training',
                              'timing': 'instrumented tracking including decode/crop/update; concurrent-load comparisons descriptive'},
              'args': vars(args), 'variants': {}, 'paired': {}}
    per_sequence, all_events, means = [], [], {}
    for label, run in zip(args.reference_labels + args.labels, args.references + args.runs):
        path = Path(run)
        config = json.loads((path / 'inference_config.json').read_text())
        receipt = json.loads((path / 'inference_completion.json').read_text())
        assert receipt['completed'] and receipt['sequences'] == len(names)
        assert {row['sequence'] for row in receipt['records']} == set(names)
        assert {p.stem for p in path.glob('*.txt')} == set(names)
        assert config['root'] == args.root and config['dataset'] == args.dataset
        assert config['validation_split'] == args.split
        assert config['limit_sequences'] == config['max_frames'] == 0 and config.get('sequence_offset', 0) == 0
        assert receipt['smoke_only'] == bool(args.split)
        counts, motion, bins, events, latency_arrays = [], [], [], [], []
        seq_means, frame_count, valid_count, iou_sum, failures = [], 0, 0, 0., []
        for name in names:
            gt = ground_truth(args.root, name, args.dataset)
            frames = len(gt if args.dataset == 'lasher' else gt['visible'])
            prediction = np.loadtxt(path / (name + '.txt'), ndmin=2)
            assert prediction.shape == (frames, 4) and np.isfinite(prediction).all()
            quality, known = localization_quality(prediction, gt, args.dataset)
            assert known.any()
            seq_means.append(float(quality[known].mean()))
            frame_count += frames
            valid_count += int(known.sum())
            iou_sum += float(quality[known].sum())
            failed = failure_events(quality, known)
            failures.extend({'sequence': name, **event} for event in failed)
            row = {'variant': label, 'sequence': name, 'frames': frames, 'valid_frames': int(known.sum()),
                   'mean_valid_iou': seq_means[-1], 'failed_frames': int((known & (quality < .2)).sum()),
                   'failure_events': len(failed), 'recovered_events': sum(event['recovered'] for event in failed)}
            actual = next(record for record in receipt['records'] if record['sequence'] == name)
            if label in args.labels:
                with np.load(path / (name + '_recoverability_decisions.npz')) as timeline:
                    c, forecast, calibration, interventions = summarize(timeline, prediction, gt, args.dataset)
                for counter, stat in (('template_updates', 'template_updates'), ('extra_visual_forwards', 'extra_visual_forwards'),
                                      ('changed_candidate_indices', 'changed_candidate_indices'), ('paused_query_writes', 'paused_query_writes')):
                    assert c[counter] == actual['stats'][stat]
                counts.append(c)
                motion.append(forecast)
                bins.append(calibration)
                events.extend({'sequence': name, **event} for event in interventions)
                row.update(c)
            latency = np.load(path / (name + '_latency.npy'))
            assert latency.shape == (frames-1,) and np.isfinite(latency).all() and (latency > 0).all()
            latency_arrays.append(latency)
            per_sequence.append(row)
        assert frame_count == receipt['frames']
        if not args.split:
            assert frame_count == EXPECTED[args.dataset][1]
        latency = np.concatenate(latency_arrays)
        delays = [event['delay_frames'] for event in failures if event['recovered']]
        value = {'frames': frame_count, 'valid_frames': valid_count, 'sequence_mean_iou': float(np.mean(seq_means)),
                 'frame_mean_iou': iou_sum/valid_count, 'failure_events': failures,
                 'recovery_rate': len(delays)/len(failures) if failures else None,
                 'median_recovery_delay_frames': float(np.median(delays)) if delays else None,
                 'efficiency': {'instrumented_tracking_fps': len(latency)/float(latency.sum()),
                                'latency_p50_ms': float(np.percentile(latency, 50)*1000),
                                'latency_p95_ms': float(np.percentile(latency, 95)*1000),
                                'peak_cuda_allocated_mib': receipt['peak_cuda_allocated_mib'],
                                'peak_cuda_reserved_mib': receipt['peak_cuda_reserved_mib']}, 'inference_config': config}
        if counts:
            totals = {key: sum(c[key] for c in counts) for key in counts[0]}
            assert totals['valid_tracking_frames'] == valid_count
            value['counters'] = totals
            value['rates'] = {'local_recall': totals['local_candidate_recalled_frames']/valid_count,
                              'budget_recall': totals['budget_candidate_recalled_frames']/valid_count,
                              'missing_candidate_reintroduction_rate': totals['missing_correct_candidates_reintroduced']/totals['local_missing_frames'] if totals['local_missing_frames'] else None,
                              'wrong_template_write_rate': totals['wrong_template_updates_localization_proxy']/totals['known_template_updates'] if totals['known_template_updates'] else None,
                              'misintervention_rate_on_correct_same_state_c1': totals['same_state_harms']/totals['c1_same_state_correct_frames'] if totals['c1_same_state_correct_frames'] else None}
            value['memory_target_localization_provenance'] = {f'{modality}_{kind}_wrong_mass_fraction':
                totals[f'memory_{modality}_{kind}_wrong_mass']/totals[f'memory_{modality}_{kind}_known_mass']
                if totals[f'memory_{modality}_{kind}_known_mass'] else None
                for modality in ('rgb', 'tir') for kind in ('target_write', 'target_read')}
            value['current_quality_calibration_bins'] = [{'bin': index, **{key: sum(b[index][key] for b in bins)
                                                        for key in bins[0][index]}} for index in range(10)]
            value['motion'] = {modality: aggregate_motion([record[modality] for record in motion]) for modality in motion[0]}
            all_events.extend({'variant': label, **event} for event in events)
        report['variants'][label] = value
        means[label] = np.asarray(seq_means)
    resamples = np.random.default_rng(42).integers(0, len(names), size=(5000, len(names)))
    for reference in args.reference_labels:
        for label in args.labels:
            delta = (means[label]-means[reference])*100
            report['paired'][label + '_vs_' + reference] = {
                'mean_iou_delta_percentage_points': float(delta.mean()),
                'paired_sequence_95_ci': np.percentile(delta[resamples].mean(1), [2.5, 97.5]).tolist(),
                'improved_sequences': int((delta > 0).sum()), 'worsened_sequences': int((delta < 0).sum()),
                'tied_sequences': int((delta == 0).sum()), 'bootstrap_resamples': 5000,
                'scope': 'fixed checkpoint sequence resampling, descriptive development results; not native metrics or training-seed uncertainty'}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'full_recoverability_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (output / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in per_sequence for key in row}))
        writer.writeheader()
        writer.writerows(per_sequence)
    with (output / 'interventions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('variant', 'sequence', 'frame_zero_based', 'region', 'candidate',
            'searched_region', 'extra_executed', 'pause', 'gt_known', 'same_state_c1_iou', 'selected_iou'))
        writer.writeheader()
        writer.writerows(all_events)
    (output / 'recoverability_metrics_completed.txt').write_text('full GT localization and mechanism diagnostics completed; native accuracy separate\n')
    print(json.dumps({'completed': True, 'native_accuracy_completed': False,
                      'sequence_mean_iou': {label: value['sequence_mean_iou'] for label, value in report['variants'].items()}}))


if __name__ == '__main__':
    main()
