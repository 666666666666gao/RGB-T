"""Full-sequence core benchmark report from completed online predictions.

Official metrics/curves use the original rgbt functions. Additional failure and
paired-frame diagnostics are explicitly localization measurements, not inferred
semantic identities. No truncated or incomplete inference is accepted here.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation import LasHeREvaluation, RGBT234Evaluation


EXPECTED = {'lasher': (245, 220703), 'rgbt234': (234, 116649)}
METRICS = {'lasher': ('NPR', 'PR', 'SR'), 'rgbt234': ('MPR', 'MSR')}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=tuple(EXPECTED), required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--variants', nargs='+', required=True, help='First variant is the paired baseline.')
    p.add_argument('--runs', nargs='+', required=True, help='Run folders with predictions/inference_completion.json.')
    p.add_argument('--output', required=True)
    return p.parse_args()


def overlap_xywh(prediction, target):
    # Same inclusive-width geometry as rgbt IoU, vectorized for diagnostics.
    pwh, twh = np.maximum(prediction[:, 2:], 0), np.maximum(target[:, 2:], 0)
    left = np.maximum(prediction[:, :2], target[:, :2])
    right = np.minimum(prediction[:, :2] + pwh, target[:, :2] + twh)
    intersection = np.maximum(right - left, 0).prod(1)
    union = pwh.prod(1) + twh.prod(1) - intersection
    return intersection / np.maximum(union, 1e-8)


def localization_quality(boxes, ground_truth, dataset):
    if dataset == 'lasher':
        valid = np.isfinite(ground_truth).all(1) & (ground_truth[:, 2:] > 0).all(1)
        quality = overlap_xywh(boxes, ground_truth)
    else:
        qualities, validity = [], []
        for modality in ('visible', 'infrared'):
            gt = ground_truth[modality]
            modality_valid = np.isfinite(gt).all(1) & (gt[:, 2:] > 0).all(1)
            qualities.append(np.where(modality_valid, overlap_xywh(boxes, gt), -1.))
            validity.append(modality_valid)
        quality = np.maximum(*qualities)
        valid = np.logical_or(*validity)
    valid[0] = False
    assert np.isfinite(quality[valid]).all()
    return quality, valid


def failure_events(quality, valid):
    # Failure: 3 observed valid frames IoU<.2. Recovery: 3 valid frames IoU>=.5.
    # Invalid GT interrupts confirmation streaks; delays use actual frame indices.
    events, active = [], None
    failed_streak, good_streak = 0, 0
    for frame in range(1, len(quality)):
        if not valid[frame]:
            failed_streak, good_streak = 0, 0
            continue
        if active is None:
            failed_streak = failed_streak + 1 if quality[frame] < .2 else 0
            if failed_streak == 3:
                active = {'start_frame_zero_based': frame - 2, 'confirmed_frame_zero_based': frame,
                          'recovered': False, 'recovery_start_frame_zero_based': None, 'delay_frames': None}
                good_streak = 0
        else:
            good_streak = good_streak + 1 if quality[frame] >= .5 else 0
            if good_streak == 3:
                active.update(recovered=True, recovery_start_frame_zero_based=frame - 2,
                              delay_frames=frame - 2 - active['start_frame_zero_based'])
                events.append(active)
                active, failed_streak, good_streak = None, 0, 0
    if active is not None:
        events.append(active)
    return events


def metric_value(metric, curve):
    return float(curve.mean()) if metric in ('SR', 'MSR') else float(curve.mean(0)[20])


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = arguments()
    assert len(args.variants) == len(args.runs) and len(set(args.variants)) == len(args.variants)
    expected_sequences, expected_frames = EXPECTED[args.dataset]
    runs = [Path(run) for run in args.runs]
    roots = [str(run / 'predictions') for run in runs]
    if args.dataset == 'lasher':
        evaluator = LasHeREvaluation(args.variants, roots).evaluator
    else:
        evaluator = RGBT234Evaluation(args.variants, roots, gt_path=args.data_root).evaluator
    names = list(evaluator.ALL)
    assert len(names) == expected_sequences
    data = Path(args.data_root)
    assert {path.name for path in data.iterdir() if path.is_dir()} == set(names)
    gt = {}
    for name in names:
        if args.dataset == 'lasher':
            actual = np.loadtxt(data / name / 'init.txt', delimiter=',', dtype=np.float32, ndmin=2)
            assert np.array_equal(actual, np.asarray(evaluator.seqs_gt[name], dtype=np.float32)), name
        else:
            actual = {m: np.loadtxt(data / name / (m + '.txt'), delimiter=',', dtype=np.float32, ndmin=2)
                      for m in ('visible', 'infrared')}
            for modality, annotations in actual.items():
                assert np.array_equal(annotations, np.asarray(evaluator.seqs_gt[name][modality], dtype=np.float32)), (name, modality)
            assert len(actual['visible']) == len(actual['infrared']), name
        gt[name] = actual
    frame_count = sum(len(annotations if args.dataset == 'lasher' else annotations['visible']) for annotations in gt.values())
    assert frame_count == expected_frames
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset': args.dataset,
              'actual_data_root': str(data), 'official_gt_source': evaluator.gt_path,
              'all_actual_ground_truth_verified': True, 'sequences': expected_sequences, 'frames': expected_frames,
              'official_protocol': 'Original rgbt metric functions and result rounding; LasHeR call order NPR, PR, SR.',
              'diagnostic_protocol': {'input_boxes': 'saved raw XYWH; GT matches the actual dataset',
                                      'initialization_excluded': True, 'valid_gt': 'finite and positive width/height',
                                      'rgbt234_quality': 'max RGB/TIR IoU over valid modality annotations',
                                      'failure': '3 consecutive valid IoU<0.2 frames',
                                      'recovery': '3 consecutive valid IoU>=0.5 frames',
                                      'delay': 'actual frame indices; unknown GT interrupts streaks',
                                      'event_is_localization_failure': 'not semantic identity ground truth'},
              'variants': {}, 'paired_against': args.variants[0], 'paired_frame_diagnostics': {},
              'mechanism_measurements_pending': ['Candidate Recall@K and mis-rejected-candidate recovery need candidate timelines.',
                                                 'State/identity-switch events and memory contamination need update/branch timelines.',
                                                 'A memory-budget and B forecast calibration curves require the corresponding modules.',
                                                 'Existing runs do not record inference peak CUDA memory; final instrumented runs must.']}
    sequence_rows, attribute_rows, overall_rows = [], [], []
    all_quality, all_valid = {}, {}
    for variant, run in zip(args.variants, runs):
        prediction_root = run / 'predictions'
        receipt = json.loads((prediction_root / 'inference_completion.json').read_text())
        assert receipt['completed'] and not receipt['smoke_only']
        assert receipt['sequences'] == expected_sequences and receipt['frames'] == expected_frames
        assert {p.stem for p in prediction_root.glob('*.txt')} == set(names)
        latencies, qualities, validities, events = [], {}, {}, []
        sequence_diagnostics = {}
        for name in names:
            boxes = np.loadtxt(prediction_root / (name + '.txt'), dtype=np.float64, ndmin=2)
            n = len(gt[name] if args.dataset == 'lasher' else gt[name]['visible'])
            assert boxes.shape == (n, 4) and np.isfinite(boxes).all(), (variant, name)
            latency = np.load(prediction_root / (name + '_latency.npy'))
            assert latency.shape == (n - 1,) and np.isfinite(latency).all() and (latency > 0).all()
            latencies.append(latency)
            quality, valid = localization_quality(boxes, gt[name], args.dataset)
            qualities[name], validities[name] = quality, valid
            failures = failure_events(quality, valid)
            events.extend({'sequence': name, **event} for event in failures)
            sequence_diagnostics[name] = {'valid_tracking_frames': int(valid.sum()),
                                          'failed_tracking_frames': int((valid & (quality < .2)).sum()),
                                          'failure_events': len(failures),
                                          'recovered_events': sum(event['recovered'] for event in failures),
                                          'tracking_fps': len(latency) / float(latency.sum())}
        # The native functions mutate some LasHeR results. Preserve official ordering.
        native = {metric: getattr(evaluator, metric)(variant) for metric in METRICS[args.dataset]}
        curves = {metric: np.asarray(result[1]) for metric, result in native.items()}
        overall = {metric: float(result[0]) * 100 for metric, result in native.items()}
        for metric, curve in curves.items():
            assert curve.shape[0] == expected_sequences and np.isfinite(curve).all()
            assert np.isclose(metric_value(metric, curve) * 100, overall[metric])
        latency = np.concatenate(latencies)
        assert len(latency) == expected_frames - expected_sequences
        delays = [event['delay_frames'] for event in events if event['recovered']]
        efficiency = {'tracking_fps': len(latency) / float(latency.sum()),
                      'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
                      'latency_p90_ms': float(np.percentile(latency, 90) * 1000),
                      'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
                      'latency_p99_ms': float(np.percentile(latency, 99) * 1000),
                      'excludes_first_frame_initialization': True,
                      'includes_decode_crop_forward_selection_update': True}
        diagnostic = {'valid_tracking_frames': sum(v['valid_tracking_frames'] for v in sequence_diagnostics.values()),
                      'failed_tracking_frames': sum(v['failed_tracking_frames'] for v in sequence_diagnostics.values()),
                      'failure_events': len(events), 'recovered_events': len(delays),
                      'recovery_rate': len(delays) / len(events) if events else None,
                      'median_recovery_delay_frames': float(np.median(delays)) if delays else None,
                      'unrecovered_events': len(events) - len(delays)}
        for row, name in enumerate(names):
            sequence_rows.append({'variant': variant, 'sequence': name,
                                  **{m: metric_value(m, c[row:row + 1]) * 100 for m, c in curves.items()},
                                  **sequence_diagnostics[name]})
        index = {name: i for i, name in enumerate(names)}
        attribute_metrics = {}
        for attribute in evaluator.get_attr_list():
            group = list(getattr(evaluator, attribute))
            rows = [index[name] for name in group]
            assert rows
            scores = {m: metric_value(m, c[rows]) * 100 for m, c in curves.items()}
            attribute_metrics[attribute] = {'sequences': len(rows), **scores}
            attribute_rows.append({'variant': variant, 'attribute': attribute, 'sequences': len(rows), **scores})
        report['variants'][variant] = {'overall_metrics_percent': overall, 'efficiency': efficiency,
                                       'localization_diagnostics': diagnostic, 'failure_event_records': events,
                                       'attributes': attribute_metrics,
                                       'mean_curves': {m: {'thresholds': getattr(evaluator, m + '_fun').thr.tolist(),
                                                           'values': c.mean(0).tolist()} for m, c in curves.items()},
                                       'original_inference_config': json.loads((prediction_root / 'inference_config.json').read_text())}
        overall_rows.append({'variant': variant, **overall, **{k: v for k, v in efficiency.items() if not isinstance(v, bool)}, **diagnostic})
        all_quality[variant], all_valid[variant] = qualities, validities
    baseline = args.variants[0]
    for variant in args.variants[1:]:
        paired = dict(baseline_correct_frames=0, baseline_failed_frames=0,
                      baseline_correct_but_variant_failed_frames=0, baseline_failed_but_variant_correct_frames=0,
                      valid_tracking_frames=0)
        for name in names:
            valid = all_valid[baseline][name] & all_valid[variant][name]
            correct = valid & (all_quality[baseline][name] >= .5)
            failed = valid & (all_quality[baseline][name] < .2)
            paired['baseline_correct_frames'] += int(correct.sum())
            paired['baseline_failed_frames'] += int(failed.sum())
            paired['baseline_correct_but_variant_failed_frames'] += int((correct & (all_quality[variant][name] < .2)).sum())
            paired['baseline_failed_but_variant_correct_frames'] += int((failed & (all_quality[variant][name] >= .5)).sum())
            paired['valid_tracking_frames'] += int(valid.sum())
        paired['official_metric_delta_percentage_points'] = {
            m: report['variants'][variant]['overall_metrics_percent'][m] - report['variants'][baseline]['overall_metrics_percent'][m]
            for m in METRICS[args.dataset]}
        report['paired_frame_diagnostics'][variant] = paired
    (out / 'full_report.json').write_text(json.dumps(report, indent=2))
    save_csv(out / 'overall.csv', overall_rows)
    save_csv(out / 'per_sequence.csv', sequence_rows)
    save_csv(out / 'per_attribute.csv', attribute_rows)
    print(json.dumps({'dataset': args.dataset, 'sequences': expected_sequences, 'frames': expected_frames,
                      'overall': {v: r['overall_metrics_percent'] for v, r in report['variants'].items()},
                      'report': str(out / 'full_report.json'), 'mechanism_measurements_pending': report['mechanism_measurements_pending']}))


if __name__ == '__main__':
    main()
