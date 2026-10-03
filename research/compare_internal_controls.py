"""Full original98 held-out localization controls; never native benchmark scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from .collect_candidate_metrics import rates, summarize
from .collect_core_metrics import failure_events, localization_quality, save_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--runs', nargs=5, required=True,
                        help='Run roots in order: baseline, C1, ABC initial, ABC last, ABC box-only')
    parser.add_argument('--abc-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    split = json.loads(Path(args.split).read_text())
    names = sorted(split['validation'])
    assert len(names) == 98 and not set(names) & set(split['train'])
    previous = json.loads(Path(args.abc_report).read_text())
    assert previous['completed'] and not previous['official_tracking_accuracy']
    labels = ('baseline', 'c1', 'initial', 'last', 'last_box_only')
    expected_variant = ('baseline', 'c1', 'abc', 'abc', 'abc_box_only')
    report = {'completed': True, 'official_tracking_accuracy': False, 'sequences': len(names),
              'protocol': 'all original98 TRAIN-held-out videos/all frames; same localization_quality and failure_events as previous ABC report; initialization and unknown GT excluded',
              'checkpoint_policy': 'ABC epoch30 already locked before official testing; controls do not reselect checkpoints',
              'args': vars(args), 'variants': {}, 'paired': {}, 'abc_mechanisms_report': args.abc_report}
    rows, means = [], {}
    for index, (label, run) in enumerate(zip(labels, args.runs)):
        root = Path(run) / 'predictions'
        config = json.loads((root / 'inference_config.json').read_text())
        receipt = json.loads((root / 'inference_completion.json').read_text())
        assert receipt['completed'] and receipt['smoke_only'] and receipt['sequences'] == len(names)
        assert config['validation_split'] == args.split and config['root'] == args.root
        assert config['dataset'] == 'lasher' and config['variant'] == expected_variant[index]
        assert config['limit_sequences'] == config['max_frames'] == 0
        if index >= 2:
            assert config['sequence_offset'] == 0
        assert {p.stem for p in root.glob('*.txt')} == set(names)
        assert {r['sequence'] for r in receipt['records']} == set(names)
        sequence_means, failures, latency_arrays, counters, bins = [], [], [], [], []
        iou_sum, valid_frames, frames, failed_frames = 0., 0, 0, 0
        for name in names:
            gt = np.loadtxt(Path(args.root) / name / 'init.txt', delimiter=',', ndmin=2)
            prediction = np.loadtxt(root / (name + '.txt'), ndmin=2)
            assert prediction.shape == gt.shape and prediction.shape[1] == 4 and np.isfinite(prediction).all()
            quality, valid = localization_quality(prediction, gt, 'lasher')
            assert valid.any()
            events = failure_events(quality, valid)
            failures.extend({'sequence': name, **event} for event in events)
            mean = float(quality[valid].mean())
            sequence_means.append(mean)
            iou_sum += float(quality[valid].sum())
            valid_frames += int(valid.sum())
            frames += len(prediction)
            failed = int((valid & (quality < .2)).sum())
            failed_frames += failed
            latency = np.load(root / (name + '_latency.npy'))
            assert latency.shape == (len(prediction) - 1,) and np.isfinite(latency).all() and (latency > 0).all()
            latency_arrays.append(latency)
            row = {'variant': label, 'sequence': name, 'frames': len(prediction),
                   'valid_tracking_frames': int(valid.sum()), 'mean_valid_iou': mean,
                   'failed_tracking_frames': failed, 'failure_events': len(events),
                   'recovered_events': sum(e['recovered'] for e in events)}
            if index < 2:
                with np.load(root / (name + '_candidates.npz')) as timeline:
                    counts, calibration = summarize(timeline, prediction, gt, 'lasher')
                actual = next(r for r in receipt['records'] if r['sequence'] == name)
                assert counts['template_updates'] == actual['template_updates']
                assert counts['alternative_selections'] == actual['alternative_selections']
                counters.append(counts)
                bins.append(calibration)
                row.update(counts | rates(counts))
            rows.append(row)
        assert frames == receipt['frames'] == 49418
        means[label] = np.asarray(sequence_means)
        latency = np.concatenate(latency_arrays)
        delays = [e['delay_frames'] for e in failures if e['recovered']]
        variant = {'frames': frames, 'valid_tracking_frames': valid_frames,
                   'sequence_mean_iou': float(means[label].mean()), 'frame_mean_iou': iou_sum / valid_frames,
                   'failed_tracking_frames': failed_frames, 'failure_events': failures,
                   'recovery_rate': len(delays) / len(failures) if failures else None,
                   'median_recovery_delay': float(np.median(delays)) if delays else None,
                   'efficiency': {'instrumented_tracking_fps': len(latency) / float(latency.sum()),
                                  'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
                                  'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
                                  'peak_cuda_allocated_mib': receipt['peak_cuda_allocated_mib']},
                   'inference_config': config}
        if index < 2:
            totals = {key: sum(c[key] for c in counters) for key in counters[0]}
            assert totals['valid_tracking_frames'] == valid_frames
            variant['counters'], variant['rates'] = totals, rates(totals)
            variant['current_quality_calibration_bins'] = [
                {'bin': b, **{key: sum(c[b][key] for c in bins)
                             for key in ('count', 'prediction_sum', 'actual_iou_sum')}} for b in range(10)]
        else:
            assert abs(variant['sequence_mean_iou'] - previous['variants'][label]['sequence_mean_iou']) < 1e-12
            assert abs(variant['frame_mean_iou'] - previous['variants'][label]['frame_mean_iou']) < 1e-12
        report['variants'][label] = variant
    resamples = np.random.default_rng(42).integers(0, len(names), size=(5000, len(names)))
    pairs = [(label, 'baseline') for label in labels[1:]] + [('last', 'c1'), ('last', 'initial'), ('last', 'last_box_only')]
    for label, reference in pairs:
        delta = (means[label] - means[reference]) * 100
        report['paired'][label + '_minus_' + reference] = {
            'reference': reference, 'variant': label, 'mean_iou_delta_percentage_points': float(delta.mean()),
            'paired_sequence_95_ci': np.percentile(delta[resamples].mean(1), [2.5, 97.5]).tolist(),
            'improved_sequences': int((delta > 0).sum()), 'worsened_sequences': int((delta < 0).sum()),
            'tied_sequences': int((delta == 0).sum()),
            'scope': '5000 paired98-sequence resamples/seed42/fixed checkpoints; not native PR/SR or training-seed uncertainty'}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'full_internal_controls.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    save_csv(output / 'per_sequence.csv', rows)
    (output / 'internal_controls_completed.txt').write_text('All five full original98 runs verified; internal localization diagnostics only.\n')
    print(json.dumps({'completed': True, 'sequence_mean_iou': {k: v['sequence_mean_iou'] for k, v in report['variants'].items()}}))


if __name__ == '__main__':
    main()
