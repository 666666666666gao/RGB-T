"""Complete 98-sequence TRAIN-held-out ABC comparison, never official scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .collect_abc_metrics import summarize, aggregate_motion
from .collect_candidate_metrics import rates
from .collect_core_metrics import localization_quality, failure_events, save_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert len(args.runs) == len(args.labels) and len(set(args.labels)) == len(args.labels)
    split = json.loads(Path(args.split).read_text())
    names = sorted(split['validation'])
    assert len(names) == 98 and not set(names) & set(split['train'])
    training = Path(args.training_run)
    assert (training / 'training_and_audit_completed.txt').is_file()
    audit = json.loads((training / 'reload_and_calibration.json').read_text())
    assert audit['completed']
    initial = torch.load(training / 'initial.pth', map_location='cpu', weights_only=False)
    best = torch.load(training / 'best.pth', map_location='cpu', weights_only=False)
    same_best = all(torch.equal(value, best['model'][key]) for key, value in initial['model'].items())
    report = {'completed': True, 'official_tracking_accuracy': False, 'sequences': len(names),
              'protocol': 'all original98 TRAIN-held-out videos/all frames; localization diagnostics exclude initialization and unknown GT',
              'selection_criterion': 'sequence-weighted mean valid-frame IoU of full online videos; official test never used',
              'cached_best_epoch': best['epoch'], 'cached_best_state_equals_initial': same_best,
              'training_audit': str(training / 'reload_and_calibration.json'), 'variants': {}, 'paired': {}}
    rows, sequence_means = [], {}
    for label, run in zip(args.labels, args.runs):
        root = Path(run) / 'predictions'
        config = json.loads((root / 'inference_config.json').read_text())
        receipt = json.loads((root / 'inference_completion.json').read_text())
        assert receipt['completed'] and receipt['smoke_only'] and receipt['sequences'] == len(names)
        assert config['validation_split'] == args.split and config['root'] == args.root
        assert config['dataset'] == 'lasher' and config['limit_sequences'] == config['max_frames'] == config['sequence_offset'] == 0
        assert {p.stem for p in root.glob('*.txt')} == set(names)
        counters, calibration, motion, switches, failures, latencies = [], [], [], [], [], []
        means, iou_sum, valid_frames, frame_count = [], 0., 0, 0
        for name in names:
            gt = np.loadtxt(Path(args.root) / name / 'init.txt', delimiter=',', ndmin=2)
            prediction = np.loadtxt(root / (name + '.txt'), ndmin=2)
            assert prediction.shape == gt.shape and prediction.shape[1] == 4 and np.isfinite(prediction).all()
            quality, valid = localization_quality(prediction, gt, 'lasher')
            assert valid.any()
            events = failure_events(quality, valid)
            failures.extend({'sequence': name, **event} for event in events)
            mean = float(quality[valid].mean())
            means.append(mean)
            iou_sum += float(quality[valid].sum())
            valid_frames += int(valid.sum())
            frame_count += len(prediction)
            branch_events = json.loads((root / (name + '_branch_events.json')).read_text())
            with np.load(root / (name + '_abc_decisions.npz')) as timeline:
                c, bins, switch_events, forecast = summarize(timeline, prediction, branch_events, gt, 'lasher',
                                                            config['variant'] == 'abc')
            actual = next(row for row in receipt['records'] if row['sequence'] == name)
            assert c['all_branch_updates'] == actual['branch_stats']['template_updates']
            assert c['switches'] == actual['branch_stats']['switches']
            counters.append(c)
            calibration.append(bins)
            motion.append(forecast['lasher'])
            switches.extend({'sequence': name, **event} for event in switch_events)
            latency = np.load(root / (name + '_latency.npy'))
            assert latency.shape == (len(prediction) - 1,) and np.isfinite(latency).all() and (latency > 0).all()
            latencies.append(latency)
            rows.append({'variant': label, 'sequence': name, 'frames': len(prediction),
                         'mean_valid_iou': mean, 'failure_events': len(events),
                         'recovered_events': sum(event['recovered'] for event in events), **c, **rates(c)})
        assert frame_count == receipt['frames'] and valid_frames == sum(c['valid_tracking_frames'] for c in counters)
        totals = {key: sum(c[key] for c in counters) for key in counters[0]}
        assert totals['tracking_frames'] == frame_count - len(names)
        latency = np.concatenate(latencies)
        delays = [event['delay_frames'] for event in failures if event['recovered']]
        report['variants'][label] = {
            'frames': frame_count, 'sequence_mean_iou': float(np.mean(means)), 'frame_mean_iou': iou_sum / valid_frames,
            'failure_events': failures, 'recovery_rate': len(delays) / len(failures) if failures else None,
            'median_recovery_delay': float(np.median(delays)) if delays else None,
            'counters': totals, 'rates': rates(totals), 'switch_records': switches,
            'memory_wrong_contribution_fractions': {
                f'{modality}_{kind}': totals[f'memory_{modality}_{kind}_wrong_mass'] / totals[f'memory_{modality}_{kind}_known_mass']
                if totals[f'memory_{modality}_{kind}_known_mass'] else None
                for modality in ('rgb', 'tir') for kind in ('write', 'used')},
            'current_quality_calibration_bins': [{'bin': index, **{key: sum(b[index][key] for b in calibration)
                                                                   for key in ('count', 'prediction_sum', 'actual_iou_sum')}}
                                                 for index in range(10)],
            'motion': aggregate_motion(motion),
            'efficiency': {'instrumented_tracking_fps': len(latency) / float(latency.sum()),
                           'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
                           'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
                           'peak_cuda_allocated_mib': receipt['peak_cuda_allocated_mib'],
                           'peak_cuda_reserved_mib': receipt['peak_cuda_reserved_mib']},
            'inference_config': config}
        sequence_means[label] = np.asarray(means)
    rng = np.random.default_rng(42)
    resamples = rng.integers(0, len(names), size=(5000, len(names)))
    reference = args.labels[0]
    for label in args.labels[1:]:
        difference = (sequence_means[label] - sequence_means[reference]) * 100
        report['paired'][label] = {'reference': reference, 'mean_iou_delta_percentage_points': float(difference.mean()),
                                   'paired_sequence_95_ci': np.percentile(difference[resamples].mean(1), [2.5, 97.5]).tolist(),
                                   'improved_sequences': int((difference > 0).sum()),
                                   'worsened_sequences': int((difference < 0).sum()),
                                   'tied_sequences': int((difference == 0).sum()),
                                   'scope': '5000 paired98-sequence resamples/seed42/fixed checkpoints; not training-seed uncertainty or native PR/SR'}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'full_internal_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    save_csv(output / 'per_sequence.csv', rows)
    print(json.dumps({'completed': True, 'official_tracking_accuracy': False, 'sequences': len(names),
                      'overall': {label: v['sequence_mean_iou'] for label, v in report['variants'].items()}}))


if __name__ == '__main__':
    main()
