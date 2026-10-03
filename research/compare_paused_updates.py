"""Original98 full-video write-pause control; no native benchmark claim."""
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
    parser.add_argument('--baseline', required=True, help='Completed original98 baseline predictions folder')
    parser.add_argument('--c1', required=True, help='Completed original98 C1 predictions folder')
    parser.add_argument('--paused', required=True, help='Completed pause3 C1 predictions folder')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    split = json.loads(Path(args.split).read_text())
    names = sorted(split['validation'])
    assert len(names) == 98 and not set(names) & set(split['train'])
    report = {'completed': True, 'official_tracking_accuracy': False, 'args': vars(args),
              'protocol': 'all98 TRAIN-held-out videos; only first GT initializes; quality/init/unknown/failure protocol matches previous internal controls; pause is a causal heuristic, not verified identity or a no-harm guarantee',
              'variants': {}, 'paired': {}}
    rows, means, qualities = [], {}, {}
    for label, folder in (('baseline', args.baseline), ('c1', args.c1), ('c1_pause3', args.paused)):
        root = Path(folder)
        config = json.loads((root / 'inference_config.json').read_text())
        receipt = json.loads((root / 'inference_completion.json').read_text())
        assert receipt['completed'] and receipt['smoke_only'] and receipt['candidate_timeline_recorded']
        assert receipt['sequences'] == 98 and receipt['frames'] == 49418
        assert config['root'] == args.root and config['validation_split'] == args.split and config['dataset'] == 'lasher'
        assert config['limit_sequences'] == config['max_frames'] == 0
        assert config['variant'] == ('baseline' if label == 'baseline' else 'c1')
        if label == 'c1_pause3':
            assert config['pause_after_correction'] == 3 and config['head_epoch'] == 3
        else:
            assert 'pause_after_correction' not in config  # Original controls were locked before this experiment.
        assert {p.stem for p in root.glob('*.txt')} == set(names)
        counters, bins, failures, latency_arrays, sequence_means = [], [], [], [], []
        frames, valid_frames, total_iou = 0, 0, 0.
        qualities[label] = []
        withheld_correct, withheld_failed, withheld_known, blocked_frames = 0, 0, 0, 0
        for name in names:
            gt = np.loadtxt(Path(args.root) / name / 'init.txt', delimiter=',', ndmin=2)
            prediction = np.loadtxt(root / (name + '.txt'), ndmin=2)
            assert prediction.shape == gt.shape and np.isfinite(prediction).all()
            quality, valid = localization_quality(prediction, gt, 'lasher')
            assert valid.any()
            mean = float(quality[valid].mean())
            sequence_means.append(mean)
            frames += len(prediction)
            valid_frames += int(valid.sum())
            total_iou += float(quality[valid].sum())
            qualities[label].append((quality, valid))
            events = failure_events(quality, valid)
            failures.extend({'sequence': name, **e} for e in events)
            with np.load(root / (name + '_candidates.npz')) as timeline:
                permission = None
                if label == 'c1_pause3':
                    permission = timeline['template_write_permitted']
                    expected, until = [], 0
                    for frame, choice in enumerate(timeline['choice'], 1):
                        if choice != 0:
                            until = frame + 2
                        expected.append(frame > until)
                    assert np.array_equal(permission, expected)
                    raw = timeline['evidence'][np.arange(len(permission)), timeline['choice'], 0]
                    withheld = ~permission & (raw > .84) & valid[1:]
                    withheld_known += int(withheld.sum())
                    withheld_correct += int((withheld & (quality[1:] >= .5)).sum())
                    withheld_failed += int((withheld & (quality[1:] < .2)).sum())
                    blocked_frames += int((~permission).sum())
                c, calibration = summarize(timeline, prediction, gt, 'lasher', permission)
            record = next(r for r in receipt['records'] if r['sequence'] == name)
            assert c['template_updates'] == record['template_updates']
            assert c['alternative_selections'] == record['alternative_selections']
            counters.append(c)
            bins.append(calibration)
            latency = np.load(root / (name + '_latency.npy'))
            assert latency.shape == (len(prediction) - 1,) and (latency > 0).all() and np.isfinite(latency).all()
            latency_arrays.append(latency)
            rows.append({'variant': label, 'sequence': name, 'frames': len(prediction),
                         'mean_valid_iou': mean, 'failure_events': len(events),
                         'recovered_events': sum(e['recovered'] for e in events), **c, **rates(c)})
        assert frames == 49418
        totals = {key: sum(c[key] for c in counters) for key in counters[0]}
        assert valid_frames == totals['valid_tracking_frames']
        means[label] = np.asarray(sequence_means)
        delays = [e['delay_frames'] for e in failures if e['recovered']]
        latency = np.concatenate(latency_arrays)
        report['variants'][label] = {
            'frames': frames, 'valid_tracking_frames': valid_frames,
            'sequence_mean_iou': float(means[label].mean()), 'frame_mean_iou': total_iou / valid_frames,
            'counters': totals, 'rates': rates(totals), 'failure_events': failures,
            'recovery_rate': len(delays) / len(failures) if failures else None,
            'median_recovery_delay': float(np.median(delays)) if delays else None,
            'pause': {'blocked_frames': blocked_frames, 'withheld_high_confidence_known_writes': withheld_known,
                      'withheld_correct_writes': withheld_correct, 'withheld_failed_writes': withheld_failed},
            'calibration_bins': [{'bin': b, **{key: sum(c[b][key] for c in bins)
                                              for key in ('count', 'prediction_sum', 'actual_iou_sum')}} for b in range(10)],
            'efficiency': {'tracking_fps': len(latency) / float(latency.sum()),
                           'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
                           'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
                           'peak_cuda_allocated_mib': receipt['peak_cuda_allocated_mib']}, 'inference_config': config}
    resamples = np.random.default_rng(42).integers(0, 98, size=(5000, 98))
    for reference in ('baseline', 'c1'):
        delta = (means['c1_pause3'] - means[reference]) * 100
        harmed, rescued, known_frames, reference_correct, reference_failed = 0, 0, 0, 0, 0
        for (q, valid), (ref, ref_valid) in zip(qualities['c1_pause3'], qualities[reference]):
            assert np.array_equal(valid, ref_valid)
            harmed += int((valid & (ref >= .5) & (q < .2)).sum())
            rescued += int((valid & (ref < .2) & (q >= .5)).sum())
            known_frames += int(valid.sum())
            reference_correct += int((valid & (ref >= .5)).sum())
            reference_failed += int((valid & (ref < .2)).sum())
        report['paired'][reference] = {
            'mean_iou_delta_percentage_points': float(delta.mean()),
            'paired_sequence_95_ci': np.percentile(delta[resamples].mean(1), [2.5, 97.5]).tolist(),
            'improved_sequences': int((delta > 0).sum()), 'worsened_sequences': int((delta < 0).sum()),
            'tied_sequences': int((delta == 0).sum()), 'paired_known_frames': known_frames,
            'reference_correct_frames_harmed': harmed, 'reference_failed_frames_rescued': rescued,
            'reference_correct_frames': reference_correct, 'reference_failed_frames': reference_failed,
            'reference_correct_harm_rate': harmed / reference_correct,
            'reference_failed_rescue_rate': rescued / reference_failed,
            'scope': 'full causal trajectory effects, not same-state instantaneous intervention; fixed-weight5000 paired sequence resamples, not independent training seeds'}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'paused_updates_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    save_csv(output / 'per_sequence.csv', rows)
    (output / 'paused_updates_completed.txt').write_text('All three original98 full runs verified; localization diagnostics only.\n')
    print(json.dumps({'completed': True, 'sequence_mean_iou': {k: v['sequence_mean_iou'] for k, v in report['variants'].items()}}))


if __name__ == '__main__':
    main()
