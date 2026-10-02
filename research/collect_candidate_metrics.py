"""All-frame candidate/update diagnostics against actual benchmark GT.

Recorded trajectories must match the supplied reference. A same-run reference
checks source consistency, not independent inference equivalence.
Candidate quality and wrong template writes are localization measurements;
these datasets do not label every distractor's semantic identity.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .collect_core_metrics import EXPECTED, failure_events, localization_quality, save_csv


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=tuple(EXPECTED), required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--variants', nargs='+', required=True)
    p.add_argument('--runs', nargs='+', required=True, help='Instrumented full inference folders')
    p.add_argument('--reference-runs', nargs='+', required=True, help='Full inference reference folders; same-run references check source consistency only')
    p.add_argument('--output', required=True)
    return p.parse_args()


def ground_truth(root, name, dataset):
    sequence = Path(root) / name
    if dataset == 'lasher':
        return np.loadtxt(sequence / 'init.txt', delimiter=',', dtype=np.float64, ndmin=2)
    return {modality: np.loadtxt(sequence / (modality + '.txt'), delimiter=',', dtype=np.float64, ndmin=2)
            for modality in ('visible', 'infrared')}


def summarize(timeline, predictions, gt, dataset):
    boxes = timeline['boxes_xyxy'].copy()
    boxes[:, :, 2:] -= boxes[:, :, :2]
    n, k, _ = boxes.shape
    assert timeline['valid'].shape == (n, k) and timeline['evidence'].shape == (n, k, 10)
    assert timeline['predicted_quality'].shape == (n, k)
    assert timeline['choice'].shape == timeline['template_updated'].shape == (n,)
    assert timeline['valid'].any(1).all() and timeline['valid'][:, 0].all()
    assert all(np.isfinite(timeline[key]).all() for key in timeline.files)
    choice = timeline['choice']
    assert ((choice >= 0) & (choice < k)).all()
    assert timeline['valid'][np.arange(n), choice].all()
    selected = boxes[np.arange(n), choice]
    assert np.allclose(selected, predictions[1:], rtol=0, atol=.000501)
    quality, valid_gt = [], None
    for candidate in range(k):
        q, valid_gt = localization_quality(np.concatenate((predictions[:1], boxes[:, candidate])), gt, dataset)
        quality.append(q[1:])
    quality = np.stack(quality, axis=1)
    valid_gt = valid_gt[1:]
    valid = timeline['valid'] & valid_gt[:, None]
    oracle = np.where(timeline['valid'], quality, -1.).max(1)
    chosen_quality = quality[np.arange(n), choice]
    original_failed = valid_gt & (quality[:, 0] < .2)
    original_correct = valid_gt & (quality[:, 0] >= .5)
    chosen_failed = valid_gt & (chosen_quality < .2)
    recalled = valid_gt & (oracle >= .5)
    rescuable = original_failed & recalled
    rescued = rescuable & (chosen_quality >= .5)
    updates = timeline['template_updated']
    assert np.array_equal(updates, timeline['evidence'][np.arange(n), choice, 0] > .84)
    wrong_updates = updates & chosen_failed
    # Before-frame active template source: first-frame initialization is valid;
    # an unannotated update makes its source quality unknown, not a negative.
    template_valid, template_wrong = True, False
    contaminated_use, known_use = 0, 0
    for frame in range(n):
        if valid_gt[frame] and template_valid:
            known_use += 1
            contaminated_use += int(template_wrong)
        if updates[frame]:
            template_valid = bool(valid_gt[frame])
            template_wrong = bool(chosen_failed[frame])
    prediction = timeline['predicted_quality'][valid]
    target = quality[valid]
    calibration = []
    for bin_index in range(10):
        lower, upper = bin_index / 10, (bin_index + 1) / 10
        inside = (prediction >= lower) & ((prediction < upper) if bin_index < 9 else (prediction <= upper))
        count = int(inside.sum())
        calibration.append({'bin': bin_index, 'count': count,
                            'prediction_sum': float(prediction[inside].sum()),
                            'actual_iou_sum': float(target[inside].sum())})
    counters = {'tracking_frames': n, 'valid_tracking_frames': int(valid_gt.sum()),
                'valid_candidate_actions': int(valid.sum()), 'candidate_count_sum': int(timeline['valid'].sum()),
                'chosen_failed_frames': int(chosen_failed.sum()),
                'recalled_frames': int(recalled.sum()), 'original_failed_frames': int(original_failed.sum()),
                'original_correct_frames': int(original_correct.sum()),
                'correct_candidate_available_when_original_failed': int(rescuable.sum()),
                'current_frame_reselection_rescues': int(rescued.sum()),
                'current_frame_reselection_harms': int((original_correct & chosen_failed).sum()),
                'mis_rejected_correct_candidate_frames': int((chosen_failed & recalled).sum()),
                'alternative_selections': int((choice != 0).sum()),
                'template_updates': int(updates.sum()), 'valid_gt_template_updates': int((updates & valid_gt).sum()),
                'wrong_template_updates_localization_proxy': int(wrong_updates.sum()),
                'known_active_template_source_frames': known_use,
                'wrong_active_template_source_frames_localization_proxy': contaminated_use,
                'candidate_quality_squared_error_sum': float(((prediction - target) ** 2).sum())}
    return counters, calibration


def rates(c):
    ratios = {'candidate_recall_at_k': ('recalled_frames', 'valid_tracking_frames'),
              'candidate_recall_given_original_failure': ('correct_candidate_available_when_original_failed', 'original_failed_frames'),
              'current_frame_reselection_rescue_rate': ('current_frame_reselection_rescues', 'correct_candidate_available_when_original_failed'),
              'current_frame_reselection_harm_rate': ('current_frame_reselection_harms', 'original_correct_frames'),
              'mis_rejected_correct_candidate_rate_given_chosen_failure': ('mis_rejected_correct_candidate_frames', 'chosen_failed_frames'),
              'wrong_template_update_rate_localization_proxy': ('wrong_template_updates_localization_proxy', 'valid_gt_template_updates'),
              'wrong_active_template_source_rate_localization_proxy': ('wrong_active_template_source_frames_localization_proxy', 'known_active_template_source_frames'),
              'candidate_quality_mse': ('candidate_quality_squared_error_sum', 'valid_candidate_actions')}
    return {name: c[top] / c[bottom] if c[bottom] else None for name, (top, bottom) in ratios.items()}


def main():
    args = arguments()
    assert len(args.variants) == len(args.runs) == len(args.reference_runs)
    assert len(set(args.variants)) == len(args.variants)
    expected_sequences, expected_frames = EXPECTED[args.dataset]
    report = {'dataset': args.dataset, 'actual_data_root': args.data_root,
              'scope': 'all-frame candidate and template-write diagnostics; reference/source trajectory consistency verified',
              'protocol': {'recall': 'any valid spatial candidate IoU>=.5', 'failure': 'IoU<.2',
                           'reselection': 'same current search/identity state, candidate0 is original Hann winner',
                           'initialization_excluded': True,
                           'template_pollution': 'localization proxy; no distractor semantic identity labels',
                           'calibration': 'predicted candidate quality vs actual IoU; original raw confidence is a proxy',
                           'rates': 'frame/action weighted; all numerators and denominators saved',
                           'speed': 'instrumented baseline includes additional candidate extraction; use original full report for comparison'},
              'variants': {}}
    sequence_rows = []
    names = sorted(p.name for p in Path(args.data_root).iterdir() if p.is_dir())
    assert len(names) == expected_sequences
    for variant, folder, reference in zip(args.variants, args.runs, args.reference_runs):
        root, original = Path(folder), Path(reference)
        receipt = json.loads((root / 'inference_completion.json').read_text())
        original_receipt = json.loads((original / 'inference_completion.json').read_text())
        for r in (receipt, original_receipt):
            assert r['completed'] and not r['smoke_only']
            assert (r['sequences'], r['frames']) == (expected_sequences, expected_frames)
        assert receipt['candidate_timeline_recorded']
        assert {p.stem for p in root.glob('*.txt')} == set(names)
        config = json.loads((root / 'inference_config.json').read_text())
        original_config = json.loads((original / 'inference_config.json').read_text())
        for key in ('dataset', 'root', 'variant', 'pretrained', 'head_epoch', 'seed', 'amp_dtype'):
            assert config[key] == original_config[key], key
        totals, bins = None, [dict(bin=i, count=0, prediction_sum=0., actual_iou_sum=0.) for i in range(10)]
        frame_count = 0
        for name in names:
            predictions = np.loadtxt(root / (name + '.txt'), ndmin=2)
            original_predictions = np.loadtxt(original / (name + '.txt'), ndmin=2)
            assert np.array_equal(predictions, original_predictions), (variant, name, 'trajectory mismatch')
            gt = ground_truth(args.data_root, name, args.dataset)
            length = len(gt if args.dataset == 'lasher' else gt['visible'])
            assert predictions.shape == (length, 4) and np.isfinite(predictions).all()
            timeline = np.load(root / (name + '_candidates.npz'))
            assert len(timeline['choice']) == length - 1
            counters, calibration = summarize(timeline, predictions, gt, args.dataset)
            quality, valid = localization_quality(predictions, gt, args.dataset)
            counters['failure_events'] = len(failure_events(quality, valid))
            if totals is None:
                totals = {key: 0 for key in counters}
            for key, value in counters.items():
                totals[key] += value
            for dst, src in zip(bins, calibration):
                for key in ('count', 'prediction_sum', 'actual_iou_sum'):
                    dst[key] += src[key]
            sequence_rows.append({'variant': variant, 'sequence': name, **counters, **rates(counters)})
            frame_count += length
        assert frame_count == expected_frames
        for row in bins:
            row['mean_predicted_quality'] = row['prediction_sum'] / row['count'] if row['count'] else None
            row['mean_actual_iou'] = row['actual_iou_sum'] / row['count'] if row['count'] else None
        calibration_error = sum(abs(row['prediction_sum'] - row['actual_iou_sum']) for row in bins) / totals['valid_candidate_actions']
        report['variants'][variant] = {'counters': totals, 'rates': rates(totals),
                                       'quality_calibration_bins': bins, 'quality_calibration_ece': calibration_error,
                                       'inference_peak_cuda_allocated_mib': receipt['peak_cuda_allocated_mib'],
                                       'inference_peak_cuda_reserved_mib': receipt['peak_cuda_reserved_mib'],
                                       'trajectory_reference': str(original),
                                       'trajectory_reference_is_separate_run': root.resolve() != original.resolve(),
                                       'exact_original_trajectory_match_all_sequences': root.resolve() != original.resolve(),
                                       'exact_reference_trajectory_match_all_sequences': True,
                                       'instrumented_wall_seconds': receipt['wall_seconds_including_initialization_and_diagnostic_serialization']}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'candidate_mechanisms.json').write_text(json.dumps(report, indent=2))
    save_csv(out / 'candidate_per_sequence.csv', sequence_rows)
    print(json.dumps({'completed': True, 'dataset': args.dataset, 'report': str(out / 'candidate_mechanisms.json')}))


if __name__ == '__main__':
    main()
