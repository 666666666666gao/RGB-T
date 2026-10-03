"""Full-benchmark ABC candidate, branch-template and forecast diagnostics.

All ground truth is consumed after inference. Template pollution and switches
are localization proxies, since these benchmarks do not label distractor IDs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from .collect_candidate_metrics import ground_truth, rates
from .collect_core_metrics import EXPECTED, localization_quality, overlap_xywh, save_csv


def xywh(boxes):
    result = np.asarray(boxes).copy()
    result[..., 2:] -= result[..., :2]
    return result


def quality_at(box, frame, gt, dataset):
    # Preserve the benchmark's two-modality max-IoU diagnostic convention.
    boxes = np.asarray(box, dtype=np.float64)[None]
    labels = gt[frame:frame + 1] if dataset == 'lasher' else {
        m: a[frame:frame + 1] for m, a in gt.items()}
    # localization_quality excludes its first row; prepend one dummy init row.
    labels = np.repeat(labels, 2, axis=0) if dataset == 'lasher' else {
        m: np.repeat(a, 2, axis=0) for m, a in labels.items()}
    quality, valid = localization_quality(np.repeat(xywh(boxes), 2, axis=0), labels, dataset)
    return float(quality[1]), bool(valid[1])


def memory_labels(box, frame, gt, dataset):
    labels = [gt[frame], gt[frame]] if dataset == 'lasher' else [gt[m][frame] for m in ('visible', 'infrared')]
    labels = np.asarray(labels)
    known = np.isfinite(labels).all(1) & (labels[:, 2:] > 0).all(1)
    quality = overlap_xywh(np.repeat(xywh(np.asarray(box)[None]), 2, 0), labels)
    return known.astype(float), (known & (quality < .2)).astype(float)


def motion_statistics(timeline, annotations):
    """Visible/TIR are measured separately; never select the better NLL label."""
    means = timeline['forecast_means'].astype(np.float64)
    std = timeline['forecast_deviations'].astype(np.float64)
    log_weights = timeline['forecast_log_weights'].astype(np.float64)
    reference = timeline['forecast_reference'].astype(np.float64)
    n, modes, horizon, coordinates = means.shape
    assert coordinates == 4 and std.shape == means.shape
    assert log_weights.shape == (n, modes) and reference.shape == (n, 4)
    assert (std > 0).all() and np.allclose(np.exp(log_weights).sum(1), 1, atol=1e-6)
    frames = np.arange(1, n + 1)[:, None] + np.arange(horizon)[None]
    inside = frames < len(annotations)
    targets = annotations[np.minimum(frames, len(annotations) - 1)]
    valid = inside & np.isfinite(targets).all(-1) & (targets[..., 2:] > 0).all(-1)
    # The model uses original centers and widths/heights clamped to >=10.
    origin = (reference[:, :2] + reference[:, 2:]) / 2
    scale = np.maximum(reference[:, 2:] - reference[:, :2], 10.)
    center = targets[..., :2] + targets[..., 2:] / 2
    normalized = np.concatenate(((center - origin[:, None]) / scale[:, None],
                                  np.log(np.maximum(targets[..., 2:], 10.) / scale[:, None])), -1)
    error = (normalized[:, None] - means) / std
    components = -.5 * (error ** 2 + np.log(2 * np.pi)).sum(-1) - np.log(std).sum(-1)
    marginal_nll = -np.logaddexp.reduce(components + log_weights[..., None], axis=1)
    joint_valid = valid.all(1)
    joint_nll = -np.logaddexp.reduce(components.sum(-1) + log_weights, axis=1) / horizon
    center_error = np.linalg.norm(means[..., :2] - normalized[:, None, :, :2], axis=-1)
    chosen_mode = log_weights.argmax(1)
    chosen_error = center_error[np.arange(n), chosen_mode]
    # Coordinate-wise PIT: calibrated mixture marginals should be uniform.
    pit = (ndtr(error) * np.exp(log_weights)[..., None, None]).sum(1)
    # Union of per-mode rectangles, each with 90% marginal center-coordinate bands.
    # This is a coverage/area diagnostic, NOT a nominal 90% joint confidence set.
    covered = (np.abs(error[..., :2]) <= 1.6448536269514722).all(-1).any(1)
    area = ((2 * 1.6448536269514722 * std[..., :2]).prod(-1)).sum(1)
    result = {'valid_joint_trajectories': int(joint_valid.sum()),
              'joint_nll_per_step_sum': float(joint_nll[joint_valid].sum()),
              'oracle_best_mode_normalized_ade_sum': float(center_error.mean(-1).min(1)[joint_valid].sum()),
              'max_probability_mode_normalized_ade_sum': float(chosen_error.mean(-1)[joint_valid].sum()),
              'steps': []}
    for step in range(horizon):
        mask = valid[:, step]
        result['steps'].append({'step': step + 1, 'valid_forecasts': int(mask.sum()),
                                'nll_sum': float(marginal_nll[mask, step].sum()),
                                'max_probability_mode_center_error_sum': float(chosen_error[mask, step].sum()),
                                'union_center_band_covered': int(covered[mask, step].sum()),
                                'sum_mode_region_area_sum': float(area[mask, step].sum()),
                                'coordinate_pit_histograms': [np.histogram(pit[mask, step, dim],
                                                                          bins=np.linspace(0, 1, 11))[0].tolist()
                                                               for dim in range(4)]})
    return result


def summarize(timeline, predictions, events, gt, dataset, restore_state=True):
    n = len(predictions) - 1
    boxes, valid_candidates = xywh(timeline['boxes_xyxy']), timeline['valid']
    k = boxes.shape[1]
    assert boxes.shape == (n, k, 4) and valid_candidates.shape == (n, k)
    assert len(events) == n and all(np.isfinite(timeline[key]).all() for key in timeline.files)
    assert valid_candidates.any(1).all()
    choices, c1_choices = timeline['choice'].astype(int), timeline['c1_choice'].astype(int)
    assert ((choices >= 0) & (choices < k)).all() and ((c1_choices >= 0) & (c1_choices < k)).all()
    assert valid_candidates[np.arange(n), choices].all() and valid_candidates[np.arange(n), c1_choices].all()
    assert np.allclose(boxes[np.arange(n), choices], predictions[1:], rtol=0, atol=.000501)
    candidate_quality = []
    for candidate in range(k):
        quality, valid_gt = localization_quality(np.concatenate((predictions[:1], boxes[:, candidate])), gt, dataset)
        candidate_quality.append(quality[1:])
    candidate_quality = np.stack(candidate_quality, 1)
    valid_gt = valid_gt[1:]
    selected = candidate_quality[np.arange(n), choices]
    c1_selected = candidate_quality[np.arange(n), c1_choices]
    original = candidate_quality[:, 0]
    recalled = valid_gt & (np.where(valid_candidates, candidate_quality, -1).max(1) >= .5)
    original_failed, original_correct = valid_gt & (original < .2), valid_gt & (original >= .5)
    chosen_failed = valid_gt & (selected < .2)
    rescuable = original_failed & recalled
    updates = timeline['template_updated']
    assert np.array_equal(updates, timeline['raw_score'][np.arange(n), choices] > .84)
    actions = valid_candidates & valid_gt[:, None]
    predicted, target = timeline['current_quality'][actions], candidate_quality[actions]
    calibration = []
    for index in range(10):
        mask = (predicted >= index / 10) & ((predicted < (index + 1) / 10) if index < 9 else (predicted <= 1))
        calibration.append({'bin': index, 'count': int(mask.sum()),
                            'prediction_sum': float(predicted[mask].sum()), 'actual_iou_sum': float(target[mask].sum())})
    c = dict(tracking_frames=n, valid_tracking_frames=int(valid_gt.sum()),
             valid_candidate_actions=int(actions.sum()), candidate_count_sum=int(valid_candidates.sum()),
             chosen_failed_frames=int(chosen_failed.sum()), recalled_frames=int(recalled.sum()),
             original_failed_frames=int(original_failed.sum()), original_correct_frames=int(original_correct.sum()),
             correct_candidate_available_when_original_failed=int(rescuable.sum()),
             current_frame_reselection_rescues=int((rescuable & (selected >= .5)).sum()),
             current_frame_reselection_harms=int((original_correct & chosen_failed).sum()),
             mis_rejected_correct_candidate_frames=int((chosen_failed & recalled).sum()),
             alternative_selections=int((choices != 0).sum()), c1_reselections=int((choices != c1_choices).sum()),
             c1_failed_frames=int((valid_gt & (c1_selected < .2)).sum()),
             rescues_against_same_search_c1=int((valid_gt & (c1_selected < .2) & (selected >= .5)).sum()),
             harms_against_same_search_c1=int((valid_gt & (c1_selected >= .5) & chosen_failed).sum()),
             template_updates=int(updates.sum()), valid_gt_template_updates=int((updates & valid_gt).sum()),
             wrong_template_updates_localization_proxy=int((updates & chosen_failed).sum()),
             known_active_template_source_frames=0, wrong_active_template_source_frames_localization_proxy=0,
             candidate_quality_squared_error_sum=float(((predicted - target) ** 2).sum()),
             all_branch_candidate_count=0, union_branch_recalled_frames=0,
             retained_correct_branch_while_main_failed=0, all_branch_updates=0,
             valid_all_branch_updates=0, wrong_all_branch_updates=0,
             switches=0, valid_switches=0, beneficial_switches=0, harmful_switches=0)
    for modality in ('rgb', 'tir'):
        for kind in ('write', 'used'):
            for label in ('known', 'wrong'):
                c[f'memory_{modality}_{kind}_{label}_mass'] = 0.
    slots = timeline['memory_write_rates'].shape[1]
    memory = {0: (np.ones((slots, 2)), np.zeros((slots, 2)))}
    committed = {0: (np.ones((slots, 2)), np.zeros((slots, 2)))}
    switch_records = []
    for index, event in enumerate(events):
        frame = index + 1
        assert event['frame'] == frame
        main = next(branch for branch in event['branches'] if branch['id'] == event['main_id'])
        assert np.array_equal(main['box'], timeline['boxes_xyxy'][index, choices[index]])
        source_frame = int(timeline['prior_template_source_frame'][index])
        assert 0 <= source_frame < frame
        source_q, source_known = quality_at(timeline['prior_template_source_box'][index], source_frame, gt, dataset)
        if valid_gt[index] and source_known:
            c['known_active_template_source_frames'] += 1
            c['wrong_active_template_source_frames_localization_proxy'] += int(source_q < .2)
        union_correct = False
        for candidates in event['candidate_sets']:
            for box, valid in zip(candidates['boxes_xyxy'], candidates['valid']):
                if valid:
                    c['all_branch_candidate_count'] += 1
                    quality, known = quality_at(box, frame, gt, dataset)
                    union_correct |= known and quality >= .5
        c['union_branch_recalled_frames'] += int(union_correct and valid_gt[index])
        retained = False
        next_memory, next_committed = {}, {}
        used_known, used_wrong = memory[int(timeline['parent_id'][index])]
        current_known, _ = memory_labels(main['box'], frame, gt, dataset)
        for modality, name in enumerate(('rgb', 'tir')):
            if current_known[modality]:
                c[f'memory_{name}_used_known_mass'] += float(used_known[:, modality].sum())
                c[f'memory_{name}_used_wrong_mass'] += float(used_wrong[:, modality].sum())
        for branch in event['branches']:
            quality, known = quality_at(branch['box'], frame, gt, dataset)
            retained |= branch['id'] != main['id'] and known and quality >= .5
            if branch['template_updated']:
                c['all_branch_updates'] += 1
                c['valid_all_branch_updates'] += int(known)
                c['wrong_all_branch_updates'] += int(known and quality < .2)
            assert 0 <= branch['committed_source_frame'] <= branch['template_source_frame'] <= frame
            identifier, parent = branch['id'], branch['parent_id']
            before_known, before_wrong = committed[parent] if identifier != parent else memory[parent]
            rate = np.asarray(branch['memory_write_rates'])
            assert rate.shape == (slots, 2) and (rate >= 0).all() and (rate <= 1).all()
            if not branch['template_updated']:
                assert not rate.any()
            known, wrong = memory_labels(branch['box'], frame, gt, dataset)
            next_memory[identifier] = ((1 - rate) * before_known + rate * known[None],
                                       (1 - rate) * before_wrong + rate * wrong[None])
            next_committed[identifier] = committed[parent]
            for modality, name in enumerate(('rgb', 'tir')):
                c[f'memory_{name}_write_known_mass'] += float(rate[:, modality].sum() * known[modality])
                c[f'memory_{name}_write_wrong_mass'] += float(rate[:, modality].sum() * wrong[modality])
        c['retained_correct_branch_while_main_failed'] += int(retained and chosen_failed[index])
        if event['switched']:
            c['switches'] += 1
            former = next(branch for branch in event['branches'] if branch['id'] == event['previous_main_id'])
            former_q, known = quality_at(former['box'], frame, gt, dataset)
            c['valid_switches'] += int(known)
            benefit, harm = known and former_q < .2 and selected[index] >= .5, known and former_q >= .5 and chosen_failed[index]
            c['beneficial_switches'] += int(benefit)
            c['harmful_switches'] += int(harm)
            switch_records.append({'frame': frame, 'from': former['id'], 'to': main['id'],
                                   'gt_valid': known, 'former_iou': former_q if known else None,
                                   'selected_iou': float(selected[index]) if known else None,
                                   'beneficial': bool(benefit), 'harmful': bool(harm)})
            if not restore_state:
                next_memory[main['id']] = next_memory[former['id']]
                next_committed[main['id']] = next_committed[former['id']]
        for branch in event['branches']:
            if branch['memory_committed']:
                next_committed[branch['id']] = next_memory[branch['id']]
        memory, committed = next_memory, next_committed
    motions = {'lasher': motion_statistics(timeline, gt)} if dataset == 'lasher' else {
        modality: motion_statistics(timeline, labels) for modality, labels in gt.items()}
    return c, calibration, switch_records, motions


def aggregate_motion(records):
    result = {key: sum(row[key] for row in records) for key in records[0] if key != 'steps'}
    count = result['valid_joint_trajectories']
    for key in ('joint_nll_per_step_sum', 'oracle_best_mode_normalized_ade_sum', 'max_probability_mode_normalized_ade_sum'):
        result[key.removesuffix('_sum')] = result[key] / count if count else None
    result['steps'] = []
    for step in range(len(records[0]['steps'])):
        rows = [r['steps'][step] for r in records]
        aggregate = {key: sum(r[key] for r in rows) for key in rows[0] if key not in ('step', 'coordinate_pit_histograms')}
        aggregate['step'] = step + 1
        aggregate['coordinate_pit_histograms'] = np.sum([r['coordinate_pit_histograms'] for r in rows], 0).tolist()
        count = aggregate['valid_forecasts']
        for key in ('nll_sum', 'max_probability_mode_center_error_sum', 'sum_mode_region_area_sum'):
            aggregate[key.removesuffix('_sum')] = aggregate[key] / count if count else None
        aggregate['union_center_band_coverage'] = aggregate['union_center_band_covered'] / count if count else None
        result['steps'].append(aggregate)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(EXPECTED), required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--runs', nargs='+', required=True, help='Run folders containing predictions/')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert len(args.variants) == len(args.runs) and len(set(args.variants)) == len(args.variants)
    expected_sequences, expected_frames = EXPECTED[args.dataset]
    names = sorted(p.name for p in Path(args.data_root).iterdir() if p.is_dir())
    assert len(names) == expected_sequences
    report = {'dataset': args.dataset, 'actual_data_root': args.data_root, 'variants': {},
              'protocol': {'localization': 'valid RGB/TIR max IoU; failure<.2, correct>=.5; initialization excluded',
                           'candidate0': 'original GOLA Hann winner within the selected branch search',
                           'c1': 'frozen C1 choice within exactly the same branch search',
                           'pollution': 'actual prior template source frame/box; localization proxy, unknown excluded',
                           'learned_memory': 'recorded slot update coefficients reconstruct known/wrong localization contribution mass; protected initialization trusted; not semantic identity purity',
                           'forecast': 'next H GT from current frame; normalized centers/log sizes; modalities separate',
                           'coverage': 'union per-mode center rectangles with 90% coordinate bands; not nominal joint90%',
                           'area': 'sum per-mode normalized center rectangle areas, overlaps counted',
                           'future_head': 'fixed-C1 counterfactual target evaluated on held-out cache, not future ABC rollout'}}
    rows = []
    for variant, run in zip(args.variants, args.runs):
        root = Path(run) / 'predictions'
        receipt = json.loads((root / 'inference_completion.json').read_text())
        config = json.loads((root / 'inference_config.json').read_text())
        assert receipt['completed'] and not receipt['smoke_only'] and receipt['branch_timeline_recorded']
        assert receipt['sequences'] == expected_sequences and receipt['frames'] == expected_frames
        assert config['dataset'] == args.dataset and config['root'] == args.data_root and config['variant'] == variant
        assert {p.stem for p in root.glob('*.txt')} == set(names)
        counts, bins, switches, motion = [], [], [], []
        for name in names:
            gt = ground_truth(args.data_root, name, args.dataset)
            frames = len(gt if args.dataset == 'lasher' else gt['visible'])
            prediction = np.loadtxt(root / (name + '.txt'), ndmin=2)
            assert prediction.shape == (frames, 4) and np.isfinite(prediction).all()
            events = json.loads((root / (name + '_branch_events.json')).read_text())
            with np.load(root / (name + '_abc_decisions.npz')) as timeline:
                c, calibration, events_summary, forecasts = summarize(timeline, prediction, events, gt, args.dataset,
                                                                      config['variant'] == 'abc')
            record = next(row for row in receipt['records'] if row['sequence'] == name)
            assert c['all_branch_updates'] == record['branch_stats']['template_updates']
            assert c['switches'] == record['branch_stats']['switches']
            assert len(events) == frames - 1
            counts.append(c)
            bins.append(calibration)
            switches.extend({'sequence': name, **event} for event in events_summary)
            motion.append(forecasts)
            rows.append({'variant': variant, 'sequence': name, **c, **rates(c)})
        totals = {key: sum(c[key] for c in counts) for key in counts[0]}
        assert totals['tracking_frames'] == expected_frames - expected_sequences
        calibration = [{'bin': index, **{key: sum(sequence[index][key] for sequence in bins)
                                        for key in ('count', 'prediction_sum', 'actual_iou_sum')}} for index in range(10)]
        report['variants'][variant] = {'counters': totals, 'rates': rates(totals),
                                       'learned_memory_rates': {
                                           f'{modality}_{kind}_wrong_contribution_fraction':
                                           totals[f'memory_{modality}_{kind}_wrong_mass'] / totals[f'memory_{modality}_{kind}_known_mass']
                                           if totals[f'memory_{modality}_{kind}_known_mass'] else None
                                           for modality in ('rgb', 'tir') for kind in ('write', 'used')},
                                       'candidate_quality_calibration_bins': calibration,
                                       'branch_switch_records': switches,
                                       'motion': {modality: aggregate_motion([r[modality] for r in motion])
                                                  for modality in motion[0]},
                                       'inference_config': config, 'inference_efficiency': {
                                           key: receipt[key] for key in ('fps_including_decode_crop_update',
                                                                        'latency_p50_ms', 'latency_p95_ms',
                                                                        'peak_cuda_allocated_mib', 'peak_cuda_reserved_mib')}}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'abc_diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    save_csv(output / 'abc_per_sequence.csv', rows)
    print(json.dumps({'completed': True, 'dataset': args.dataset, 'variants': args.variants,
                      'sequences': expected_sequences, 'frames': expected_frames}))


if __name__ == '__main__':
    main()
