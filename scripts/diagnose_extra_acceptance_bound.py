"""Actual saved developer states: optimistic C-score bounds for recovered candidates.

The tracker does not save harm/risk logits, so this is a bound, not a replay.
For an extra-region action, scores <= raw_advantage + .025*keep_would_write - .01.
The harm penalty is nonnegative; query risk is in[0,1] and masked by raw>.84.
Regular keep has score0 under the action-write verification used by old4.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.collect_abc_metrics import xywh
from research.collect_candidate_metrics import ground_truth
from research.collect_core_metrics import localization_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    root = Path(args.predictions)
    config = json.loads((root / 'inference_config.json').read_text())
    completion = json.loads((root / 'inference_completion.json').read_text())
    assert config['dataset'] == 'lasher' and config['validation_split']
    assert config['root'] == '/data/wangwj/dataset/LasHeR/traingset'
    assert config['write_verification'] == 'action' and config['search_value'] == 'gross'
    assert config['threshold'] == .03 and completion['completed']
    assert completion['sequences'] == 98 and completion['frames'] == 49418
    counts = {'reintroduced_frames': 0, 'selected_correct_frames': 0,
              'even_without_extra_cost_bound_below_threshold': 0,
              'only_cost_band_could_block': 0, 'with_cost_bound_can_cross_threshold': 0}
    episodes, per_sequence, bounds = [], [], []
    for record in completion['records']:
        name = record['sequence']
        prediction = np.loadtxt(root / (name + '.txt'), ndmin=2)
        gt = ground_truth(config['root'], name, 'lasher')
        with np.load(root / (name + '_recoverability_decisions.npz')) as timeline:
            boxes = xywh(timeline['boxes_xyxy']).reshape(len(prediction) - 1, 35, 4)
            valid = timeline['valid'].reshape(len(boxes), 35)
            advantage = timeline['predicted_advantage'].reshape(len(boxes), 35, 2)
            raw = timeline['raw_score'].reshape(len(boxes), 35)
            keep = timeline['original_choice'].astype(int)
            choice = timeline['choice'].astype(int)
        rows = np.arange(len(boxes))
        assert (advantage[rows, keep, 0] == 0).all()
        qualities = []
        for candidate in range(35):
            quality, known = localization_quality(np.concatenate((prediction[:1], boxes[:, candidate])), gt, 'lasher')
            qualities.append(quality[1:])
        quality, known = np.stack(qualities, 1), known[1:]
        local = np.where(valid[:, :5], quality[:, :5], -1).max(1) >= .5
        correct_extra = valid[:, 5:] & (quality[:, 5:] >= .5)
        opportunity = known & ~local & correct_extra.any(1)
        selected_correct = opportunity & (quality[rows, choice] >= .5)
        legal = correct_extra[..., None] & np.stack((np.ones_like(raw[:, 5:], dtype=bool), raw[:, 5:] > .84), -1)
        optimistic = advantage[:, 5:] + .025 * (raw[rows, keep] > .84)[:, None, None]
        optimistic = np.where(legal, optimistic, -np.inf).max((1, 2))
        with_cost = optimistic - .01
        impossible = opportunity & (optimistic <= config['threshold'])
        cost_band = opportunity & (optimistic > config['threshold']) & (with_cost <= config['threshold'])
        possible = opportunity & (with_cost > config['threshold'])
        assert np.array_equal(impossible | cost_band | possible, opportunity)
        assert (with_cost[selected_correct] > config['threshold'] - 1e-6).all()
        values = {'reintroduced_frames': int(opportunity.sum()),
                  'selected_correct_frames': int(selected_correct.sum()),
                  'even_without_extra_cost_bound_below_threshold': int(impossible.sum()),
                  'only_cost_band_could_block': int(cost_band.sum()),
                  'with_cost_bound_can_cross_threshold': int(possible.sum())}
        for key, value in values.items():
            counts[key] += value
        per_sequence.append({'sequence': name, **values})
        bounds.extend(optimistic[opportunity].tolist())
        starts = np.flatnonzero(opportunity & ~np.r_[False, opportunity[:-1]])
        ends = np.flatnonzero(opportunity & ~np.r_[opportunity[1:], False]) + 1
        assert len(starts) == len(ends)
        for start, end in zip(starts, ends):
            episodes.append({'sequence': name, 'start_frame_zero_based': int(start + 1),
                             'end_frame_zero_based': int(end), 'frames': int(end - start),
                             'correctly_selected_any': bool(selected_correct[start:end].any()),
                             'no_cost_bound_can_cross_any': bool((optimistic[start:end] > .03).any()),
                             'with_cost_bound_can_cross_any': bool((with_cost[start:end] > .03).any())})
    assert counts['reintroduced_frames'] == 442 and counts['selected_correct_frames'] == 2
    result = {'status': 'PASS', 'scope': 'Already completed98developer videos/49418frames; actual saved states and offlineGT only.',
              'counts': counts, 'events': {'opportunity_episodes': len(episodes),
              'correctly_selected_episodes': sum(e['correctly_selected_any'] for e in episodes),
              'no_cost_bound_can_cross_episodes': sum(e['no_cost_bound_can_cross_any'] for e in episodes),
              'with_cost_bound_can_cross_episodes': sum(e['with_cost_bound_can_cross_any'] for e in episodes)},
              'optimistic_no_cost_score_quantiles': np.quantile(bounds, [0, .25, .5, .75, 1]).tolist(),
              'event_unit': 'Consecutive known-GT frames where originalregion lacksIoU>=.5 but executedextra contains one. UnknownGT excluded; not semanticidentity tracks or independent training seeds.',
              'limitation': 'Upper bounds, not actual harm/risk-adjusted scores. Crossing a bound does not prove a switch, future benefit or acceptable identity. No native scores, futureGT or newlabels used for deployment.',
              'execution': {'neural_forward_calls': 0, 'optimizer_steps': 0, 'GPU_queries': 0, 'native_test_reads': False},
              'per_sequence': per_sequence, 'episodes': episodes}
    output = Path(args.output)
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'counts', 'events', 'optimistic_no_cost_score_quantiles', 'execution')}))


if __name__ == '__main__':
    main()
