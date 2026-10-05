"""Same saved states: quality-head extra acceptance, preserving observed C's local choice.

Offline diagnostic only. It does not advance a tracker or supply geometry/write labels.
The replacement choice uses predicted quality/validity only; GT scores it afterwards.
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
    done = json.loads((root / 'inference_completion.json').read_text())
    assert config['dataset'] == 'lasher' and config['validation_split'] and config['search_value'] == 'gross'
    assert config['root'] == '/data/wangwj/dataset/LasHeR/traingset'
    assert done['completed'] and done['sequences'] == 98 and done['frames'] == 49418
    counts = dict(known_tracking_frames=0, changed_frames=0, reintroduced_frames=0,
                  actual_C_correct_on_reintroduced=0, quality_correct_on_reintroduced=0,
                  same_state_rescues=0, same_state_harms=0, iou_increased_frames=0,
                  iou_decreased_frames=0, known_iou_sum_delta=0.)
    per_sequence, events = [], []
    for record in done['records']:
        name = record['sequence']
        prediction = np.loadtxt(root / (name + '.txt'), ndmin=2)
        with np.load(root / (name + '_recoverability_decisions.npz')) as timeline:
            valid = timeline['valid'].reshape(len(prediction) - 1, 35)
            boxes = xywh(timeline['boxes_xyxy']).reshape(len(valid), 35, 4)
            actual = timeline['choice'].astype(int)
            predicted_quality = timeline['current_quality'].reshape(len(valid), 35)
        rows = np.arange(len(valid))
        score = predicted_quality - .01 * (np.arange(35) >= 5)[None]
        available = valid.copy()
        available[:, :5] = False
        available[rows, actual] = True
        index = np.where(available, score, -np.inf).argmax(1)
        chosen = np.where(score[rows, index] > score[rows, actual] + .03, index, actual)
        assert valid[rows, chosen].all()
        # All choices are fixed before reading GT. Original local choices are retained.
        gt = ground_truth(config['root'], name, 'lasher')
        qualities = []
        for candidate in range(35):
            quality, known = localization_quality(np.concatenate((prediction[:1], boxes[:, candidate])), gt, 'lasher')
            qualities.append(quality[1:])
        quality, known = np.stack(qualities, 1), known[1:]
        before, after = quality[rows, actual], quality[rows, chosen]
        local = np.where(valid[:, :5], quality[:, :5], -1).max(1) >= .5
        extra = (valid[:, 5:] & (quality[:, 5:] >= .5)).any(1)
        opportunity = known & ~local & extra
        values = {'known_tracking_frames': int(known.sum()), 'changed_frames': int((known & (chosen != actual)).sum()),
                  'reintroduced_frames': int(opportunity.sum()),
                  'actual_C_correct_on_reintroduced': int((opportunity & (before >= .5)).sum()),
                  'quality_correct_on_reintroduced': int((opportunity & (after >= .5)).sum()),
                  'same_state_rescues': int((known & (before < .2) & (after >= .5)).sum()),
                  'same_state_harms': int((known & (before >= .5) & (after < .2)).sum()),
                  'iou_increased_frames': int((known & (after > before + 1e-8)).sum()),
                  'iou_decreased_frames': int((known & (after < before - 1e-8)).sum()),
                  'known_iou_sum_delta': float((after - before)[known].sum())}
        for key, value in values.items():
            counts[key] += value
        per_sequence.append({'sequence': name, **values})
        starts = np.flatnonzero(opportunity & ~np.r_[False, opportunity[:-1]])
        ends = np.flatnonzero(opportunity & ~np.r_[opportunity[1:], False]) + 1
        for start, end in zip(starts, ends):
            events.append({'sequence': name, 'start_frame_zero_based': int(start + 1),
                           'end_frame_zero_based': int(end), 'frames': int(end - start),
                           'actual_C_correct_any': bool((before[start:end] >= .5).any()),
                           'quality_correct_any': bool((after[start:end] >= .5).any())})
    assert counts['reintroduced_frames'] == 442 and counts['actual_C_correct_on_reintroduced'] == 2
    result = {'status': 'PASS', 'scope': 'Same frozen full98 saved states/49418frames, no tracker rollout or native accuracy.',
              'counts': counts, 'opportunity_events': {'total': len(events),
              'actual_C_correct': sum(e['actual_C_correct_any'] for e in events),
              'quality_correct': sum(e['quality_correct_any'] for e in events)},
              'rule': 'Compare predicted quality-.01(extra) of observed C choice to executed extra candidates, change only >.03. No new local reordering/Hann replacement.',
              'event_unit': 'Consecutive known-GT reintroduction opportunity frames; not semanticidentity tracks or independent seeds.',
              'limitation': 'Instantaneous same-state localization only. Future identity, geometry, appearance/write consequences and continuous recovery are untested; do not promote as deployed performance.',
              'execution': {'neural_forward_calls': 0, 'optimizer_steps': 0, 'GPU_queries': 0, 'native_test_reads': False},
              'per_sequence': per_sequence, 'events': events}
    output = Path(args.output)
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'counts', 'opportunity_events', 'execution')}))


if __name__ == '__main__':
    main()
