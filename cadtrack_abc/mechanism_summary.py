"""Ground-truth-assisted diagnostics; these do not affect online decisions."""
import json
from pathlib import Path

import numpy as np


def box_iou_xywh(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    start = np.maximum(prediction[..., :2], truth[..., :2])
    end = np.minimum(prediction[..., :2] + prediction[..., 2:], truth[..., :2] + truth[..., 2:])
    intersection = np.maximum(end - start, 0).prod(-1)
    area = np.maximum(prediction[..., 2:], 0).prod(-1)
    target_area = np.maximum(truth[..., 2:], 0).prod(-1)
    return intersection / np.maximum(area + target_area - intersection, 1e-12)


def recovery_events(overlap, valid):
    """Start at IoU<.2, recover at IoU>=.5; invalid GT neither starts nor ends."""
    events, start = [], None
    for frame, (value, observed) in enumerate(zip(overlap, valid), 1):
        if not observed:
            continue
        if start is None and value < .2:
            start = frame
        elif start is not None and value >= .5:
            events.append({'start_frame': start, 'recovery_frame': frame,
                           'delay_frames': frame - start, 'right_censored': False})
            start = None
    if start is not None:
        events.append({'start_frame': start, 'recovery_frame': None,
                       'delay_frames': None, 'observed_until_frame': len(overlap), 'right_censored': True})
    return events


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_lasher(root, predictions, names):
    root, predictions = Path(root), Path(predictions)
    count_names = ['tracked_frames', 'valid_gt_frames', 'extra_visual_forwards', 'candidate_recalled_frames',
                   'local_candidate_recalled_frames', 'newly_recalled_frames', 'newly_recalled_selected_frames',
                   'alternative_selections', 'same_state_immediate_rescues', 'same_state_immediate_harms',
                   'failure_frames', 'geometry_commit_frames', 'incorrect_geometry_commit_frames',
                   'template_request_frames', 'template_commit_frames', 'incorrect_template_commit_frames',
                   'target_memory_commit_frames', 'incorrect_target_memory_commit_frames']
    counts = {name: 0 for name in count_names}
    modal_templates, modal_memory = [0, 0], [0, 0]
    events, per_sequence, overlap_sum = [], [], 0.
    for name in names:
        labels = np.loadtxt(root / name / 'init.txt', delimiter=',', ndmin=2)[1:]
        boxes = np.loadtxt(predictions / (name + '.txt'), ndmin=2)[1:]
        records = json.loads((predictions / (name + '_mechanisms.json')).read_text())
        assert len(records) == len(labels) == len(boxes)
        valid = (labels[:, 2:] > 0).all(-1)
        assert valid.any(), name
        overlap = box_iou_xywh(boxes, labels)
        keep = box_iou_xywh(np.round(np.asarray([r['keep_box_xywh'] for r in records]), 3), labels)
        successful = overlap >= .5
        failed = overlap < .2
        counts['tracked_frames'] += len(records)
        counts['valid_gt_frames'] += int(valid.sum())
        counts['failure_frames'] += int((valid & failed).sum())
        counts['same_state_immediate_rescues'] += int((valid & (keep < .2) & successful).sum())
        counts['same_state_immediate_harms'] += int((valid & (keep >= .5) & failed).sum())
        overlap_sum += float(overlap[valid].sum())
        for frame, record in enumerate(records):
            counts['extra_visual_forwards'] += record['extra_forwards']
            counts['alternative_selections'] += int(record['choice'] != 0)
            counts['template_request_frames'] += int(record['template_write_requested'])
            template = np.asarray(record['template_commit_rgb_tir'], dtype=bool)
            memory = np.asarray(record['target_memory_commit_rgb_tir'], dtype=bool)
            assert template.shape == memory.shape == (2,)
            modal_templates = [a + int(b) for a, b in zip(modal_templates, template)]
            modal_memory = [a + int(b) for a, b in zip(modal_memory, memory)]
            if not valid[frame]:
                continue
            candidate = np.asarray(record['candidate_boxes_xyxy'])
            candidate = np.concatenate((candidate[:, :2], candidate[:, 2:] - candidate[:, :2]), -1)
            quality = box_iou_xywh(candidate, labels[frame])
            local_recalled = quality[:5].max() >= .5
            recalled = quality.max() >= .5
            newly_recalled = not local_recalled and recalled and record['extra_forwards'] > 0
            counts['candidate_recalled_frames'] += int(recalled)
            counts['local_candidate_recalled_frames'] += int(local_recalled)
            counts['newly_recalled_frames'] += int(newly_recalled)
            counts['newly_recalled_selected_frames'] += int(newly_recalled and successful[frame])
            counts['geometry_commit_frames'] += int(record['geometry_commit'])
            counts['incorrect_geometry_commit_frames'] += int(record['geometry_commit'] and failed[frame])
            counts['template_commit_frames'] += int(template.any())
            counts['incorrect_template_commit_frames'] += int(template.any() and failed[frame])
            counts['target_memory_commit_frames'] += int(memory.any())
            counts['incorrect_target_memory_commit_frames'] += int(memory.any() and failed[frame])
        sequence_events = recovery_events(overlap, valid)
        events.extend(dict(sequence=name, **event) for event in sequence_events)
        per_sequence.append({'sequence': name, 'valid_gt_frames': int(valid.sum()),
                             'mean_iou': float(overlap[valid].mean()), 'failure_events': len(sequence_events)})
    recovered = [event['delay_frames'] for event in events if not event['right_censored']]
    return {'counts': counts, 'template_commits_rgb_tir_all_frames': modal_templates,
            'target_memory_commits_rgb_tir_all_frames': modal_memory,
            'candidate_recall': ratio(counts['candidate_recalled_frames'], counts['valid_gt_frames']),
            'local_candidate_recall': ratio(counts['local_candidate_recalled_frames'], counts['valid_gt_frames']),
            'newly_recalled_acceptance': ratio(counts['newly_recalled_selected_frames'], counts['newly_recalled_frames']),
            'incorrect_template_commit_fraction': ratio(counts['incorrect_template_commit_frames'], counts['template_commit_frames']),
            'incorrect_geometry_commit_fraction': ratio(counts['incorrect_geometry_commit_frames'], counts['geometry_commit_frames']),
            'extra_forward_fraction': ratio(counts['extra_visual_forwards'], counts['tracked_frames']),
            'sequence_equal_mean_iou': float(np.mean([r['mean_iou'] for r in per_sequence])),
            'frame_equal_mean_iou': overlap_sum / counts['valid_gt_frames'],
            'failure_events': len(events), 'recovered_events': len(recovered),
            'right_censored_events': len(events) - len(recovered),
            'recovery_delay_median_frames': float(np.median(recovered)) if recovered else None,
            'events': events, 'per_sequence': per_sequence,
            'definitions': {'correct_candidate': 'Emitted candidate box IoU>=0.5; un-clipped candidate boxes as used by training labels.',
                            'selected_and_keep': 'Actual clipped online output and same-state clipped local candidate0, both rounded to3 decimals, IoU against GT.',
                            'failure': 'IoU<0.2; an event ends at IoU>=0.5. Delay counts actual frame indices.',
                            'ground_truth': 'Initialization excluded; diagnostic decisions use finite GT with positive width/height. GT with x/y=0 remains a box; this diagnostic mask differs from native official PR/NPR/SR.',
                            'commits': 'Actual per-sensor template and target-memory gates; erroneous writes use selected IoU<0.2 as a localization proxy, not semantic identity labels.',
                            'harm_scope': 'Immediate same-state rescue/harm does not capture later recovery-opportunity losses or provide a causal future comparison.',
                            'rates': 'Null means denominator0, not zero error. Modal write counts include all tracked frames; pollution denominators use valid GT only.'}}
