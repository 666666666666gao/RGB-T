"""Offline GT diagnostics separating actual visual sources from private ABC state."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .collect_candidate_metrics import ground_truth
from .collect_core_metrics import localization_quality


def xywh(boxes):
    result = boxes.copy()
    result[..., 2:] -= result[..., :2]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('lasher', 'rgbt234'), required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    path = Path(args.predictions)
    config = json.loads((path / 'inference_config.json').read_text())
    done = json.loads((path / 'inference_completion.json').read_text())
    assert done['completed'] and config['reference_mode'] in ('parent', 'geometry', 'visual', 'visual_motion')
    rows = []
    for record in done['records']:
        name, frames = record['sequence'], record['frames']
        pred = np.loadtxt(path / (name + '.txt'), ndmin=2)
        gt = ground_truth(args.root, name, args.dataset)
        assert len(pred) == frames
        assert all(len(value) == frames for value in gt.values()) if isinstance(gt, dict) else len(gt) == frames
        output_quality, known = localization_quality(pred, gt, args.dataset)
        with np.load(path / (name + '_recoverability_decisions.npz')) as timeline:
            protected = np.concatenate((pred[:1], xywh(timeline['protected_reference_box'])), 0)
            motion = np.concatenate((pred[:1], xywh(timeline['committed_motion_observation'])), 0)
            ref_quality, ref_known = localization_quality(protected, gt, args.dataset)
            motion_quality, motion_known = localization_quality(motion, gt, args.dataset)
            assert np.array_equal(known, ref_known) and np.array_equal(known, motion_known)
            # The prior primary source uses a previously observed frame. Query
            # GT is not a template identity label; compare to its source-frame GT.
            source_frames = timeline['incoming_primary_template_frame']
            assert ((source_frames >= 0) & (source_frames < np.arange(1, frames))).all()
            source_boxes = pred[np.maximum(source_frames, 0)]
            if config['reference_mode'] in ('visual', 'visual_motion'):
                source_boxes = protected[source_frames]
            source_gt = ({key: np.concatenate((value[:1], value[source_frames]), 0) for key, value in gt.items()}
                         if isinstance(gt, dict) else np.concatenate((gt[:1], gt[source_frames]), 0))
            source_quality, source_known = localization_quality(np.concatenate((pred[:1], source_boxes), 0), source_gt, args.dataset)
            source_quality, source_known = source_quality[1:], source_known[1:]
            usable = known[1:]
            rows.append(dict(sequence=name, frames=frames, valid_tracking_frames=int(usable.sum()),
                failed_output_frames=int((usable & (output_quality[1:] < .2)).sum()),
                failed_protected_reference_frames=int((usable & (ref_quality[1:] < .2)).sum()),
                failed_motion_observation_frames=int((usable & (motion_quality[1:] < .2)).sum()),
                failed_output_correct_reference_frames=int((usable & (output_quality[1:] < .2) & (ref_quality[1:] >= .5)).sum()),
                correct_output_failed_reference_frames=int((usable & (output_quality[1:] >= .5) & (ref_quality[1:] < .2)).sum()),
                known_primary_template_source_frames=int(source_known.sum()),
                failed_primary_template_source_frames_localization_proxy=int((source_known & (source_quality < .2)).sum()),
                protected_reference_template_writes=int(timeline['protected_reference_template_updated'].sum()),
                actual_primary_template_writes=int((timeline['protected_reference_template_updated']
                    if config['reference_mode'] in ('visual', 'visual_motion') else timeline['template_updated']).sum())))
    report = dict(completed=True, dataset=args.dataset, reference_mode=config['reference_mode'],
        sequences=len(rows), frames=sum(row['frames'] for row in rows), model_GT_inference_inputs='initialization only',
        new_optimizer_updates=0, reference_is_prediction_not_ground_truth=True,
        localization_proxy_is_not_exact_distractor_identity=True,
        private_ABC_templates_not_primary_sources=config['reference_mode'] in ('visual', 'visual_motion'),
        counters={key: sum(row[key] for row in rows) for key in rows[0] if key != 'sequence'})
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    (out / 'reference_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (out / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
