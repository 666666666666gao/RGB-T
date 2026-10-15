"""Fresh causal local/one-extra evidence for R3 candidate calibration.

The raw-stage teacher's decision heads are still untrained and frozen. Only
the first annotation initializes its prefix; query GT supplies overlap labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import read_pair, xyxy, iou
from .dense_consequences import assert_same_snapshot
from .dense_main_tracker import DenseABCTracker


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    checkpoint = Path(args.checkpoint).resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['model_family'] == 'CADTrackDenseABC-R3' and payload['stage'] == 'dense_motion'
    config = payload['config']
    assert not config['future_actions_trained'] and not config['search_actions_trained']
    manifest = json.loads(Path(args.manifest).read_text())
    allowed = set(manifest['train_sequences' if args.split == 'train' else 'validation_sequences'])
    queries = json.loads(Path(args.queries).read_text())
    assert queries and len({(q['sequence'], q['frame']) for q in queries}) == len(queries)
    assert {q['sequence'] for q in queries} <= allowed
    tracker = DenseABCTracker(config['pretrained'], config['intervention_threshold'])
    tracker.load_payload(payload)
    tracker.network.requires_grad_(False).eval()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for name in sorted({q['sequence'] for q in queries}):
        directory = Path(config['train_root']) / name
        labels = np.loadtxt(directory / 'init.txt', delimiter=',', dtype=np.float64).reshape(-1, 4)
        rgb, tir = sorted((directory / 'visible').iterdir()), sorted((directory / 'infrared').iterdir())
        assert len(rgb) == len(tir) == len(labels) and (labels[0, 2:] > 0).all()
        tracker.initialize(read_pair(rgb[0], tir[0]), labels[0])
        next_frame = 1
        for item in sorted((q for q in queries if q['sequence'] == name), key=lambda q: q['frame']):
            frame = item['frame']
            assert next_frame <= frame < len(labels) and (labels[frame, 2:] > 0).all()
            while next_frame < frame:
                tracker.track(read_pair(rgb[next_frame], tir[next_frame]))
                next_frame += 1
            image = read_pair(rgb[frame], tir[frame])
            prior = tracker.snapshot()
            local = tracker.observe_local(image)
            order = local['distribution']['log_weights'][0].argsort(descending=True, stable=True)
            queries_observed = [local] + [tracker.observe_extra(image, local, int(mode)) for mode in order]
            assert_same_snapshot(prior, tracker.snapshot())
            contexts = []
            for query in queries_observed:
                boxes = torch.tensor(query['native_boxes'], dtype=torch.float64)
                quality = iou(xyxy(boxes), xyxy(torch.as_tensor(labels[frame]))[None]).float()
                contexts.append(dict(
                    candidate={key: value.detach().cpu() for key, value in query['pair'].items()
                        if key in ('boxes', 'raw', 'hann', 'fused')},
                    appearance={key: value.detach().cpu() for key, value in query['appearance'].items()},
                    quality=quality))
            distribution = {key: value.detach().cpu() for key, value in local['distribution'].items()}
            filename = f'{name}_{frame:06d}.pt'
            metadata = dict(sequence=name, frame=frame, split=args.split, query_spec=item,
                checkpoint_sha256=digest, model_family=payload['model_family'],
                full_predicted_prefix_from_frame0=True, GT_for_quality_only=True)
            torch.save(dict(metadata=metadata, contexts=contexts, distribution=distribution), output / filename)
            tracker.track(image)
            assert tracker.frame == frame
            records.append(dict(metadata, file=filename))
            print(json.dumps(dict(event='evidence_query_complete', **metadata)), flush=True)
            next_frame = frame + 1
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    receipt = dict(completed=True, split=args.split, checkpoint_sha256=digest,
        model_family=payload['model_family'], queries=len(records), records=records,
        contexts_per_query=4, extra_contexts_are_independent=True,
        future_images_used=False, identity_truth_or_sensor_quality_labels=False,
        scope='Current localization overlap calibration; not future consequences or formal tracking metrics.')
    (output / 'collection_complete.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
