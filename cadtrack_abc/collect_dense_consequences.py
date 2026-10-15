"""Collect fresh R3 consequences from its own full predicted prefixes.

This entry does not start a GPU job itself. It is deployed only after the R2
pipeline, dense-motion training/selection and actual R3 qualification close.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import read_pair
from .dense_consequences import collect_matched_query
from .dense_main_tracker import DenseABCTracker


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--horizon', type=int, default=3)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3')
    assert args.horizon >= 1
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    checkpoint = Path(args.checkpoint).resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['model_family'] == 'CADTrackDenseABC-R3'
    config = payload['config']
    manifest = json.loads(Path(args.manifest).read_text())
    allowed = set(manifest['train_sequences' if args.split == 'train' else 'validation_sequences'])
    queries = json.loads(Path(args.queries).read_text())
    assert queries and len({(item['sequence'], item['frame']) for item in queries}) == len(queries)
    assert {item['sequence'] for item in queries} <= allowed
    tracker = DenseABCTracker(config['pretrained'], config['intervention_threshold'])
    tracker.load_payload(payload)
    tracker.network.requires_grad_(False).eval()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    for name in sorted({item['sequence'] for item in queries}):
        directory = Path(config['train_root']) / name
        labels = np.loadtxt(directory / 'init.txt', delimiter=',', dtype=np.float32).reshape(-1, 4)
        rgb, tir = sorted((directory / 'visible').iterdir()), sorted((directory / 'infrared').iterdir())
        assert len(rgb) == len(tir) == len(labels) and (labels[0, 2:] > 0).all()
        tracker.initialize(read_pair(rgb[0], tir[0]), labels[0])
        next_frame = 1
        for query in sorted((item for item in queries if item['sequence'] == name), key=lambda item: item['frame']):
            frame = query['frame']
            assert next_frame <= frame and frame + args.horizon < len(labels)
            truth = labels[frame:frame + args.horizon + 1]
            assert (truth[0, 2:] > 0).all() and (truth[1:, 2:] > 0).all(-1).any()
            while next_frame < frame:
                tracker.track(read_pair(rgb[next_frame], tir[next_frame]))
                next_frame += 1
            assert tracker.frame == frame - 1
            image = read_pair(rgb[frame], tir[frame])
            future_images = [read_pair(rgb[index], tir[index])
                for index in range(frame + 1, frame + args.horizon + 1)]
            consequences = collect_matched_query(tracker, image, future_images, truth)
            assert tracker.frame == frame
            filename = f'{name}_{frame:06d}.pt'
            metadata = dict(sequence=name, frame=frame, split=args.split, query_spec=query,
                checkpoint=str(checkpoint), checkpoint_sha256=digest, teacher_epoch=payload['epoch'],
                model_family=payload['model_family'], horizon=args.horizon,
                geometry_defer_semantics='R3 quarantine: no geometry or modality-memory/template commit',
                full_predicted_prefix_from_frame0=True, first_GT_only_initialization=True,
                future_images_and_GT_labels_only=True, old_R1_action_labels_used=False)
            torch.save(dict(metadata=metadata, **consequences), output / filename)
            record = dict(metadata, file=filename, actions=consequences['actions'],
                actual_native_replay_equal=consequences['actual_native_replay_equal'],
                seconds_since_start=time.monotonic() - started)
            records.append(record)
            print(json.dumps(dict(event='query_complete', **record)), flush=True)
            next_frame = frame + 1
    assert len(records) == len(queries)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    receipt = dict(completed=True, model_family=payload['model_family'], checkpoint_sha256=digest,
        split=args.split, queries=len(records), records=records, horizon=args.horizon,
        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']), prefix_policy='actual loaded R3 teacher configuration',
        action_label_semantics='Fresh native R3 counterfactuals, all six actions, same frozen continuation policy.',
        future_harm_available_separately=True, deployed_max_extra_visual_forwards=1,
        collection_only=True, training_complete=False, formal_metrics=None, goal_achieved=False)
    (output / 'collection_complete.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
