"""Full pure-CAD inference on the two datasets lacking a user baseline score."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import read_pair
from .pure_baseline_tracker import PureCADTracker


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=['rgbt210', 'rgbt234'], required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    assert args.shard in (0, 1)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    tracker = PureCADTracker(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sequences = sorted(path for path in Path(args.root).iterdir() if path.is_dir())
    expected = {'rgbt210': 210, 'rgbt234': 234}[args.dataset]
    assert len(sequences) == expected
    records = []
    for index, sequence in enumerate(sequences[args.shard::2], 1):
        rgb, tir = sorted((sequence / 'visible').iterdir()), sorted((sequence / 'infrared').iterdir())
        annotation = sequence / ('visible.txt' if args.dataset == 'rgbt234' else 'init.txt')
        labels = np.loadtxt(annotation, delimiter=',', ndmin=2)
        assert len(rgb) == len(tir) == len(labels)
        initial = labels[0]
        assert initial.shape == (4,) and (initial[2:] > 0).all()
        tracker.initialize(read_pair(rgb[0], tir[0]), initial)
        boxes, latencies, updates = [initial], [], 0
        for frame in range(1, len(rgb)):
            torch.cuda.synchronize()
            started = time.perf_counter()
            observation = tracker.track(read_pair(rgb[frame], tir[frame]))
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            boxes.append(observation['box'])
            updates += observation['template_updated']
        np.savetxt(output / f'{sequence.name}.txt', np.stack(boxes), delimiter='\t', fmt='%.3f')
        np.save(output / f'{sequence.name}_latency.npy', np.asarray(latencies))
        record = dict(sequence=sequence.name, frames=len(boxes), seconds=sum(latencies),
                      template_updates=updates)
        records.append(record)
        print(json.dumps(dict(event='sequence_complete', shard=args.shard, index=index, **record)), flush=True)
    receipt = dict(completed=True, dataset=args.dataset, checkpoint=args.checkpoint,
        epoch=tracker.network.loaded_epoch, loaded_tensors=tracker.network.loaded_tensors,
        policy_family='pure_cad', shard=args.shard, shards=2, records=records,
        sequences=len(records), frames=sum(row['frames'] for row in records),
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
        strict_base_load=True, first_box_only_initialization=True,
        protocol='Original CAD crop/Hann argmax/Center bbox/map/clip; weighted response >0.65 every5 frames; protected first template.')
    (output / f'complete_shard{args.shard}.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
