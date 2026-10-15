"""Finite TRAIN-only epoch0 parity qualification; never a baseline benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import VideoClips
from .dense_tracker import DenseCADTracker
from .pure_baseline_tracker import PureCADTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = json.loads(Path(args.config).read_text())
    pretrained = Path(config['pretrained'])
    digest = hashlib.sha256(pretrained.read_bytes()).hexdigest()
    assert digest == 'cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731'
    names = sorted(p.name for p in Path(config['train_root']).iterdir() if p.is_dir())
    development = set(random.Random(20261013).sample(names, 98))
    names = sorted(set(names)-development)
    assert len(names) == 881
    dataset = VideoClips(config['train_root'], names, 2, 42, steps=12)
    reference = PureCADTracker(pretrained)
    control = DenseCADTracker(pretrained, False)
    prompt = DenseCADTracker(pretrained, True)
    assert torch.count_nonzero(prompt.network.prompt.output.weight) == 0
    assert torch.count_nonzero(prompt.network.prompt.output.bias) == 0
    records = []
    started = time.monotonic()
    for index in range(2):
        clip = dataset[index]
        initial = clip['boxes'][0].astype(np.float64)
        for tracker in (reference, control, prompt):
            tracker.initialize(clip['images'][0], initial)
        for frame in range(1, 13):
            image = clip['images'][frame]
            pure = reference.track(image)
            observed = [control.track(image), prompt.track(image)]
            differences = [float(np.max(np.abs(value['box']-pure['box']))) for value in observed]
            same = [np.array_equal(value['box'], pure['box']) and
                value['confidence'] == pure['confidence'] and
                value['template_updated'] == pure['template_updated'] for value in observed]
            template_equal = [all(torch.equal(a,b) and a.stride() == b.stride()
                for a,b in zip(reference.templates, tracker.templates)) for tracker in (control, prompt)]
            row = dict(sequence=clip['name'],frame=int(clip['frames'][frame]),
                maximum_box_errors=differences, exact_outputs_H_HA=same,
                template_values_and_strides_equal_H_HA=template_equal,
                template_updated=pure['template_updated'])
            records.append(row)
            print(json.dumps(row), flush=True)
            assert all(same) and all(template_equal), row
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='DENSE_EPOCH0_PARITY_PASS',records=records,
        actual_causal_query_frames=len(records),train_clips=2,
        pretrained_sha256=digest,pretrained_strict_tensor_count=reference.network.loaded_tensors,
        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),elapsed_seconds=time.monotonic()-started,
        later_ground_truth_used_by_tracker=False,official_baseline_lasher_rerun=False,
        gradient_probe_complete=False,formal_training_started=False,
        scope='Two TRAIN clips,12 causal predictions each, exact outputs/confidence/template writes/values/strides. Epoch0 only, not accuracy or full M0 completion.')
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
