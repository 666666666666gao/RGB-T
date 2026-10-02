"""Measure exact untrained-C1 trajectory equivalence on train-held-out videos.

This control uses complete selected videos, not official benchmark scores.
GT only initializes frame 0. Later labels are never passed to the tracker
or used for decisions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .candidate_learning import CandidateQualityHead, FrozenCandidateExtractor
from .evaluate_online import track_sequence
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--sequence-offset', type=int, default=0)
    p.add_argument('--sequences', type=int, default=3)
    p.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    p.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    p.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json')
    p.add_argument('--pretrained', default='pretrained_models/gola_b224.bin')
    args = p.parse_args()
    assert args.sequences > 0 and args.sequence_offset >= 0
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    split = json.loads(Path(args.split).read_text())
    assert not set(split['train']) & set(split['validation'])
    names = sorted(split['validation'])[args.sequence_offset:args.sequence_offset + args.sequences]
    assert len(names) == args.sequences
    data = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    lookup = {data[index].get_name(): index for index in range(len(data))}
    extractor = FrozenCandidateExtractor(args.pretrained).to(device)
    head = CandidateQualityHead().to(device).eval().requires_grad_(False)
    assert not head.scorer[-1].weight.count_nonzero() and not head.scorer[-1].bias.count_nonzero()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        seq = data[lookup[name]]
        pairs = [seq[frame].get_image_path() for frame in range(len(seq))]
        visible, infrared = [Path(pair[0]) for pair in pairs], [Path(pair[1]) for pair in pairs]
        initial = seq[0].get_bounding_box().copy()
        baseline, _, base_updates, _ = track_sequence(visible, infrared, initial, extractor, None, device, torch.float16)
        zero, _, zero_updates, selections = track_sequence(visible, infrared, initial, extractor, head, device, torch.float16)
        difference = np.abs(zero - baseline)
        assert np.isfinite(difference).all()
        np.savez_compressed(out / (name + '.npz'), baseline_xywh=baseline, zero_head_xywh=zero)
        record = {'sequence': name, 'frames': len(seq), 'baseline_updates': base_updates,
                  'zero_head_updates': zero_updates, 'zero_head_alternative_selections': selections,
                  'maximum_coordinate_difference_pixels': float(difference.max()),
                  'rounded_three_decimal_different_frames': int((np.round(zero, 3) != np.round(baseline, 3)).any(1).sum()),
                  'raw_coordinates_identical': bool(np.array_equal(zero, baseline))}
        records.append(record)
        print(json.dumps(record), flush=True)
    receipt = {'completed': True, 'scope': __doc__, 'args': vars(args),
               'sequences': len(records), 'frames': sum(row['frames'] for row in records),
               'records': records, 'head_last_layer_zero': True, 'pretrained_base_frozen': True,
               'official_benchmark_accuracy': False,
               'all_three_decimal_trajectories_identical': all(row['rounded_three_decimal_different_frames'] == 0 for row in records)}
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
