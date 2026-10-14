"""Continuous tracking on the fixed98 TRAIN-held-out development videos."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import iou, read_pair, xyxy
from .future_actions import CADTrackFuturePolicy
from .search_actions import CADTrackSearchActionPolicy


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy', choices=['future_actions', 'search_actions'], default='future_actions')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--sanity', action='store_true', help='One DEV video, eight frames; never a full-validation receipt.')
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    config = json.loads(Path(args.config).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    names = sorted(manifest['validation_sequences'])
    assert len(names) == len(set(names)) == 98
    assert not set(names) & set(manifest['train_sequences'])
    if args.sanity:
        assert args.shard == 0 and args.shards == 1
        names = names[:1]
    torch.set_num_threads(4)
    if args.device == 'cuda':
        torch.cuda.set_device(0)
    policy = CADTrackSearchActionPolicy if args.policy == 'search_actions' else CADTrackFuturePolicy
    model = policy(config['pretrained'], config['threshold']).to(args.device).eval()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert payload['policy_family'] == args.policy
    assert payload['config']['pretrained'] == config['pretrained'] and payload['config']['threshold'] == config['threshold']
    model.abc.load_state_dict(payload['abc'], strict=True)
    model.future_action.load_state_dict(payload['future_action'], strict=True)
    if args.policy == 'search_actions':
        model.search_action.load_state_dict(payload['search_action'], strict=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records, started = [], time.perf_counter()
    for name in names[args.shard::args.shards]:
        sequence = Path(config['train_root']) / name
        rgb, tir = sorted((sequence / 'visible').iterdir()), sorted((sequence / 'infrared').iterdir())
        labels = np.loadtxt(sequence / 'init.txt', delimiter=',', dtype=np.float32).reshape(-1, 4)
        assert len(rgb) == len(tir) == len(labels) and (labels[0, 2:] > 0).all()
        count = min(8, len(rgb)) if args.sanity else len(rgb)
        state = model.initialize([read_pair(rgb[0], tir[0])], torch.tensor(labels[:1]), torch.tensor([0]))
        boxes, extra, interventions = [labels[0]], 0, 0
        for frame in range(1, count):
            observation, state = model.step([read_pair(rgb[frame], tir[frame])], torch.tensor([frame]), state)
            boxes.append(observation['box'][0].cpu().numpy())
            extra += observation['extra_forwards']
            interventions += int(observation['future_net_advantage'] > config['threshold'])
        # Annotations after frame0 are read for scoring only, after the causal run.
        prediction = np.stack(boxes)
        target = labels[:count]
        valid = (target[:, 2:] > 0).all(-1)
        overlaps = iou(xyxy(torch.tensor(prediction)), xyxy(torch.tensor(target))).numpy()[valid]
        np.savetxt(output / (name + '.txt'), prediction, delimiter='\t', fmt='%.6f')
        record = {'sequence': name, 'frames': count, 'valid_frames': int(valid.sum()),
                  'iou_sum': float(overlaps.sum()), 'mean_iou': float(overlaps.mean()),
                  'extra_forwards': extra, 'future_interventions': interventions}
        records.append(record)
        print(json.dumps(record), flush=True)
    receipt = {'completed': not args.sanity, 'full_videos': not args.sanity, 'sanity': args.sanity,
               'checkpoint': str(Path(args.checkpoint).resolve()), 'epoch': payload['epoch'],
               'policy_family': args.policy, 'shard': args.shard, 'shards': args.shards,
               'records': records, 'sequences': len(records), 'seconds': time.perf_counter() - started,
               'scope': 'Repeated TRAIN-held-out development set; positive-width/height IoU, not official LasHeR SR.',
               'strict_abc_and_future_head_load': True, 'first_gt_only_state_initialization': True,
               'official_test_used': False, 'formal_metrics': None}
    (output / ('development_shard' + str(args.shard) + '.json')).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(receipt, records=None)), flush=True)


if __name__ == '__main__':
    main()
