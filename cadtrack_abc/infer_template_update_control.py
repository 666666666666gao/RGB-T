"""Full98 or full benchmark inference for four fixed-weight R1 controls."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import iou, read_pair, xyxy
from .search_actions import CADTrackSearchActionPolicy
from .template_update_control import R1TemplateUpdateControl, TEMPLATE_POLICIES


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--template-policy', choices=TEMPLATE_POLICIES, required=True)
    parser.add_argument('--dataset', choices=['development', 'lasher', 'rgbt210', 'rgbt234'], required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', help='Required for the fixed98 development split.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--update-interval', type=int, default=5)
    parser.add_argument('--update-threshold', type=float, default=.65)
    args = parser.parse_args()
    assert args.shards == 2 and 0 <= args.shard < args.shards
    config = json.loads(Path(args.config).read_text())
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    policy = CADTrackSearchActionPolicy(config['pretrained'], config['threshold']).cuda().eval()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert payload['policy_family'] == 'search_actions'
    assert payload['config']['pretrained'] == config['pretrained']
    assert payload['config']['threshold'] == config['threshold']
    policy.abc.load_state_dict(payload['abc'], strict=True)
    policy.future_action.load_state_dict(payload['future_action'], strict=True)
    policy.search_action.load_state_dict(payload['search_action'], strict=True)
    model = R1TemplateUpdateControl(policy, args.template_policy, args.update_interval, args.update_threshold)
    root, output = Path(args.root), Path(args.output)
    if args.dataset == 'development':
        assert args.manifest and root == Path(config['train_root'])
        manifest = json.loads(Path(args.manifest).read_text())
        names = sorted(manifest['validation_sequences'])
        assert len(names) == len(set(names)) == 98
        assert not set(names) & set(manifest['train_sequences'])
    else:
        names = sorted(path.name for path in root.iterdir() if path.is_dir())
        assert len(names) == {'lasher': 245, 'rgbt210': 210, 'rgbt234': 234}[args.dataset]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, name in enumerate(names[args.shard::args.shards], 1):
        sequence = root / name
        rgb, tir = sorted((sequence / 'visible').iterdir()), sorted((sequence / 'infrared').iterdir())
        assert len(rgb) == len(tir)
        annotation = sequence / ('visible.txt' if args.dataset == 'rgbt234' else 'init.txt')
        with annotation.open() as stream:
            initial = np.fromstring(stream.readline().strip(), sep=',', dtype=np.float32)
        assert initial.shape == (4,) and (initial[2:] > 0).all()
        state = model.initialize([read_pair(rgb[0], tir[0])], torch.tensor(initial[None]), torch.tensor([0]))
        boxes, latencies, diagnostics = [initial], [], []
        for frame in range(1, len(rgb)):
            torch.cuda.synchronize()
            started = time.perf_counter()
            observation, state = model.step([read_pair(rgb[frame], tir[frame])], torch.tensor([frame]), state)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            boxes.append(observation['box'][0].cpu().numpy())
            diagnostics.append({
                'choice': int(observation['choice'][0]), 'extra_forwards': observation['extra_forwards'],
                'geometry_commit': bool(observation['geometry_commit'][0]),
                'template_write_requested': bool(observation['write'][0]),
                'template_commit_rgb_tir': observation['template_commit'][0].cpu().tolist(),
                'target_memory_commit_rgb_tir': observation['target_memory_commit'][0].cpu().tolist(),
                'next_geometry_reference_xywh': observation['geometry_reference'][0].cpu().tolist(),
                'keep_box_xywh': observation['keep_box'][0].cpu().tolist(),
                'candidate_boxes_xyxy': observation['pair']['boxes'][0].cpu().tolist(),
                'predicted_quality': float(observation['selected_quality'][0]),
                'future_mode': observation['future_mode'],
                'future_net_advantage': observation['future_net_advantage'],
                'search_region_rank': observation['search_region_rank'],
                'search_region_mode': observation['search_region_mode'],
                'search_net_gain': observation['search_net_gain'],
                'template_control': observation['template_control'],
            })
        prediction = np.stack(boxes)
        np.savetxt(output / (name + '.txt'), prediction, delimiter='\t', fmt='%.3f')
        np.save(output / (name + '_latency.npy'), np.asarray(latencies))
        (output / (name + '_mechanisms.json')).write_text(json.dumps(diagnostics))
        record = {'sequence': name, 'frames': len(boxes), 'seconds': sum(latencies),
                  'extra_forwards': sum(item['extra_forwards'] for item in diagnostics)}
        if args.dataset == 'development':
            # The complete causal trajectory is finished before scoring GT.
            truth = np.loadtxt(annotation, delimiter=',', dtype=np.float32).reshape(-1, 4)
            assert truth.shape == prediction.shape
            valid = (truth[:, 2:] > 0).all(-1)
            overlaps = iou(xyxy(torch.tensor(prediction)), xyxy(torch.tensor(truth))).numpy()[valid]
            record.update(valid_frames=int(valid.sum()), iou_sum=float(overlaps.sum()), mean_iou=float(overlaps.mean()))
        records.append(record)
        print(json.dumps(dict(event='sequence_complete', index=index, **record)), flush=True)
    receipt = {
        'completed': True, 'full_videos': True, 'dataset': args.dataset,
        'checkpoint': str(Path(args.checkpoint).resolve()), 'epoch': payload['epoch'],
        'policy_family': 'search_actions', 'template_policy': args.template_policy,
        'resolved_template_interval': args.update_interval, 'resolved_template_threshold': args.update_threshold,
        'shard': args.shard, 'shards': args.shards, 'records': records,
        'sequences': len(records), 'frames': sum(record['frames'] for record in records),
        'strict_base_and_abc_load': True, 'first_box_only_initialization': True,
        'formal_metrics': None, 'official_test_used': args.dataset != 'development',
        'template_proxy_scope': 'For cad_local, template-source IoU must use template_control.source_box_xywh; selected-box IoU is not template-source IoU.',
    }
    prefix = 'development_shard' if args.dataset == 'development' else 'complete_shard'
    (output / (prefix + str(args.shard) + '.json')).write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
