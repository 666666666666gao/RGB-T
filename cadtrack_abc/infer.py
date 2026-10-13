import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import read_pair
from .model import CADTrackABC


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=['lasher', 'rgbt234'], required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    model = CADTrackABC(config['pretrained'], config['threshold']).cuda().eval()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.abc.load_state_dict(payload['abc'], strict=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_sequences = sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    expected = 245 if args.dataset == 'lasher' else 234
    assert len(all_sequences) == expected
    records = []
    for index, sequence in enumerate(all_sequences[args.shard::args.shards], 1):
        rgb = sorted((sequence / 'visible').iterdir())
        tir = sorted((sequence / 'infrared').iterdir())
        assert len(rgb) == len(tir)
        annotation = sequence / ('init.txt' if args.dataset == 'lasher' else 'visible.txt')
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
            diagnostics.append({'choice': int(observation['choice'][0]),
                                'extra_forwards': observation['extra_forwards'],
                                'geometry_commit': bool(observation['geometry_commit'][0]),
                                'template_write_requested': bool(observation['write'][0]),
                                'template_commit_rgb_tir': observation['template_commit'][0].cpu().tolist(),
                                'target_memory_commit_rgb_tir': observation['target_memory_commit'][0].cpu().tolist(),
                                'next_geometry_reference_xywh': observation['geometry_reference'][0].cpu().tolist(),
                                'keep_box_xywh': observation['keep_box'][0].cpu().tolist(),
                                'candidate_boxes_xyxy': observation['pair']['boxes'][0].cpu().tolist(),
                                'predicted_quality': observation['selected_quality'][0].item()})
        np.savetxt(output / (sequence.name + '.txt'), np.stack(boxes), delimiter='\t', fmt='%.3f')
        np.save(output / (sequence.name + '_latency.npy'), np.array(latencies))
        (output / (sequence.name + '_mechanisms.json')).write_text(json.dumps(diagnostics))
        record = {'sequence': sequence.name, 'frames': len(boxes), 'seconds': sum(latencies),
                  'extra_forwards': sum(d['extra_forwards'] for d in diagnostics)}
        records.append(record)
        print(json.dumps(dict(event='sequence_complete', shard=args.shard, index=index, **record)), flush=True)
    receipt = {'completed': True, 'dataset': args.dataset, 'shard': args.shard, 'shards': args.shards,
               'checkpoint': args.checkpoint, 'epoch': payload['epoch'], 'records': records,
               'sequences': len(records), 'frames': sum(r['frames'] for r in records),
               'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
               'strict_base_and_abc_load': True, 'first_box_only_initialization': True}
    (output / ('complete_shard' + str(args.shard) + '.json')).write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
