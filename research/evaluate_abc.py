"""Causal online ABC evaluation; GT after row1 never enters tracker state."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .abc_tracker import ABCTracker
from .candidate_learning import FrozenCandidateExtractor
from .evaluate_online import read_pair
from .temporal_modules import TemporalModules


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['lasher', 'rgbt234'], required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--pretrained', default='pretrained_models/gola_b224.bin')
    p.add_argument('--c1-head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--output', required=True)
    p.add_argument('--variant', choices=['abc', 'abc_box_only'], default='abc')
    p.add_argument('--validation-split', help='Restrict to original TRAIN-held-out names, never an official result.')
    p.add_argument('--sequence-offset', type=int, default=0)
    p.add_argument('--limit-sequences', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--branches', type=int, default=3)
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--switch-patience', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def numpy_value(value):
    return value.cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value).copy()


@torch.inference_mode()
def track(paths_v, paths_i, initial, extractor, modules, device, args):
    tracker = ABCTracker(extractor, modules, read_pair(paths_v[0], paths_i[0], device), initial,
                         torch.float16, args.branches, args.window, args.switch_patience,
                         args.variant == 'abc')
    protected = tracker.identity_anchor.clone()
    predictions, latency, decisions, events = [initial.copy()], [], [], []
    for frame in range(1, len(paths_v)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions.append(tracker.step(read_pair(paths_v[frame], paths_i[frame], device)))
        torch.cuda.synchronize(device)
        latency.append(time.perf_counter() - started)
        # Diagnostic transfer/serialization is outside measured tracker latency.
        decisions.append({key: numpy_value(value) for key, value in tracker.last_decision.items()})
        events.append(tracker.last_event)
    assert torch.equal(tracker.identity_anchor, protected)
    assert len(decisions) == len(latency) == len(predictions) - 1
    for box, decision in zip(predictions[1:], decisions):
        assert np.array_equal(box, decision['boxes_xyxy'][int(decision['choice'])])
    predictions = np.asarray(predictions)
    predictions[:, 2:] -= predictions[:, :2]
    return predictions, np.asarray(latency), decisions, events, tracker.stats


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    c1 = torch.load(args.c1_head, map_location='cpu', weights_only=False)
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    assert checkpoint['module'] == 'ABC_temporal'
    architecture = checkpoint['args']
    assert architecture['c1_head'] == args.c1_head
    modules = TemporalModules(c1, architecture['slots'], architecture['motion_history'],
                              architecture['modes'], checkpoint['horizon']).to(device)
    modules.load_state_dict(checkpoint['model'], strict=True)
    modules.eval().requires_grad_(False)
    extractor = FrozenCandidateExtractor(args.pretrained, c1['args']['candidates'], .45, c1['args']['nms_iou']).to(device)
    sequences = sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    if args.validation_split:
        split = json.loads(Path(args.validation_split).read_text())
        assert not set(split['train']) & set(split['validation']) and args.dataset == 'lasher'
        sequences = [p for p in sequences if p.name in split['validation']]
    sequences = sequences[args.sequence_offset:]
    if args.limit_sequences:
        sequences = sequences[:args.limit_sequences]
    assert sequences
    smoke = bool(args.limit_sequences or args.max_frames or args.sequence_offset or args.validation_split)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'scope': 'strict online ABC with bounded private memory/template/motion restoration',
                           'smoke_only': smoke, 'amp_dtype': 'float16', 'head_epoch': checkpoint['epoch'],
                           'pretrained_load': extractor.load_receipt,
                           'candidate_box_scaling': 'float32_multiply_then_double',
                           'timing_excludes_first_frame_initialization': True,
                           'diagnostic_serialization_after_timing': True,
                           'candidate_diagnostic_requires_branch_aware_template_provenance': True,
                           'gt_input': 'only first annotation initializes; later GT only offline scoring',
                           'initialization': 'init.txt first row' if args.dataset == 'lasher' else 'visible.txt first row'}
    (out / 'inference_config.json').write_text(json.dumps(config, indent=2))
    records, all_latency, started = [], [], time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with (out / 'progress.jsonl').open('w') as stream:
        for index, sequence in enumerate(sequences, 1):
            visible = sorted(p for p in (sequence / 'visible').iterdir() if p.is_file())
            infrared = sorted(p for p in (sequence / 'infrared').iterdir() if p.is_file())
            assert len(visible) == len(infrared)
            with (sequence / ('init.txt' if args.dataset == 'lasher' else 'visible.txt')).open() as labels:
                initial = np.fromstring(labels.readline().strip(), sep=',')
            assert initial.shape == (4,)
            initial[2:] += initial[:2]
            if args.max_frames:
                visible, infrared = visible[:args.max_frames], infrared[:args.max_frames]
            prediction, latency, decisions, events, stats = track(visible, infrared, initial, extractor, modules, device, args)
            assert np.isfinite(prediction).all()
            np.savetxt(out / (sequence.name + '.txt'), prediction, delimiter='\t', fmt='%.3f')
            np.save(out / (sequence.name + '_latency.npy'), latency)
            np.savez_compressed(out / (sequence.name + '_abc_decisions.npz'),
                                **{key: np.stack([r[key] for r in decisions]) for key in decisions[0]})
            (out / (sequence.name + '_branch_events.json')).write_text(json.dumps(events))
            record = {'sequence': sequence.name, 'sequence_index': index, 'sequences': len(sequences),
                      'frames': len(prediction), 'elapsed_seconds': float(latency.sum()),
                      'template_updates': stats['template_updates'],
                      'alternative_selections': stats['alternative_selections'], 'branch_stats': stats}
            records.append(record)
            all_latency.extend(latency.tolist())
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            print('SEQUENCE', json.dumps(record), flush=True)
    latency = np.asarray(all_latency)
    receipt = {'completed': True, 'sequences': len(records), 'frames': sum(r['frames'] for r in records),
               'smoke_only': smoke, 'records': records, 'branch_timeline_recorded': True,
               'candidate_timeline_recorded': False,
               'fps_including_decode_crop_update': len(latency) / latency.sum(),
               'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
               'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
               'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
               'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
               'wall_seconds_including_initialization_and_diagnostic_serialization': time.perf_counter() - started,
               'official_accuracy': 'not computed here; native actual-GT collector required'}
    (out / 'inference_completion.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETED', json.dumps({k: v for k, v in receipt.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
