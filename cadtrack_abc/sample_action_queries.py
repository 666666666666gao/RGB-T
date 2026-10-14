"""TRAIN/DEV queries from actual teacher trajectories and critical-event windows.

This is offline supervision sampling. GT after frame0 is consulted only after
the teacher's prediction and never enters its state or search. Official test
videos are not accepted through the fixed training manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import iou, read_pair, xyxy
from .future_actions import CADTrackFuturePolicy
from .model import CADTrackABC

EVENT_OFFSETS = (-2, -1, 0, 1, 3)
EVENT_TYPES = ('loss_transition', 'wrong_identity_commit', 'missed_correct_candidate')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--teacher', required=True, help='Immutable selected teacher, not the live overwritten best.')
    parser.add_argument('--teacher-policy', choices=['native', 'future_actions'], default='native')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--sanity', action='store_true', help='One TRAIN video, sixteen frames; never full split completion.')
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards and args.horizon >= 1
    config = json.loads(Path(args.config).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    names = sorted(manifest['train_sequences' if args.split == 'train' else 'validation_sequences'])
    assert len(names) == len(set(names)) == (881 if args.split == 'train' else 98)
    assert not set(manifest['train_sequences']) & set(manifest['validation_sequences'])
    if args.sanity:
        assert args.split == 'train' and args.shard == 0 and args.shards == 1
        names = names[:1]
    torch.set_num_threads(4)
    if args.device == 'cuda':
        torch.cuda.set_device(0)
    teacher_sha = hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest()
    payload = torch.load(args.teacher, map_location='cpu', weights_only=False)
    assert payload['config']['pretrained'] == config['pretrained'] and payload['config']['threshold'] == config['threshold']
    policy = CADTrackABC if args.teacher_policy == 'native' else CADTrackFuturePolicy
    model = policy(config['pretrained'], config['threshold']).to(args.device).eval().requires_grad_(False)
    model.abc.load_state_dict(payload['abc'], strict=True)
    if args.teacher_policy == 'future_actions':
        assert payload['policy_family'] == 'future_actions'
        model.future_action.load_state_dict(payload['future_action'], strict=True)
    else:
        assert 'future_action' not in payload
    model_family = ('CADTrackABC-native-current-quality' if args.teacher_policy == 'native'
                    else 'CADTrackFuturePolicy-future-actions')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    queries, records, started = [], [], time.perf_counter()
    for name in names[args.shard::args.shards]:
        sequence = Path(config['train_root']) / name
        rgb, tir = sorted((sequence / 'visible').iterdir()), sorted((sequence / 'infrared').iterdir())
        truth = np.loadtxt(sequence / 'init.txt', delimiter=',', dtype=np.float32).reshape(-1, 4)
        assert len(rgb) == len(tir) == len(truth) and (truth[0, 2:] > 0).all()
        count = min(16, len(rgb)) if args.sanity else len(rgb)
        valid = (truth[:count, 2:] > 0).all(-1)
        eligible = [frame for frame in range(1, count - args.horizon)
                    if valid[frame] and valid[frame + 1:frame + args.horizon + 1].any()]
        assert len(eligible) >= 2, name
        # Each video retains two uniform queries, including late parts of long
        # videos, independently of whether this teacher produces a failure.
        rng = random.Random(str(config['seed']) + ':' + name)
        reasons = {frame: {'uniform'} for frame in rng.sample(eligible, 2)}
        state = model.initialize([read_pair(rgb[0], tir[0])], torch.tensor(truth[:1]), torch.tensor([0]))
        previous_quality, previous_valid = 1., True
        first_events, event_counts = {}, dict.fromkeys(EVENT_TYPES, 0)
        failure_frames, template_writes, target_memory_writes = 0, 0, 0
        gate_counts = dict(absence_gate_frames=0, gain_gate_frames=0, triggered_frames=0,
                           failed_absence_gate_frames=0, failed_gain_gate_frames=0, failed_triggered_frames=0,
                           local_missing_frames=0, local_missing_triggered_frames=0)
        search_sum = np.zeros(3, dtype=np.float64)
        search_min = np.full(3, np.inf)
        search_max = np.full(3, -np.inf)
        for frame in range(1, count):
            observation, state = model.step([read_pair(rgb[frame], tir[frame])], torch.tensor([frame]), state)
            assert int(state['age'][0]) == frame and int(state['frames'][0, -1]) == frame
            template_commit = bool(observation['template_commit'].any())
            memory_commit = bool(observation['target_memory_commit'].any())
            template_writes += int(template_commit)
            target_memory_writes += int(memory_commit)
            cues = torch.stack((observation['search_logits'][0, 0].sigmoid(),
                                observation['search_logits'][0, 1].tanh(),
                                observation['local_scored']['quality'][0, 0].sigmoid())).cpu().numpy()
            absent = bool(cues[0] > .5 or cues[2] < .5)
            gain = bool(cues[1] > config['threshold'] + .01)
            triggered = absent and gain
            assert int(observation['extra_forwards']) == int(triggered)
            gate_counts['absence_gate_frames'] += int(absent)
            gate_counts['gain_gate_frames'] += int(gain)
            gate_counts['triggered_frames'] += int(triggered)
            search_sum += cues
            search_min = np.minimum(search_min, cues)
            search_max = np.maximum(search_max, cues)
            if valid[frame]:
                target = xyxy(torch.tensor(truth[frame:frame + 1], device=model.device))
                quality = float(iou(xyxy(observation['box']), target)[0])
                candidate_best = float(iou(observation['pair']['boxes'], target[:, None]).max())
                local_best = float(iou(observation['local']['boxes'], target[:, None]).max())
                failure_frames += int(quality < .2)
                if quality < .2:
                    gate_counts['failed_absence_gate_frames'] += int(absent)
                    gate_counts['failed_gain_gate_frames'] += int(gain)
                    gate_counts['failed_triggered_frames'] += int(triggered)
                gate_counts['local_missing_frames'] += int(local_best < .5)
                gate_counts['local_missing_triggered_frames'] += int(local_best < .5 and triggered)
                events = {'loss_transition': previous_valid and previous_quality >= .5 and quality < .2,
                          'wrong_identity_commit': quality < .2 and (template_commit or memory_commit),
                          'missed_correct_candidate': quality < .2 and candidate_best >= .5}
                for category, detected in events.items():
                    event_counts[category] += int(detected)
                    if detected and category not in first_events:
                        first_events[category] = frame
                previous_quality = quality
            previous_valid = bool(valid[frame])
        eligible = set(eligible)
        for category, event_frame in first_events.items():
            for offset in EVENT_OFFSETS:
                frame = event_frame + offset
                if frame in eligible:
                    reasons.setdefault(frame, set()).add(category + ':' + str(offset))
        selected = [{'sequence': name, 'frame': frame, 'sampling_reasons': sorted(reasons[frame])}
                    for frame in sorted(reasons)]
        queries.extend(selected)
        record = {'sequence': name, 'frames': count, 'queries': len(selected),
                  'uniform_queries': 2, 'first_events': first_events, 'event_counts': event_counts,
                  'failure_frames': failure_frames, 'template_write_frames': template_writes,
                  'target_memory_write_frames': target_memory_writes,
                  'search_gates': gate_counts,
                  'search_cues_order': ['predicted_absence_probability', 'predicted_extra_gain', 'local_keep_quality'],
                  'search_cues_mean': (search_sum / (count - 1)).tolist(),
                  'search_cues_min': search_min.tolist(), 'search_cues_max': search_max.tolist(),
                  'prefix_from_frame0': True, 'full_video': not args.sanity}
        records.append(record)
        print(json.dumps(record), flush=True)
    assert len({(query['sequence'], query['frame']) for query in queries}) == len(queries)
    query_path = output / ('queries_shard' + str(args.shard) + '.json')
    query_path.write_text(json.dumps(queries, indent=2))
    receipt = {'program_complete': True, 'full_shard_queries_complete': not args.sanity,
               'full_split_queries_complete': not args.sanity and args.shards == 1,
               'expected_full_split_sequences': 881 if args.split == 'train' else 98,
               'full_sequence_rollouts': not args.sanity, 'sanity': args.sanity,
               'teacher': str(Path(args.teacher).resolve()), 'teacher_epoch': payload['epoch'],
               'teacher_sha256': teacher_sha, 'model_family': model_family,
               'split': args.split, 'shard': args.shard, 'shards': args.shards,
               'sequences': len(records), 'queries': len(queries), 'query_file': str(query_path),
               'horizon': args.horizon, 'event_offsets': list(EVENT_OFFSETS), 'records': records,
               'seconds': time.perf_counter() - started, 'device': args.device,
               'cuda_initialized': torch.cuda.is_initialized(), 'official_test_used': False,
               'formal_metrics': None,
               'scope': 'Offline event/valid-label sampling from this actual teacher policy, not a benchmark score. Wrong commits are localization IoU proxies.'}
    (output / ('sampling_shard' + str(args.shard) + '.json')).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(receipt, records=None)), flush=True)


if __name__ == '__main__':
    main()
