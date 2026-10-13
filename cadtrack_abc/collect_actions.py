"""Native full-prefix counterfactual labels for future C/B training, TRAIN/DEV only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .counterfactual import (ACTION_MODES, advance_candidate, assert_same_state, clone_state,
                             decision_cache, label_rollout, observe_query, parent_action, rollout)
from .data import read_pair
from .model import CADTrackABC
from .future_actions import CADTrackFuturePolicy


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--teacher', required=True, help='Fixed modules checkpoint, not the live overwritten best file.')
    parser.add_argument('--teacher-policy', choices=['native', 'future_actions'], default='native')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--queries', required=True, help='JSON list of sequence/frame objects, zero-based frame.')
    parser.add_argument('--split', choices=['train', 'validation'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--modes', nargs='+', choices=ACTION_MODES, default=list(ACTION_MODES))
    args = parser.parse_args()
    assert args.horizon >= 1 and 'policy' in args.modes and len(set(args.modes)) == len(args.modes)
    assert args.teacher_policy == 'native' or set(args.modes) == set(ACTION_MODES)
    torch.set_num_threads(4)
    config = json.loads(Path(args.config).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    allowed = set(manifest['train_sequences' if args.split == 'train' else 'validation_sequences'])
    queries = json.loads(Path(args.queries).read_text())
    assert len(queries) and len({(q['sequence'], q['frame']) for q in queries}) == len(queries)
    assert {q['sequence'] for q in queries} <= allowed
    teacher_bytes = Path(args.teacher).read_bytes()
    teacher_sha = hashlib.sha256(teacher_bytes).hexdigest()
    del teacher_bytes
    payload = torch.load(args.teacher, map_location='cpu', weights_only=False)
    assert payload['config']['pretrained'] == config['pretrained']
    assert payload['config']['threshold'] == config['threshold']
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
    records, start = [], time.perf_counter()
    for name in sorted({q['sequence'] for q in queries}):
        directory = Path(config['train_root']) / name
        labels = np.loadtxt(directory / 'init.txt', delimiter=',', dtype=np.float32).reshape(-1, 4)
        rgb, tir = sorted((directory / 'visible').iterdir()), sorted((directory / 'infrared').iterdir())
        assert len(rgb) == len(tir) == len(labels) and (labels[0, 2:] > 0).all()
        state = model.initialize([read_pair(rgb[0], tir[0])], torch.tensor(labels[:1], device=model.device),
                                 torch.tensor([0], device=model.device))
        next_frame = 1
        for item in sorted((q for q in queries if q['sequence'] == name), key=lambda q: q['frame']):
            frame = item['frame']
            assert next_frame <= frame and frame + args.horizon < len(labels)
            assert (labels[frame, 2:] > 0).all() and (labels[frame + 1:frame + args.horizon + 1, 2:] > 0).all(-1).any()
            while next_frame < frame:
                _, state = model.step([read_pair(rgb[next_frame], tir[next_frame])],
                                      torch.tensor([next_frame], device=model.device), state)
                next_frame += 1
            prior = clone_state(state)
            images = [read_pair(rgb[frame], tir[frame])]
            times = torch.tensor([frame], device=model.device)
            query = observe_query(model, images, prior)
            native, native_state = model.step(images, times, clone_state(prior))
            reference = (parent_action(model, query) if args.teacher_policy == 'native' else
                         (int(native['extra_forwards'] > 0), int(native['choice'][0]), native['future_mode']))
            native_trace, reconstructed = advance_candidate(model, images, times, prior, query, *reference)
            assert torch.equal(native_trace['box'], native['box']) and int(native['choice'][0]) == reference[1]
            assert_same_state(native_state, reconstructed)
            future_images = [read_pair(rgb[f], tir[f]) for f in range(frame + 1, frame + args.horizon + 1)]
            future_frames = list(range(frame + 1, frame + args.horizon + 1))
            action_labels = []
            for context, candidate_set in enumerate(query['candidates']):
                for candidate in range(candidate_set['boxes'].shape[1]):
                    for mode in args.modes:
                        trace, advanced = advance_candidate(model, images, times, prior, query, context, candidate, mode)
                        traces = [trace] + rollout(model, advanced, future_images, future_frames)
                        labeled = label_rollout(traces, labels[frame:frame + args.horizon + 1])
                        assert torch.isfinite(labeled['utility'])
                        action_labels.append(dict(context=context, candidate=candidate, mode=mode, **labeled))
            reference_label = next(r for r in action_labels if (r['context'], r['candidate'], r['mode']) == reference)
            for label in action_labels:
                label['advantage'] = label['utility'] - reference_label['utility']
                label['harm'] = bool(label['advantage'] < -.05)
            filename = teacher_sha[:12] + '_' + name + '_' + str(frame).zfill(6) + '.pt'
            record = {'model_family': model_family, 'teacher': args.teacher,
                      'teacher_sha256': teacher_sha, 'teacher_epoch': payload['epoch'], 'sequence': name,
                      'frame': frame, 'split': args.split, 'horizon': args.horizon, 'modes': args.modes,
                      'query_spec': item,
                      'threshold': config['threshold'], 'reference_action': reference,
                      'native_state_reconstruction_equal': True, 'full_predicted_prefix_from_frame0': True}
            torch.save({'metadata': record, 'decision': decision_cache(query, prior), 'labels': action_labels}, output / filename)
            record.update(file=filename, action_labels=len(action_labels), seconds_since_start=time.perf_counter() - start)
            records.append(record)
            print(json.dumps(dict(event='query_complete', **record)), flush=True)
            # Keep the actual deployed teacher's query state, not an intervention.
            state = native_state
            next_frame = frame + 1
    receipt = {'completed': True, 'model_family': model_family, 'teacher_sha256': teacher_sha,
               'queries': len(records), 'records': records, 'device': args.device,
               'first_gt_only_initialization': True, 'future_gt_labels_only': True,
               'context_budget': 'local or local+one of three independent motion regions; no cross-region feature mixing',
               'utility': '.7 current_IoU + .3 valid_future_mean_IoU - .1 mean(wrong_appearance_fraction,wrong_geometry_fraction) - .01 mean_extra_forwards',
               'commit_proxy': 'Wrong means selected localization IoU<.2 on valid GT, not semantic identity truth.',
               'not_results': 'TRAIN/DEV teacher counterfactual labels; not formal metrics and not trained future-head results.'}
    (output / 'collection_complete.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
