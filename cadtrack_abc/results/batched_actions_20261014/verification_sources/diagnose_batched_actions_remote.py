"""Named native numerical diagnostics and traced H3 reference/batch replay."""
import argparse
from pathlib import Path
import time
import json
import hashlib

import numpy as np
import torch

from cadtrack_abc.batched_actions import advance_actions, repeat_state, unpack_traces
from cadtrack_abc.counterfactual import ACTION_MODES, STATE_FIELDS, advance_candidate, clone_state, observe_query, label_rollout
from cadtrack_abc.data import read_pair, crop_batch, clip_boxes
from cadtrack_abc.model import CADTrackABC
from cadtrack_abc.ported_modules import geometry


class PresetSearch(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, inputs):
        assert len(inputs) == len(self.logits)
        return inputs.new_tensor(self.logits)


def fields(state, row):
    return {key: [value[row:row + 1] for value in state[key]] if key == 'templates'
            else state[key][row:row + 1] for key in STATE_FIELDS}


def state_errors(first, second):
    errors = {}
    for key in STATE_FIELDS:
        left = first[key] if key == 'templates' else [first[key]]
        right = second[key] if key == 'templates' else [second[key]]
        if left[0].is_floating_point():
            errors[key] = max(float((a - b).abs().max()) for a, b in zip(left, right))
        else:
            assert all(torch.equal(a, b) for a, b in zip(left, right)), key
            errors[key] = 0.
    return errors


@torch.inference_mode()
def mixed_diagnostic(model, image, next_image, frame, prior):
    original = model.abc.search
    outputs = {}
    for name, logits in [('mixed', [[-3., -1.], [3., 1.]]), ('off', [[-3., -1.], [-3., -1.]]),
                         ('on', [[3., 1.], [3., 1.]])]:
        model.abc.search = PresetSearch(logits)
        outputs[name] = model.step([image, image], torch.tensor([frame, frame]), repeat_state(prior, 2))
    result = {'frame': frame, 'fixed_batch2': [], 'singleton': [], 'original_1e5_test_reclassified': False}
    for row, (name, logits) in enumerate([('off', [-3., -1.]), ('on', [3., 1.])]):
        mixed_obs, mixed_state = outputs['mixed']
        control_obs, control_state = outputs[name]
        for key in ['choice', 'write', 'template_commit', 'target_memory_commit', 'geometry_commit']:
            assert torch.equal(mixed_obs[key][row:row + 1], control_obs[key][row:row + 1]), (row, key)
        error = state_errors(fields(mixed_state, row), fields(control_state, row))
        assert all(value == 0. for value in error.values()), (row, error)
        result['fixed_batch2'].append({'row': row, 'control': name, 'state_errors': error, 'decisions_exact': True})
        model.abc.search = PresetSearch([logits])
        single_obs, single_state = model.step([image], torch.tensor([frame]), clone_state(prior))
        for key in ['choice', 'write', 'template_commit', 'target_memory_commit', 'geometry_commit']:
            assert torch.equal(mixed_obs[key][row:row + 1], single_obs[key]), (row, key)
        error = state_errors(fields(mixed_state, row), single_state)
        a, at = crop_batch([next_image], mixed_state['box'][row:row + 1], 4., 256, model.device)
        b, bt = crop_batch([next_image], single_state['box'], 4., 256, model.device)
        result['singleton'].append({'row': row, 'state_errors': error,
                                    'mixed_box': mixed_state['box'][row].cpu().tolist(),
                                    'single_box': single_state['box'][0].cpu().tolist(),
                                    'next_crop_transform_equal': bool(torch.equal(at, bt)),
                                    'next_crop_pixels_equal': bool(torch.equal(a, b)), 'decisions_exact': True})
    model.abc.search = original
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert args.batch in [1, 64]
    torch.set_num_threads(4)
    root = Path('/home/gaob/CADTrack-ABC')
    parent = root / 'outputs/cad_abc_20261013'
    config = json.loads((root / 'cadtrack_abc/config.json').read_text())
    inventory = json.loads((parent / 'manifest.json').read_text())
    payload = torch.load(parent / 'best.pth', map_location='cpu', weights_only=False)
    model = CADTrackABC(config['pretrained'], config['threshold']).to('cuda').eval().requires_grad_(False)
    model.abc.load_state_dict(payload['abc'], strict=True)
    name = sorted(inventory['train_sequences'])[0]
    sequence = Path(config['train_root']) / name
    rgb, tir = sorted((sequence / 'visible').iterdir()), sorted((sequence / 'infrared').iterdir())
    truth = np.loadtxt(sequence / 'init.txt', delimiter=',', dtype=np.float32).reshape(-1, 4)
    images = [read_pair(rgb[frame], tir[frame]) for frame in range(9)]
    state = model.initialize([images[0]], torch.tensor(truth[:1]), torch.tensor([0]))
    packets, mixed, initial_checks, durations = [], [], [], []
    for frame in range(1, 6):
        image = images[frame]
        if frame in [4, 5]:
            prior = clone_state(state)
            query = observe_query(model, [image], prior)
            actions = [(context, candidate, mode) for context, candidates in enumerate(query['candidates'])
                       for candidate in range(candidates['boxes'].shape[1]) for mode in ACTION_MODES]
            assert len(actions) == 210
            if args.batch == 64:
                mixed.append(mixed_diagnostic(model, image, images[frame + 1], frame, prior))
            labels, captures = [], []
            started = time.perf_counter()
            for begin in range(0, len(actions), args.batch):
                chunk = actions[begin:begin + args.batch]
                if args.batch == 1:
                    trace, advanced = advance_candidate(model, [image], torch.tensor([frame], device=model.device), prior, query, *chunk[0])
                    trace['extra_forwards'] = torch.tensor([trace['extra_forwards']], device=model.device)
                else:
                    trace, advanced = advance_actions(model, image, frame, prior, query, chunk)
                    maximum = dict.fromkeys(STATE_FIELDS, 0.)
                    for row, action in enumerate(chunk):
                        _, expected = advance_candidate(model, [image], torch.tensor([frame], device=model.device), prior, query, *action)
                        actual = fields(advanced, row)
                        error = state_errors(actual, expected)
                        for key in STATE_FIELDS:
                            maximum[key] = max(maximum[key], error[key])
                            left = actual[key] if key == 'templates' else [actual[key]]
                            right = expected[key] if key == 'templates' else [expected[key]]
                            for a, b in zip(left, right):
                                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
                    initial_checks.append({'frame': frame, 'begin': begin, 'count': len(chunk), 'max_error': maximum})
                traces, captured = [trace], []
                for future_frame in range(frame + 1, frame + 4):
                    before = advanced['box'].clone()
                    future_image = images[future_frame]
                    observation, advanced = model.step([future_image] * len(chunk),
                                                       torch.full((len(chunk),), future_frame, device=model.device), advanced)
                    logits = observation['search_logits']
                    extra = (((logits[:, 0].sigmoid() > .5) | (observation['local_scored']['quality'][:, 0].sigmoid() < .5))
                             & (logits[:, 1].tanh() > model.threshold + .01))
                    distribution = observation['distribution']
                    target = distribution['means'][torch.arange(len(chunk), device=model.device), distribution['log_weights'].argmax(-1), 0]
                    center, size = geometry(distribution['reference'])
                    center = center + target[:, :2] * size
                    size = size * target[:, 2:].clamp(-1.3863, 1.3863).exp()
                    extra_box = clip_boxes(torch.cat((center - size / 2, center + size / 2), -1), [future_image] * len(chunk))
                    future_trace = {key: observation[key] for key in ['box', 'template_commit', 'target_memory_commit', 'geometry_commit']}
                    future_trace['extra_forwards'] = extra.long()
                    traces.append(future_trace)
                    captured.append({'frame': future_frame, 'local_crop_box': before.cpu(), 'extra_crop_box': extra_box.cpu(),
                                     'choice': observation['choice'].cpu(), 'write': observation['write'].cpu(),
                                     **{key: value.cpu() for key, value in future_trace.items()}})
                for row, (action, single) in enumerate(zip(chunk, unpack_traces(traces, len(chunk)))):
                    labels.append(dict(context=action[0], candidate=action[1], mode=action[2], **label_rollout(single, truth[frame:frame + 4])))
                    captures.append([{key: value if key == 'frame' else value[row:row + 1]
                                      for key, value in capture.items()} for capture in captured])
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            assert len(labels) == len(captures) == 210 and all(len(capture) == 3 for capture in captures)
            packets.append({'frame': frame, 'actions': actions, 'labels': labels, 'future_traces': captures})
        _, state = model.step([image], torch.tensor([frame], device=model.device), state)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    torch.save({'packets': packets, 'teacher_epoch': payload['epoch'], 'teacher_sha256': hashlib.sha256((parent / 'best.pth').read_bytes()).hexdigest(),
                'batch': args.batch}, output / 'traced_replay.pt')
    receipt = {'status': 'BOUNDED_TRACED_REPLAY_COMPLETE_NOT_YET_COMPARISON', 'batch': args.batch, 'queries': 2, 'actions': 420,
               'sequence': name, 'split': 'train', 'horizon': 3, 'measured_rollout_seconds': durations,
               'mixed_diagnostics': mixed, 'initial_commit_checks': initial_checks, 'updates': 0,
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
               'original_strict_mixed_failure_preserved': True, 'baseline_rerun': False,
               'scope': 'Same2 actual TRAIN queries. Native singleton floating drift is recorded without reclassifying original tolerance failure.'}
    (output / 'replay_complete.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


main()
