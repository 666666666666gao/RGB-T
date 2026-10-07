"""Real-video parent parity and reference independence before full controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .collect_recoverability import InstanceExtractor
from .evaluate_online import read_pair, track_sequence
from .protected_visual_reference import ProtectedVisualReferenceTracker, MODES
from .recoverability_modules import RecoverabilityModules
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import TemporalModules


def equal_states(before, after):
    assert before.stats == after.stats
    assert before.frame == after.frame
    assert before.template_source_frame == after.template_source_frame
    assert np.array_equal(before.template_source_box, after.template_source_box)
    for key in ('identifier', 'born', 'selected_streak'):
        assert getattr(before.branch, key) == getattr(after.branch, key), key
    for key in ('box', 'search_box'):
        assert np.array_equal(getattr(before.branch, key), getattr(after.branch, key)), key
    for key in ('template', 'mask', 'committed_template', 'committed_mask'):
        assert torch.equal(getattr(before.branch, key), getattr(after.branch, key)), key
    assert list(before.branch.scores) == list(after.branch.scores)
    assert len(before.branch.boxes) == len(after.branch.boxes)
    assert all(np.array_equal(a, b) for a, b in zip(before.branch.boxes, after.branch.boxes))
    assert len(before.branch.pending) == len(after.branch.pending)
    for a, b in zip(before.branch.pending, after.branch.pending):
        assert a[0] == b[0] and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    for key in ('anchor', 'anchor_mask', 'anchor_features', 'identity_anchor', 'motion_anchor',
                'identity_memory', 'motion_memory'):
        assert torch.equal(getattr(before, key), getattr(after, key)), key
    assert len(before.history) == len(after.history)
    assert all(np.array_equal(a[0], b[0]) and a[1:] == b[1:] for a, b in zip(before.history, after.history))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--sequence-offset', type=int, required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.set_num_threads(4)
    device = torch.device('cuda:0')
    c1 = torch.load('/data/gb/outputs/c1_initial_seed42/best.pth', map_location='cpu', weights_only=False)
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    assert checkpoint['module'] == 'ABC_candidate_relations'
    modules = RecoverabilityModules(c1, candidate_relations=True).to(device)
    modules.load_state_dict(checkpoint['model'], strict=True)
    modules.eval().requires_grad_(False)
    motion_root = Path('/data/gb/outputs/abc_joint_v1_seed42')
    config = json.loads((motion_root / 'config.json').read_text())
    motion_checkpoint = torch.load(motion_root / 'last.pth', map_location='cpu', weights_only=False)
    motion = TemporalModules(c1, config['slots'], config['motion_history'], config['modes'], motion_checkpoint['horizon']).to(device)
    motion.load_state_dict(motion_checkpoint['model'], strict=True)
    motion.eval().requires_grad_(False)
    extractor = InstanceExtractor('pretrained_models/gola_b224.bin', 5, .45, c1['args']['nms_iou']).to(device)
    split = json.loads(Path(args.split).read_text())
    assert not set(split['train']) & set(split['validation'])
    sequences = sorted(path for path in Path(args.root).iterdir() if path.name in split['validation'])
    assert len(sequences) == 98
    sequence = sequences[args.sequence_offset]
    visible = sorted(path for path in (sequence / 'visible').iterdir() if path.is_file())[:64]
    infrared = sorted(path for path in (sequence / 'infrared').iterdir() if path.is_file())[:64]
    with (sequence / 'init.txt').open() as stream:
        initial = np.fromstring(stream.readline().strip(), sep=',')
    initial[2:] += initial[:2]
    first_image = read_pair(visible[0], infrared[0], device)
    control = ProtectedVisualReferenceTracker(extractor, modules, motion, first_image, initial,
        checkpoint['args']['threshold'], write_verification='action', search_value='gross', reference_mode=args.mode)
    reference = RecoverabilityTracker(extractor, modules, motion, first_image, initial,
        checkpoint['args']['threshold'], write_verification='action', search_value='gross') if args.mode == 'parent' else None
    protected = [initial.copy()]
    forced_private_state_frame = None
    for frame in range(1, len(visible)):
        if args.mode in ('visual', 'visual_motion') and frame == 4:
            # Actual witnessed failure path: untrusted output/search/template
            # state can differ. It must not alter the protected primary source.
            control.branch.search_box += 100.
            control.branch.template = torch.zeros_like(control.branch.template)
            forced_private_state_frame = frame
        image = read_pair(visible[frame], infrared[frame], device)
        previous_motion = control.motion_memory.clone()
        actual = control.step(image)
        assert int(control.last_decision['valid'].sum()) <= 10
        if reference is not None:
            expected = reference.step(image)
            assert np.array_equal(actual, expected), (sequence.name, frame)
            assert all(torch.equal(value, control.last_decision[key]) for key, value in reference.last_decision.items())
            equal_states(reference, control)
        protected.append(control.reference_branch.box.copy())
        if args.mode == 'visual_motion':
            assert np.array_equal(control.history[-1][0], control.reference_branch.box)
            assert control.history[-1][1] == frame
            candidates, quality, _, _ = control.reference_observation
            point = motion.memory.encode(candidates['modality_features'][:, 0].half().float())
            expected_memory = motion.memory.update(previous_motion, point, candidates['evidence'][:, 0].float(),
                quality[0][None], torch.tensor([float(candidates['raw_score'][0, 0]) > .84], device=device))
            assert torch.equal(expected_memory, control.motion_memory)
    native_exact = None
    if args.mode in ('visual', 'visual_motion'):
        native, _, _, _ = track_sequence(visible, infrared, initial, extractor, None, device, torch.float16)
        protected = np.asarray(protected)
        protected[:, 2:] -= protected[:, :2]
        assert np.array_equal(protected, native), (sequence.name, np.abs(protected-native).max())
        native_exact = True
    assert all(p.grad is None for group in (extractor, modules, motion) for p in group.parameters())
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=True, sequence=sequence.name, frames=len(visible), reference_mode=args.mode,
        all_parent_decisions_and_states_exact=args.mode == 'parent', protected_native_reference_bitwise_exact=native_exact,
        private_bad_state_injected_only_in_sanity=forced_private_state_frame,
        protected_motion_history_and_memory_exact=args.mode == 'visual_motion',
        original_A_B_C_loaded_and_frozen=True, optimizer_updates=0, native_accuracy=False,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    (out / 'completion.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
