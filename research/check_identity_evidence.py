"""Real-video zero-residual parity and observed-only identity evidence checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .collect_recoverability import InstanceExtractor
from .collect_rollouts import observe_actions
from .evaluate_online import read_pair
from .identity_evidence import IdentityEvidenceModules, IdentityEvidenceTracker
from .identity_representation_probe import PairedInstanceExtractor
from .recoverability_modules import RecoverabilityModules, select_actions
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import TemporalModules


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--projection', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--sequence-offset', type=int, required=True)
    p.add_argument('--weight', type=float, required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    c1 = torch.load('/data/gb/outputs/c1_initial_seed42/best.pth', map_location='cpu', weights_only=False)
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    assert checkpoint['module'] == 'ABC_candidate_relations'
    base = RecoverabilityModules(c1, candidate_relations=True).to(device)
    augmented = IdentityEvidenceModules(c1, candidate_relations=True).to(device)
    for module in (base, augmented):
        module.load_state_dict(checkpoint['model'], strict=True)
        module.eval().requires_grad_(False)
    augmented.configure_identity(args.projection, 0.)
    assert all(torch.equal(value, augmented.state_dict()[key]) for key, value in base.state_dict().items())
    motion_root = Path('/data/gb/outputs/abc_joint_v1_seed42')
    config = json.loads((motion_root / 'config.json').read_text())
    motion_checkpoint = torch.load(motion_root / 'last.pth', map_location='cpu', weights_only=False)
    motion = TemporalModules(c1, config['slots'], config['motion_history'], config['modes'], motion_checkpoint['horizon']).to(device)
    motion.load_state_dict(motion_checkpoint['model'], strict=True)
    motion.eval().requires_grad_(False)
    extractors = [kind('pretrained_models/gola_b224.bin', 5, .45, c1['args']['nms_iou']).to(device)
                  for kind in (InstanceExtractor, PairedInstanceExtractor)]
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
    reference = RecoverabilityTracker(extractors[0], base, motion, first_image, initial,
                                     checkpoint['args']['threshold'], write_verification='action', search_value='gross')
    control = IdentityEvidenceTracker(extractors[1], augmented, motion, first_image, initial,
                                     checkpoint['args']['threshold'], write_verification='action', search_value='gross')
    protected = control.encoded_identity_anchor.clone()
    assert torch.equal(reference.anchor_features, control.anchor_features)
    assert torch.equal(reference.motion_anchor, control.motion_anchor)
    for frame in range(1, len(visible)):
        image = read_pair(visible[frame], infrared[frame], device)
        original, unchanged = reference.step(image), control.step(image)
        assert np.array_equal(original, unchanged), (sequence.name, frame)
        for key, value in reference.last_decision.items():
            assert torch.equal(value, control.last_decision[key]), (sequence.name, frame, key)
        assert reference.stats == control.stats
        assert reference.frame == control.frame
        assert np.array_equal(reference.branch.search_box, control.branch.search_box)
        for key in ('template', 'mask', 'committed_template', 'committed_mask'):
            assert torch.equal(getattr(reference.branch, key), getattr(control.branch, key)), key
        for key in ('identifier', 'born', 'selected_streak'):
            assert getattr(reference.branch, key) == getattr(control.branch, key), key
        assert list(reference.branch.scores) == list(control.branch.scores)
        assert len(reference.branch.boxes) == len(control.branch.boxes)
        assert all(np.array_equal(a, b) for a, b in zip(reference.branch.boxes, control.branch.boxes))
        assert len(reference.branch.pending) == len(control.branch.pending)
        for before, after in zip(reference.branch.pending, control.branch.pending):
            assert before[0] == after[0] and torch.equal(before[1], after[1]) and torch.equal(before[2], after[2])
        assert torch.equal(reference.identity_memory, control.identity_memory)
        assert torch.equal(reference.motion_memory, control.motion_memory)
        for key in ('anchor', 'anchor_mask', 'anchor_features', 'identity_anchor', 'motion_anchor'):
            assert torch.equal(getattr(reference, key), getattr(control, key)), key
        assert torch.equal(protected, control.encoded_identity_anchor)
        assert len(reference.history) == len(control.history)
        assert all(np.array_equal(a[0], b[0]) and a[1:] == b[1:] for a, b in zip(reference.history, control.history))
    augmented.identity_weight = args.weight
    image = read_pair(visible[-1], infrared[-1], device)
    observation = observe_actions([(control, control.branch, image)], extractors[1], augmented.c1)[0]
    history = control.history_inputs()
    distribution = motion.motion(history['history_boxes'], history['history_frames'], history['history_valid'],
                                 history['history_quality'], control.motion_anchor, control.motion_memory)
    data = control.original_inputs(observation, distribution, history)
    expected = augmented.decide(data, control.identity_anchor, control.identity_memory)
    parent = base.decide(data, control.identity_anchor, control.identity_memory)
    assert torch.equal(expected['scores'], parent['scores'] + expected['semantic_identity_residual'][..., None])
    for key in ('quality_logits', 'region_advantage', 'region_success_logits', 'absence_logit', 'target_probability'):
        assert torch.equal(expected[key], parent[key]), key
    for region in (0, 1):
        if region:
            extra, _ = control.extra_observation(region, distribution, image)
            assert extra is not None
            control.insert_region(data, region, extra)
            expected = augmented.decide(data, control.identity_anchor, control.identity_memory)
        changed = {key: value.clone() for key, value in data.items()}
        changed['encoded_instance_features'][~changed['valid']] = 1000
        changed['gt'] = torch.tensor([-999.], device=device)
        actual = augmented.decide(changed, control.identity_anchor, control.identity_memory)
        for key in ('scores', 'semantic_identity_support', 'semantic_identity_residual'):
            assert torch.equal(expected[key][data['valid']], actual[key][data['valid']]), (region, key)
        for key in ('region_advantage', 'region_success_logits', 'absence_logit'):
            assert torch.equal(expected[key], actual[key]), (region, key)
        before = select_actions(expected, data, checkpoint['args']['threshold'], 'action', 'gross')
        after = select_actions(actual, changed, checkpoint['args']['threshold'], 'action', 'gross')
        assert all(torch.equal(value, after[key]) for key, value in before.items())
    assert all(p.grad is None for group in (*extractors, base, augmented, motion) for p in group.parameters())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(completed=True, sequence=sequence.name, frames=len(visible), identity_weight_checked=args.weight,
                  zero_residual_boxes_and_all_parent_decisions_exact=True, zero_residual_all_parent_states_exact=True,
                  original_pre_attention_motion_and_memory_preserved=True, protected_self_context_anchor_unchanged=True,
                  original_quality_search_and_write_evidence_exact=True, unobserved_encoded_regions_and_GT_do_not_change_actions=True,
                  trained_projection_loaded=True, frozen_gradients_absent=True, new_optimizer_updates=0,
                  native_accuracy=False, peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    (output / 'completion.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
