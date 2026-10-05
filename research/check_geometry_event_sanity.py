"""Real-video fixed-head event continuation must match a private old4 rollout."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .collect_recoverability import InstanceExtractor
from .evaluate_online import read_pair
from .geometry_commit import GeometryCommitHead
from .geometry_commit_event import GeometryCommitEventTracker
from .probe_geometry_commit import private_state
from .recoverability_modules import RecoverabilityModules
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import TemporalModules


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', required=True)
    p.add_argument('--commit-model', required=True)
    p.add_argument('--sequence-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--pretrained', default='/data/gb/GOLA/pretrained_models/gola_b224.bin')
    p.add_argument('--c1-head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--motion-run', default='/data/gb/outputs/abc_joint_v1_seed42')
    a = p.parse_args()
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    c1 = torch.load(a.c1_head, map_location='cpu', weights_only=False)
    parent = torch.load(a.parent, map_location='cpu', weights_only=False)
    commit = torch.load(a.commit_model, map_location='cpu', weights_only=False)
    assert parent['epoch'] == 4 and commit['epoch'] == 41 and commit['parent_model'] == a.parent
    assert commit['args']['horizon'] == 3
    modules = RecoverabilityModules(c1).to(device)
    modules.load_state_dict(parent['model'], strict=True)
    modules.eval().requires_grad_(False)
    config = json.loads((Path(a.motion_run)/'config.json').read_text())
    saved = torch.load(Path(a.motion_run)/'last.pth', map_location='cpu', weights_only=False)
    motion = TemporalModules(c1, config['slots'], config['motion_history'], config['modes'], saved['horizon']).to(device)
    motion.load_state_dict(saved['model'], strict=True)
    motion.eval().requires_grad_(False)
    head = GeometryCommitHead().to(device)
    head.load_state_dict(commit['head'], strict=True)
    extractor = InstanceExtractor(a.pretrained, c1['args']['candidates'], .45, c1['args']['nms_iou']).to(device)
    root = Path(a.sequence_root)
    visible, infrared = sorted((root/'visible').iterdir()), sorted((root/'infrared').iterdir())
    assert len(visible) == len(infrared) and len(visible) > 120
    with (root/'init.txt').open() as labels:
        initial = np.fromstring(labels.readline().strip(), sep=',')
    initial[2:] += initial[:2]
    teacher = None
    matched_frames, completed_events = 0, 0
    with torch.inference_mode():
        tracker = GeometryCommitEventTracker(extractor, modules, motion, read_pair(visible[0], infrared[0], device),
                  initial, parent['args']['threshold'], write_verification='action', commit_head=head, continuation_horizon=3)
        for frame in range(1, 120):
            image = read_pair(visible[frame], infrared[frame], device)
            box = tracker.step(image)
            if teacher is not None:
                expected = teacher.step(image)
                assert bool(tracker.last_decision['old4_continuation'])
                assert np.array_equal(box, expected) and np.array_equal(tracker.branch.search_box, teacher.branch.search_box)
                assert torch.equal(tracker.branch.template, teacher.branch.template)
                assert torch.equal(tracker.identity_memory, teacher.identity_memory) and torch.equal(tracker.motion_memory, teacher.motion_memory)
                assert all(np.array_equal(x[0], y[0]) and x[1:] == y[1:] for x, y in zip(tracker.history, teacher.history))
                matched_frames += 1
                if frame == tracker.old4_until_frame:
                    completed_events += 1
                    teacher = None
            elif tracker.old4_until_frame > frame:
                state = private_state(tracker)
                teacher = object.__new__(RecoverabilityTracker)
                teacher.__dict__.update(state.__dict__)
        assert completed_events > 0 and matched_frames >= completed_events * 3
        assert matched_frames == tracker.stats['old4_continuation_frames']
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    result = {'complete': True, 'sequence': root.name, 'frames': 120, 'parent_epoch': 4, 'commit_epoch': 41,
              'continuation_horizon': 3, 'complete_event_windows': completed_events, 'matched_old4_frames': matched_frames,
              'stats': tracker.stats, 'real_old4_continuation_output_search_template_identity_motion_history_exact': True,
              'future_GT_input': False, 'scope': 'Real NN event continuation behavior, not an accuracy claim; only first GT initializes.'}
    (out/'completion.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'COMPLETE').write_text('PASS\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
