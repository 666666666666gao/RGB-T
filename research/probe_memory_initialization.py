"""TRAIN same-prestate audit: persistent memory versus anchor plus last8 writes."""
import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from .collect_recoverability import InstanceExtractor
from .collect_rollouts import iou
from .collect_temporal import paths
from .evaluate_online import read_pair
from .geometry_commit import selected_index
from .recoverability_modules import RecoverabilityModules, select_actions
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import TemporalModules
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


class MemoryAuditTracker(RecoverabilityTracker):
    def __init__(self, *args, queries, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries = set(queries)
        self.recent = deque(maxlen=8)
        self.audits = []

    def original_inputs(self, observation, distribution, history):
        self.audit_data = super().original_inputs(observation, distribution, history)
        return self.audit_data

    def replay(self, initial):
        memory = initial.clone()
        for descriptor, evidence, quality, write, _ in self.recent:
            gate = self.modules.memory.gate_logits(descriptor, evidence, quality, self.identity_anchor)
            probability = gate.softmax(-1)
            observed = torch.ones(1, dtype=torch.bool, device=self.device)
            verified = probability[..., 0] >= .5
            commit = verified & write
            memory = self.modules.memory.update(memory, descriptor, probability, observed, commit)
        return memory

    def accept_observation(self, image, observation, pause):
        candidates, quality, boxes, choice = observation
        if self.frame in self.queries:
            assert self.recent
            persistent = self.identity_memory.clone()
            faithful = self.replay(self.recent[0][4])
            assert torch.allclose(faithful, persistent, atol=1e-6, rtol=1e-6)
            reset = self.replay(self.modules.memory.initialize(self.identity_anchor))
            full_output = self.modules.decide(self.audit_data, self.identity_anchor, persistent)
            reset_output = self.modules.decide(self.audit_data, self.identity_anchor, reset)
            full_action = select_actions(full_output, self.audit_data, self.threshold, self.write_verification)
            reset_action = select_actions(reset_output, self.audit_data, self.threshold, self.write_verification)
            matched = int(selected_index(self.audit_data, boxes[choice])[0])
            actual_action = matched * 2 + int(pause)
            assert int(full_action['flat_action'][0]) == actual_action
            alternate = int(reset_action['flat_action'][0])
            flat_boxes = self.audit_data['image_boxes'][0].reshape(35, 4)
            reset_box = (boxes[choice].copy() if alternate // 2 == matched else
                         flat_boxes[alternate // 2].cpu().numpy().copy())
            score_difference = (full_output['scores']-reset_output['scores']).abs()
            valid_scores = self.audit_data['valid'][..., None].expand_as(score_difference)
            self.audits.append({'frame': self.frame, 'history_count': len(self.recent),
                'persistent_vs_reset_l2': float((persistent-reset).norm()),
                'faithful_replay_max_abs_error': float((faithful-persistent).abs().max()),
                'scores_max_abs_difference': float(score_difference[valid_scores].max()),
                'full_action': actual_action, 'reset_action': alternate,
                'search_trigger_changed': bool(full_action['search_triggered'][0] != reset_action['search_triggered'][0]),
                'searched_region_full': int(full_action['searched_region'][0]),
                'searched_region_reset': int(reset_action['searched_region'][0]),
                'actual_box_xyxy': boxes[choice].copy().tolist(), 'reset_box_xyxy': reset_box.tolist(),
                'same_actual_candidate_pool': True})
        before = self.identity_memory.clone()
        result = super().accept_observation(image, observation, pause)
        descriptor = self.modules.memory.encode(candidates['instance_features'][:, choice].half().float())
        self.recent.append((descriptor.clone(), candidates['evidence'][:, choice].float().clone(),
                            quality[choice][None].float().clone(), bool(result[1]), before))
        return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs', required=True); p.add_argument('--output', required=True)
    p.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    p.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    p.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json')
    p.add_argument('--parent', default='/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth')
    p.add_argument('--c1-head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--motion-run', default='/data/gb/outputs/abc_joint_v1_seed42')
    p.add_argument('--pretrained', default='/data/gb/GOLA/pretrained_models/gola_b224.bin')
    a = p.parse_args(); torch.set_num_threads(4); torch.manual_seed(42)
    jobs = json.loads(Path(a.jobs).read_text())['jobs']
    split = json.loads(Path(a.split).read_text())
    assert not set(split['train']) & set(split['validation'])
    assert jobs and len({(j['sequence'], j['query_frame']) for j in jobs}) == len(jobs)
    assert {j['sequence'] for j in jobs} <= set(split['train'])
    grouped = {}
    for job in jobs: grouped.setdefault(job['sequence'], []).append(job)
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(a.root, a.cache)
    sequences = {s.get_name(): s for s in dataset}
    device = torch.device('cuda:0')
    c1 = torch.load(a.c1_head, map_location='cpu', weights_only=False)
    parent = torch.load(a.parent, map_location='cpu', weights_only=False)
    assert parent['epoch'] == 4
    modules = RecoverabilityModules(c1).to(device)
    modules.load_state_dict(parent['model'], strict=True); modules.eval().requires_grad_(False)
    cfg = json.loads((Path(a.motion_run)/'config.json').read_text())
    motion_saved = torch.load(Path(a.motion_run)/'last.pth', map_location='cpu', weights_only=False)
    motion = TemporalModules(c1, cfg['slots'], cfg['motion_history'], cfg['modes'], motion_saved['horizon']).to(device)
    motion.load_state_dict(motion_saved['model'], strict=True); motion.eval().requires_grad_(False)
    extractor = InstanceExtractor(a.pretrained, c1['args']['candidates'], .45, c1['args']['nms_iou']).to(device)
    out = Path(a.output); assert not out.exists(); out.mkdir(parents=True)
    results = []
    for name, sequence_jobs in sorted(grouped.items()):
        sequence = sequences[name]; queries = sorted(j['query_frame'] for j in sequence_jobs)
        assert min(queries) > 0 and max(queries) < len(sequence)
        initial = sequence[0].get_bounding_box().copy()
        tracker = MemoryAuditTracker(extractor, modules, motion, read_pair(*paths(sequence, 0), device), initial,
                                    parent['args']['threshold'], write_verification='action', queries=queries)
        for frame in range(1, max(queries)+1): tracker.step(read_pair(*paths(sequence, frame), device))
        assert [r['frame'] for r in tracker.audits] == queries
        for row in tracker.audits:
            target = sequence[row['frame']].get_bounding_box()
            job = next(j for j in sequence_jobs if j['query_frame'] == row['frame'])
            row.update(sequence=name, event_id=job['event_id'], cached_event_role=job['cached_event_role'],
                       actual_iou=iou(np.array(row['actual_box_xyxy']), target),
                       reset_iou=iou(np.array(row['reset_box_xyxy']), target))
            results.append(row)
        (out/'progress.json').write_text(json.dumps({'sequences':name,'completed_queries':len(results),'queries':len(jobs)}))
        print(json.dumps({'sequence':name,'completed_queries':len(results),'queries':len(jobs)}), flush=True)
    result = {'complete': True, 'queries':len(jobs), 'sequences':len(grouped), 'records':results,
              'weights_fixed_parent_epoch':4, 'model_updates':0, 'NN_decision_GT_input':'first initialization only',
              'scope':'Same-prestate TRAIN memory intervention. Reset8 uses the same live encoded last8 observations but an anchor-only start; faithful replay restores the actual prefix state. No changed search image, no future rollout, no native accuracy.'}
    (out/'memory_initialization_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'COMPLETE').write_text('Matched memory audit complete\n')


if __name__ == '__main__': main()
