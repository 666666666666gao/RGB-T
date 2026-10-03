"""Strictly online ABC recovery with bounded branch-private learned state."""
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import torch

from .bounded_recovery import Branch, BoundedRecoveryTracker


@dataclass(kw_only=True)
class ABCBranch(Branch):
    template_source_frame: int
    template_source_box: np.ndarray
    committed_source_frame: int
    committed_source_box: np.ndarray
    memory: torch.Tensor
    committed_memory: torch.Tensor
    memory_pending: deque
    motion_boxes: deque
    motion_frames: deque
    motion_quality: deque

    def copy(self, identifier=None, born=None):
        return replace(self, identifier=self.identifier if identifier is None else identifier,
                       born=self.born if born is None else born, box=self.box.copy(),
                       search_box=self.search_box.copy(), pending=deque(self.pending, maxlen=self.pending.maxlen),
                       scores=deque(self.scores, maxlen=self.scores.maxlen),
                       boxes=deque(self.boxes, maxlen=self.boxes.maxlen),
                       template_source_box=self.template_source_box.copy(),
                       committed_source_box=self.committed_source_box.copy(),
                       memory=self.memory.clone(), committed_memory=self.committed_memory.clone(),
                       memory_pending=deque(self.memory_pending, maxlen=self.memory_pending.maxlen),
                       motion_boxes=deque((b.copy() for b in self.motion_boxes), maxlen=self.motion_boxes.maxlen),
                       motion_frames=deque(self.motion_frames, maxlen=self.motion_frames.maxlen),
                       motion_quality=deque(self.motion_quality, maxlen=self.motion_quality.maxlen))


class ABCTracker(BoundedRecoveryTracker):
    def __init__(self, extractor, modules, image, init_box, amp_dtype,
                 branches=3, window=5, switch_patience=2, restore_template_state=True):
        super().__init__(extractor, modules.c1, image, init_box, amp_dtype,
                         branches, window, switch_patience, restore_template_state)
        self.modules = modules.eval().requires_grad_(False)
        # Same separate pre-attention identity descriptors as the collector.
        anchors = []
        with torch.autocast('cuda', dtype=amp_dtype):
            weights = self.anchor_mask.flatten(1).float()
            for channel in (slice(0, 3), slice(3, 6)):
                tokens = extractor.base.patch_embed(self.anchor[None, channel])
                anchors.append((tokens * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp(min=1))
        self.identity_anchor = modules.memory.encode(torch.stack(anchors, 1).float())
        memory = modules.memory.initialize(self.identity_anchor)[0]
        initial = self.branches[0]
        n = modules.motion.history
        self.branches = [ABCBranch(**vars(initial), template_source_frame=0,
                                   template_source_box=init_box.copy(), committed_source_frame=0,
                                   committed_source_box=init_box.copy(),
                                   memory=memory, committed_memory=memory.clone(),
                                   memory_pending=deque(maxlen=window),
                                   motion_boxes=deque([init_box.copy()], maxlen=n),
                                   motion_frames=deque([0], maxlen=n), motion_quality=deque([1.], maxlen=n))]
        self.decisions = {}

    def _observe_many(self, branches, image):
        original = super()._observe_many(branches, image)
        candidates = {key: torch.cat([row[0][key] for row in original]) for key in original[0][0]}
        candidates['image_boxes'] = torch.from_numpy(np.stack([row[2] for row in original])).to(self.device).float()
        candidates['c1_quality'] = torch.stack([row[1] for row in original])
        boxes, frames, quality, valid = [], [], [], []
        n = self.modules.motion.history
        for branch in branches:
            size = len(branch.motion_boxes)
            boxes.append(np.stack([branch.motion_boxes[0]] * (n - size) + list(branch.motion_boxes)))
            frames.append([0] * (n - size) + list(branch.motion_frames))
            quality.append([0.] * (n - size) + list(branch.motion_quality))
            valid.append([False] * (n - size) + [True] * size)
        output = self.modules.decide(candidates, self.identity_anchor.expand(len(branches), -1, -1),
                                     torch.stack([b.memory for b in branches]),
                                     torch.tensor(np.stack(boxes), device=self.device, dtype=torch.float32),
                                     torch.tensor(frames, device=self.device),
                                     torch.tensor(valid, device=self.device),
                                     torch.tensor(quality, device=self.device, dtype=torch.float32))
        choices = output['scores'].masked_fill(~candidates['valid'], -torch.inf).argmax(1).tolist()
        c1_choices = output['c1_scores'].masked_fill(~candidates['valid'], -torch.inf).argmax(1)
        self.candidate_sets = [{'parent_id': branch.identifier, 'boxes_xyxy': original[index][2].copy(),
                                'valid': candidates['valid'][index].detach(),
                                'choice': choices[index], 'c1_choice': c1_choices[index]}
                               for index, branch in enumerate(branches)]
        result = []
        for index, (_, _, image_boxes, _) in enumerate(original):
            row = {key: value[index:index + 1] for key, value in candidates.items()}
            row['current_quality'] = output['current_logits'][index:index + 1].sigmoid()
            row['future_quality'] = output['future_logits'][index:index + 1].sigmoid()
            row['predicted_cost'] = output['cost'][index:index + 1]
            row['identity_support'] = output['identity_support'][index:index + 1]
            row['forecast_means'] = output['distribution']['means'][index:index + 1]
            row['forecast_deviations'] = output['distribution']['deviations'][index:index + 1]
            row['forecast_log_weights'] = output['distribution']['log_weights'][index:index + 1]
            row['forecast_reference'] = output['distribution']['reference'][index:index + 1]
            row['c1_choice'] = c1_choices[index]
            result.append((row, output['quality'][index], image_boxes, choices[index]))
        return result

    def _advance(self, parent, image, boxes, candidates, quality, choice, fork=False):
        branch = super()._advance(parent, image, boxes, candidates, quality, choice, fork)
        if fork:
            branch.template_source_frame = parent.committed_source_frame
            branch.template_source_box = parent.committed_source_box.copy()
            branch.memory = parent.committed_memory.clone()
            branch.memory_pending.clear()
        branch.memory_pending = deque((entry for entry in branch.memory_pending
                                       if self.frame - entry[0] < self.window), maxlen=self.window)
        current_quality = candidates['current_quality'][0, choice]
        write = candidates['raw_score'][0, choice] > .84
        descriptor = self.modules.memory.encode(candidates['modality_features'][:, choice].float())
        evidence = candidates['evidence'][:, choice].float()
        observations = torch.stack((evidence[:, [0, 2, 3]], evidence[:, [0, 4, 5]]), 1)
        gate_input = torch.cat((descriptor, observations, current_quality.expand(1, 2, 1)), -1)
        trust = self.modules.memory.gate(gate_input).sigmoid().squeeze(-1) * write
        assignment = torch.einsum('bmd,smd->bsm', descriptor, self.modules.memory.slot_keys).softmax(1)
        memory_write_rates = assignment * trust[:, None]
        branch.memory = self.modules.memory.update(branch.memory[None], descriptor,
                                                   candidates['evidence'][:, choice].float(),
                                                   current_quality[None], write[None])[0]
        if bool(write):
            branch.template_source_frame = self.frame
            branch.template_source_box = branch.box.copy()
            branch.memory_pending.append((self.frame, branch.memory.clone()))
        branch.motion_boxes.append(branch.box.copy())
        branch.motion_frames.append(self.frame)
        branch.motion_quality.append(float(current_quality))
        self.decisions[branch.identifier] = {'parent_id': parent.identifier, 'choice': choice,
                                            'c1_choice': candidates['c1_choice'],
                                            'prior_template_source_frame': parent.template_source_frame,
                                            'prior_template_source_box': parent.template_source_box.copy(),
                                            'memory_write_rates': memory_write_rates[0].detach(),
                                            'boxes_xyxy': boxes.copy(), 'valid': candidates['valid'][0].detach(),
                                            'current_quality': candidates['current_quality'][0].detach(),
                                            'future_quality': candidates['future_quality'][0].detach(),
                                            'predicted_cost': candidates['predicted_cost'][0].detach(),
                                            'identity_support': candidates['identity_support'][0].detach(),
                                            'raw_score': candidates['raw_score'][0].detach(),
                                            'evidence': candidates['evidence'][0].detach(),
                                            'forecast_means': candidates['forecast_means'][0].detach(),
                                            'forecast_deviations': candidates['forecast_deviations'][0].detach(),
                                            'forecast_log_weights': candidates['forecast_log_weights'][0].detach(),
                                            'forecast_reference': candidates['forecast_reference'][0].detach(),
                                            'template_updated': bool(write)}
        return branch

    @torch.inference_mode()
    def step(self, image):
        self.decisions = {}
        commits = self.stats['template_commits']
        box = super().step(image)
        main = next(b for b in self.branches if b.identifier == self.main_id)
        if self.last_event['switched'] and not self.restore_template_state:
            previous = next(b for b in self.branches if b.identifier == self.last_event['previous_main_id'])
            # Matched box-only control carries contaminated former-main state.
            main.memory, main.committed_memory = previous.memory.clone(), previous.committed_memory.clone()
            main.template_source_frame = previous.template_source_frame
            main.template_source_box = previous.template_source_box.copy()
            main.committed_source_frame = previous.committed_source_frame
            main.committed_source_box = previous.committed_source_box.copy()
            main.memory_pending = deque(previous.memory_pending, maxlen=self.window)
            main.motion_boxes = deque((b.copy() for b in previous.motion_boxes), maxlen=previous.motion_boxes.maxlen)
            main.motion_boxes[-1] = main.box.copy()
            main.motion_frames = deque(previous.motion_frames, maxlen=previous.motion_frames.maxlen)
            main.motion_quality = deque(previous.motion_quality, maxlen=previous.motion_quality.maxlen)
        if self.stats['template_commits'] > commits:
            main.committed_source_frame = main.template_source_frame
            main.committed_source_box = main.template_source_box.copy()
            main.committed_memory = main.memory.clone()
            main.memory_pending.clear()
        assert all(b.memory.shape == (self.modules.memory.slots, 2, self.identity_anchor.shape[-1]) for b in self.branches)
        assert all(len(b.memory_pending) <= self.window and len(b.motion_boxes) <= self.modules.motion.history
                   for b in self.branches)
        self.last_decision = self.decisions[self.main_id]
        self.last_event['candidate_sets'] = self.candidate_sets
        for record, branch in zip(self.last_event['branches'], self.branches):
            decision = self.decisions[branch.identifier]
            record.update(parent_id=decision['parent_id'], choice=decision['choice'],
                          template_updated=decision['template_updated'],
                          memory_write_rates=decision['memory_write_rates'],
                          memory_committed=branch is main and self.stats['template_commits'] > commits,
                          template_source_frame=branch.template_source_frame,
                          template_source_box=branch.template_source_box.tolist(),
                          committed_source_frame=branch.committed_source_frame,
                          committed_source_box=branch.committed_source_box.tolist())
        return box
