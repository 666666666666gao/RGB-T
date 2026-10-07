"""Matched state-propagation controls sharing one primary visual observation.

The protected reference chooses GOLA's retained Hann winner (index zero).
It is a model prediction, never a trusted ground-truth observation. Geometry
alone is not a native-GOLA trajectory: that arm uses the ABC online template.
Visual-reference arms preserve both native search and native online template;
the ABC output, identity memory and optional extra search remain separate.
No new optimizer update or new trained-module contribution is claimed here.
"""
import copy
from collections import deque

import torch

from .bounded_recovery import BoundedRecoveryTracker
from .collect_rollouts import observe_actions
from .recoverability_tracker import RecoverabilityTracker

MODES = ('parent', 'geometry', 'visual', 'visual_motion')


class ProtectedVisualReferenceTracker(RecoverabilityTracker):
    def __init__(self, *args, reference_mode, **kwargs):
        super().__init__(*args, **kwargs)
        assert reference_mode in MODES
        self.reference_mode = reference_mode
        self.reference_branch = self.branch.copy()
        self.reference_template_frame = 0
        self.reference_template_box = self.branch.box.copy()
        self.reference_write = False

    @torch.inference_mode()
    def local_observation(self, image):
        if self.reference_mode == 'parent':
            observation = super().local_observation(image)
        else:
            source = self.branch.copy()
            source.search_box = self.reference_branch.search_box.copy()
            if self.reference_mode in ('visual', 'visual_motion'):
                source.template, source.mask = self.reference_branch.template, self.reference_branch.mask
            observation = observe_actions([(self, source, image)], self.extractor, self.head)[0]
        self.reference_observation = observation
        self.reference_incoming_box = (self.branch.search_box.copy() if self.reference_mode == 'parent'
                                       else self.reference_branch.search_box.copy())
        self.reference_incoming_template_frame = (self.template_source_frame if self.reference_mode in ('parent', 'geometry')
                                                 else self.reference_template_frame)
        return observation

    def accept_observation(self, image, observation, pause):
        if self.reference_mode == 'parent':
            result = super().accept_observation(image, observation, pause)
            self.reference_write = result[1]
            return result
        candidates, quality, boxes, _ = self.reference_observation
        # Advance the reference from the SAME visual forward, without changing
        # ABC counters or using any additional region/backbone evaluation.
        shadow = copy.copy(self)
        shadow.stats = self.stats.copy()
        reference = BoundedRecoveryTracker._advance(shadow, self.reference_branch, image,
                                                   boxes, candidates, quality, 0)
        reference_write = float(candidates['raw_score'][0, 0]) > .84
        self.reference_write = reference_write
        if reference_write:
            self.reference_template_frame, self.reference_template_box = self.frame, reference.box.copy()
        if self.reference_mode == 'visual_motion':
            previous_motion = self.motion_memory.clone()
            previous_history = deque(self.history, maxlen=self.history.maxlen)
        result = super().accept_observation(image, observation, pause)
        self.reference_branch = reference
        if self.reference_mode == 'visual_motion':
            point = self.motion_model.memory.encode(candidates['modality_features'][:, 0].half().float())
            self.motion_memory = self.motion_model.memory.update(previous_motion, point,
                candidates['evidence'][:, 0].float(), quality[0][None],
                torch.tensor([reference_write], device=self.device))
            previous_history.append((reference.box.copy(), self.frame, float(quality[0])))
            self.history = previous_history
        return result

    @torch.inference_mode()
    def step(self, image):
        result = super().step(image)
        reference = self.branch if self.reference_mode == 'parent' else self.reference_branch
        self.last_decision.update(
            protected_reference_box=torch.as_tensor(reference.box, device=self.device),
            protected_reference_search_box=torch.as_tensor(reference.search_box, device=self.device),
            incoming_primary_search_box=torch.as_tensor(self.reference_incoming_box, device=self.device),
            incoming_primary_template_frame=torch.tensor(self.reference_incoming_template_frame, device=self.device),
            protected_reference_template_frame=torch.tensor(
                self.template_source_frame if self.reference_mode == 'parent' else self.reference_template_frame,
                device=self.device),
            protected_reference_template_updated=torch.tensor(self.reference_write, device=self.device),
            committed_motion_observation=torch.as_tensor(self.history[-1][0], device=self.device))
        return result
