"""C2 prototype: finite alternative paths with private template state.

The scorer is the already trained C1 head. This stage does not implement the
C3 future-utility supervision, learned A compression, or B motion predictor.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .candidate_learning import foreground_mask, selection_scores, box_iou
from trackit.core.utils.siamfc_cropping import (
    get_siamfc_cropping_params, apply_siamfc_cropping,
    apply_siamfc_cropping_to_boxes, reverse_siamfc_cropping_params,
)
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider


@dataclass
class Branch:
    identifier: int
    born: int
    box: np.ndarray
    search_box: np.ndarray
    template: torch.Tensor
    mask: torch.Tensor
    committed_template: torch.Tensor
    committed_mask: torch.Tensor
    pending: deque
    scores: deque
    boxes: deque
    selected_streak: int = 0

    def copy(self, identifier=None, born=None):
        return Branch(self.identifier if identifier is None else identifier,
                      self.born if born is None else born, self.box.copy(), self.search_box.copy(),
                      self.template, self.mask, self.committed_template, self.committed_mask,
                      deque(self.pending, maxlen=self.pending.maxlen),
                      deque(self.scores, maxlen=self.scores.maxlen),
                      deque(self.boxes, maxlen=self.boxes.maxlen), self.selected_streak)

    @property
    def value(self):
        return sum(self.scores) / len(self.scores)


class BoundedRecoveryTracker:
    def __init__(self, extractor, head, image, init_box, amp_dtype,
                 branches=3, window=5, switch_patience=2, restore_template_state=True):
        assert branches >= 2 and window >= switch_patience >= 1
        self.extractor, self.head, self.amp_dtype = extractor, head, amp_dtype
        self.budget, self.window, self.switch_patience = branches, window, switch_patience
        self.restore_template_state = restore_template_state
        self.normalization = get_dataset_norm_stats_transform('mm', inplace=True)
        self.template_size, self.search_size = np.array((112, 112)), np.array((224, 224))
        self.device = image.device
        params = get_siamfc_cropping_params(init_box, 2., self.template_size)
        z, self.image_mean, params = apply_siamfc_cropping(image, self.template_size, params, 'bilinear', False)
        self.anchor = self.normalization(z / 255.)
        self.anchor_mask = foreground_mask(init_box, params).to(self.device).unsqueeze(0)
        self.branches = [Branch(0, 0, init_box.copy(), init_box.copy(), self.anchor, self.anchor_mask,
                                self.anchor, self.anchor_mask, deque(maxlen=window),
                                deque(maxlen=window), deque([init_box.copy()], maxlen=window))]
        self.main_id, self.next_id, self.frame = 0, 1, 0
        self.challenger_id, self.challenger_streak = None, 0
        self.stats = dict(forks=0, expired=0, switches=0, template_state_restores=0,
                          template_commits=0, template_updates=0, max_branches=1,
                          max_pending_per_branch=0, alternative_selections=0)
        self.last_event = None

    def _observe_many(self, branches, image):
        crops, parameters = [], []
        for branch in branches:
            provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
            provider.initialize(branch.search_box)
            crop, _, params = apply_siamfc_cropping(
                image, self.search_size, provider.get(self.search_size), 'bilinear', False, self.image_mean)
            crops.append(self.normalization(crop / 255.))
            parameters.append(params)
        n = len(branches)
        batch = {'z': self.anchor.unsqueeze(0).expand(n, -1, -1, -1),
                 'd': torch.stack([branch.template for branch in branches]), 'x': torch.stack(crops),
                 'z_feat_mask': self.anchor_mask.expand(n, -1, -1),
                 'd_feat_mask': torch.cat([branch.mask for branch in branches])}
        with torch.autocast('cuda', dtype=self.amp_dtype):
            candidates = self.extractor(batch)
            logits = self.head(candidates).float()
            scores = selection_scores(logits, candidates, self.extractor.window_penalty)
        choices = scores.masked_fill(~candidates['valid'], -torch.inf).argmax(1).tolist()
        crop_boxes = (candidates['boxes'] * 224.).double().cpu().numpy()
        size = np.array((image.shape[-1], image.shape[-2]))
        observations = []
        for index, params in enumerate(parameters):
            boxes = apply_siamfc_cropping_to_boxes(crop_boxes[index], reverse_siamfc_cropping_params(params))
            for box in boxes:
                bbox_clip_to_image_boundary_(box, size)
            assert np.isfinite(boxes).all()
            observations.append(({key: value[index:index + 1] for key, value in candidates.items()},
                                 logits.sigmoid()[index], boxes, choices[index]))
        return observations

    def _advance(self, parent, image, boxes, candidates, quality, choice, fork=False):
        branch = parent.copy(self.next_id, self.frame) if fork else parent.copy()
        if fork:
            self.next_id += 1
            branch.selected_streak = 0
            branch.template, branch.mask = parent.committed_template, parent.committed_mask
            branch.pending.clear()
            self.stats['forks'] += 1
        branch.box = boxes[choice].copy()
        branch.boxes.append(branch.box.copy())
        branch.scores.append(float(quality[choice]))
        branch.pending = deque((entry for entry in branch.pending if self.frame - entry[0] < self.window), maxlen=self.window)
        confidence = float(candidates['raw_score'][0, choice])
        assert np.isfinite(confidence) and np.isfinite(branch.value)
        provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
        provider.initialize(parent.search_box)
        provider.update(confidence, branch.box, np.array((image.shape[-1], image.shape[-2])))
        branch.search_box = provider.cached_bbox.copy()
        if confidence > .84:
            params = get_siamfc_cropping_params(branch.box, 2., self.template_size)
            template, _, _ = apply_siamfc_cropping(image, self.template_size, params, 'bilinear', False)
            template = self.normalization(template / 255.)
            mask = foreground_mask(branch.box, params).to(self.device).unsqueeze(0)
            branch.template, branch.mask = template, mask
            branch.pending.append((self.frame, template, mask))
            self.stats['template_updates'] += 1
        return branch

    @torch.inference_mode()
    def step(self, image):
        self.frame += 1
        kept = [b for b in self.branches if b.identifier == self.main_id or self.frame - b.born < self.window]
        self.stats['expired'] += len(self.branches) - len(kept)
        observations, next_branches, proposals = [], [], []
        for parent, observation in zip(kept, self._observe_many(kept, image)):
            candidates, quality, boxes, choice = observation
            observations.append((parent, candidates, quality, boxes, choice))
            next_branches.append(self._advance(parent, image, boxes, candidates, quality, choice))
            self.stats['alternative_selections'] += int(choice != 0)
            for other in torch.where(candidates['valid'][0])[0].tolist():
                if other != choice:
                    proposals.append((float(quality[other]), len(observations) - 1, other))
        for _, parent_index, choice in sorted(proposals, reverse=True):
            if len(next_branches) == self.budget:
                break
            parent, candidates, quality, boxes, _ = observations[parent_index]
            existing = torch.as_tensor(np.array([b.box for b in next_branches]), device=self.device)
            candidate_box = torch.as_tensor(boxes[choice], device=self.device)
            if bool((box_iou(existing, candidate_box) < self.extractor.nms_iou).all()):
                next_branches.append(self._advance(parent, image, boxes, candidates, quality, choice, fork=True))
        previous_main = next(b for b in next_branches if b.identifier == self.main_id)
        best = max(next_branches, key=lambda b: (b.value, b.identifier == self.main_id))
        if best.identifier != self.main_id:
            self.challenger_streak = self.challenger_streak + 1 if self.challenger_id == best.identifier else 1
            self.challenger_id = best.identifier
        else:
            self.challenger_id, self.challenger_streak = None, 0
        switched = best.identifier != self.main_id and self.challenger_streak >= self.switch_patience
        if switched:
            previous_main.born = self.frame
            if not self.restore_template_state:
                # Matched box-only ablation: carry the former main template state.
                best.template, best.mask = previous_main.template, previous_main.mask
                best.committed_template, best.committed_mask = previous_main.committed_template, previous_main.committed_mask
                best.pending = deque(previous_main.pending, maxlen=self.window)
            else:
                self.stats['template_state_restores'] += 1
            self.main_id = best.identifier
            self.stats['switches'] += 1
            self.challenger_id, self.challenger_streak = None, 0
        main = next(b for b in next_branches if b.identifier == self.main_id)
        for branch in next_branches:
            branch.selected_streak = branch.selected_streak + 1 if branch is main else 0
        if main.selected_streak >= self.window and main.pending:
            _, main.committed_template, main.committed_mask = main.pending[-1]
            main.pending.clear()
            self.stats['template_commits'] += 1
        self.branches = next_branches
        self.stats['max_branches'] = max(self.stats['max_branches'], len(self.branches))
        self.stats['max_pending_per_branch'] = max(self.stats['max_pending_per_branch'], max(len(b.pending) for b in self.branches))
        assert len(self.branches) <= self.budget
        assert all(len(b.pending) <= self.window and len(b.scores) <= self.window and len(b.boxes) <= self.window for b in self.branches)
        self.last_event = {'frame': self.frame, 'main_id': self.main_id, 'switched': switched,
                           'previous_main_id': previous_main.identifier,
                           'branches': [{'id': b.identifier, 'value': b.value, 'born': b.born,
                                         'pending_updates': len(b.pending), 'box': b.box.tolist()} for b in self.branches]}
        return main.box.copy()
