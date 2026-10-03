"""Causal one-extra-region RGB-T tracking with fixed-size learned state.

No GT except initialization. Extra search is selected before its visual forward.
The trained action set is regular query write versus query-only paused write;
this implementation does not claim arbitrary historical state reconstruction.
"""
from collections import deque

import numpy as np
import torch

from .bounded_recovery import BoundedRecoveryTracker
from .candidate_learning import selection_scores
from .collect_recoverability import decode
from .collect_rollouts import observe_actions
from .recoverability_modules import select_actions
from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider


class RecoverabilityTracker(BoundedRecoveryTracker):
    def __init__(self, extractor, modules, motion, image, init_box, threshold=.03,
                 policy='learned', search_enabled=True, safe_writes=True, write_verification='identity'):
        assert policy in ('learned', 'c1')
        assert write_verification in ('identity', 'action')
        super().__init__(extractor, modules.c1, image, init_box, torch.float16)
        self.modules = modules.eval().requires_grad_(False)
        self.motion_model = motion.eval().requires_grad_(False)
        self.threshold, self.policy = threshold, policy
        self.search_enabled, self.safe_writes = search_enabled, safe_writes
        self.write_verification = write_verification
        self.branch = self.branches[0]
        weights = self.anchor_mask.flatten(1).float()
        anchors = []
        with torch.autocast('cuda', dtype=torch.float16):
            for channel in (slice(0, 3), slice(3, 6)):
                tokens = extractor.base.patch_embed(self.anchor[None, channel])
                anchors.append((tokens * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp(min=1))
        self.anchor_features = torch.stack(anchors, 1).float()
        self.identity_anchor = modules.memory.encode(self.anchor_features)
        self.identity_memory = modules.memory.initialize(self.identity_anchor)
        self.motion_anchor = motion.memory.encode(self.anchor_features)
        self.motion_memory = motion.memory.initialize(self.motion_anchor)
        self.history = deque([(init_box.copy(), 0, 1.)], maxlen=8)
        self.template_source_frame, self.template_source_box = 0, init_box.copy()
        self.stats.update(extra_searches_requested=0, extra_visual_forwards=0,
                          skipped_empty_extra_regions=0, changed_candidate_indices=0,
                          paused_query_writes=0, max_motion_history=1)
        self.last_decision = None

    def history_inputs(self):
        size = len(self.history)
        entries = [(self.history[0][0], 0, 0.)] * (8 - size) + list(self.history)
        return {'history_boxes': torch.as_tensor(np.stack([entry[0] for entry in entries]), device=self.device).float()[None],
                'history_frames': torch.tensor([[entry[1] for entry in entries]], device=self.device),
                'history_quality': torch.tensor([[entry[2] for entry in entries]], device=self.device),
                'history_valid': torch.tensor([[False] * (8 - size) + [True] * size], device=self.device)}

    def original_inputs(self, observation, distribution, history):
        candidates, quality, boxes, choice = observation
        fields = ('features', 'evidence', 'raw_score', 'boxes', 'instance_features', 'valid')
        data = {key: torch.zeros((1, 7, *candidates[key].shape[1:]),
                                 dtype=candidates[key].dtype, device=self.device) for key in fields}
        data.update(image_boxes=torch.zeros((1, 7, 5, 4), device=self.device),
                    c1_quality=torch.zeros((1, 7, 5), device=self.device),
                    original_choice=torch.tensor([choice], device=self.device),
                    anchor_features=self.anchor_features,
                    motion_means=distribution['means'], motion_log_weights=distribution['log_weights'])
        data.update(history)
        self.insert_region(data, 0, observation)
        return data

    def insert_region(self, data, region, observation):
        candidates, quality, boxes, _ = observation
        for key in ('features', 'evidence', 'raw_score', 'boxes', 'instance_features', 'valid'):
            data[key][:, region] = candidates[key]
        data['image_boxes'][:, region] = torch.as_tensor(boxes, device=self.device).float()[None]
        data['c1_quality'][:, region] = quality[None]

    def extra_observation(self, region, distribution, image):
        box = self.branch.search_box.copy()
        factor = 4.
        if region == 2:
            factor = 8.
        elif region == 3:
            directions = np.asarray(((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)))
            provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
            provider.initialize(box)
            shift = directions[self.frame % 8] * (224. / provider.get(self.search_size)[0]) * .5
            box += np.tile(shift, 2)
        elif region >= 4:
            reference = distribution['reference'][0].double().cpu().numpy()
            size = np.maximum(reference[2:] - reference[:2], 10.)
            center = reference.reshape(2, 2).mean(0)
            center += distribution['means'][0, region - 4, 0, :2].double().cpu().numpy() * size
            box = np.concatenate((center - size / 2, center + size / 2))
        provider = SiamFCCroppingParameterSimpleProvider(factor, 10.)
        provider.initialize(box)
        requested = provider.get(self.search_size)
        requested_area = float(np.prod(224. / requested[0]))
        crop, _, transform = apply_siamfc_cropping(image, self.search_size, requested,
                                                  'bilinear', False, self.image_mean)
        # Same witnessed padding-only region as the TRAIN collector; no new search fallback.
        if not np.isfinite(transform).all():
            self.stats['skipped_empty_extra_regions'] += 1
            return None, requested_area
        batch = {'z': self.anchor[None], 'd': self.anchor[None],
                 'x': self.normalization(crop / 255.)[None],
                 'z_feat_mask': self.anchor_mask, 'd_feat_mask': self.anchor_mask}
        self.extractor.proposal_policy = 'dense_raw5'
        with torch.autocast('cuda', dtype=torch.float16):
            candidates = self.extractor(batch)
            logits = self.head(candidates).float()
            scores = selection_scores(logits, candidates, self.extractor.window_penalty)
        self.extractor.proposal_policy = 'peaks'
        self.stats['extra_visual_forwards'] += 1
        choice = int(scores.masked_fill(~candidates['valid'], -torch.inf).argmax(1)[0])
        return (candidates, logits.sigmoid()[0], decode(candidates, transform, image), choice), requested_area

    @torch.inference_mode()
    def step(self, image):
        self.frame += 1
        parent = self.branch
        self.extractor.proposal_policy = 'peaks'
        original = observe_actions([(self, parent, image)], self.extractor, self.head)[0]
        history = self.history_inputs()
        distribution = self.motion_model.motion(history['history_boxes'], history['history_frames'],
                                                 history['history_valid'], history['history_quality'],
                                                 self.motion_anchor, self.motion_memory)
        data = self.original_inputs(original, distribution, history)
        first = self.modules.decide(data, self.identity_anchor, self.identity_memory)
        proposed = select_actions(first, data, self.threshold, self.write_verification)
        requested_search = self.search_enabled and bool(proposed['search_triggered'][0]) and self.policy == 'learned'
        region_searched, extra, requested_area = 0, None, 0.
        if requested_search:
            region_searched = int(proposed['searched_region'][0])
            self.stats['extra_searches_requested'] += 1
            extra, requested_area = self.extra_observation(region_searched, distribution, image)
            if extra is not None:
                self.insert_region(data, region_searched, extra)
        output = self.modules.decide(data, self.identity_anchor, self.identity_memory) if extra is not None else first
        selected = select_actions(output, data, self.threshold, self.write_verification)
        if self.policy == 'c1':
            region, choice, pause = 0, original[3], False
        else:
            region, choice, pause = int(selected['region'][0]), int(selected['candidate'][0]), bool(selected['pause'][0])
            if not self.search_enabled:
                assert region == 0
            if not self.safe_writes:
                pause = False
        observation = original if region == 0 else extra
        assert observation is not None and bool(data['valid'][0, region, choice])
        candidates, quality, boxes, _ = observation
        self.last_observation = (candidates, quality, boxes, choice)
        confidence = float(candidates['raw_score'][0, choice])
        prior_frame, prior_box = self.template_source_frame, self.template_source_box.copy()
        branch = self._advance(parent, image, boxes, candidates, quality, choice)
        write = confidence > .84 and not pause
        if confidence > .84 and pause:
            # Pause changes only appearance write, not confidence/position/provider update.
            branch.template, branch.mask = parent.template, parent.mask
            branch.pending = deque((entry for entry in parent.pending if self.frame - entry[0] < self.window), maxlen=self.window)
            self.stats['template_updates'] -= 1
            self.stats['paused_query_writes'] += 1
        if write:
            self.template_source_frame, self.template_source_box = self.frame, branch.box.copy()
        # Match float16 descriptor storage used by the causal training prefixes.
        descriptor = self.modules.memory.encode(candidates['instance_features'][:, choice].half().float())
        gate = self.modules.memory.gate_logits(descriptor, candidates['evidence'][:, choice].float(),
                                               quality[choice][None], self.identity_anchor)
        probability = gate.softmax(-1)
        verified = probability[..., 0] >= .5
        observed = torch.ones(1, dtype=torch.bool, device=self.device)
        memory_rates = self.modules.memory.write_rates(descriptor, probability, observed, verified & write)
        sources = torch.nn.functional.normalize(torch.cat((self.identity_anchor[:, None], self.identity_memory[:, :2]), 1), dim=-1)
        target_sources = torch.einsum('bsmd,bkmd->bksm', sources, output['descriptors']).argmax(2)
        self.identity_memory = self.modules.memory.update(self.identity_memory, descriptor, probability,
                                                           observed, verified & write)
        point = self.motion_model.memory.encode(candidates['modality_features'][:, choice].half().float())
        self.motion_memory = self.motion_model.memory.update(self.motion_memory, point,
                                                             candidates['evidence'][:, choice].float(), quality[choice][None],
                                                             torch.tensor([write], device=self.device))
        self.history.append((branch.box.copy(), self.frame, float(quality[choice])))
        self.branch, self.branches = branch, [branch]
        self.stats['changed_candidate_indices'] += int(region != 0 or choice != original[3])
        self.stats['max_motion_history'] = max(self.stats['max_motion_history'], len(self.history))
        self.stats['max_pending_per_branch'] = max(self.stats['max_pending_per_branch'], len(branch.pending))
        assert len(self.history) <= 8 and len(branch.pending) <= self.window
        diagnostic_boxes = torch.zeros((7, 5, 4), dtype=torch.float64, device=self.device)
        diagnostic_boxes[0] = torch.as_tensor(original[2], device=self.device)
        if extra is not None:
            diagnostic_boxes[region_searched] = torch.as_tensor(extra[2], device=self.device)
        self.last_decision = {'boxes_xyxy': diagnostic_boxes, 'valid': data['valid'][0],
                              'choice': torch.tensor(region * 5 + choice, device=self.device),
                              'original_choice': torch.tensor(original[3], device=self.device),
                              'region': torch.tensor(region, device=self.device), 'pause': torch.tensor(pause, device=self.device),
                              'evidence': data['evidence'][0], 'raw_score': data['raw_score'][0],
                              'current_quality': output['quality_logits'][0].sigmoid(),
                              'predicted_advantage': output['advantage'][0],
                              'target_probability': output['target_probability'][0],
                              'region_advantage': output['region_advantage'][0],
                              'absence_probability': output['absence_logit'][0].sigmoid(),
                              'search_requested': torch.tensor(requested_search, device=self.device),
                              'searched_region': torch.tensor(region_searched, device=self.device),
                              'extra_executed': torch.tensor(extra is not None, device=self.device),
                              'requested_extra_area': torch.tensor(requested_area, device=self.device, dtype=torch.float64),
                              'template_updated': torch.tensor(write, device=self.device),
                              'prior_template_frame': torch.tensor(prior_frame, device=self.device),
                              'prior_template_box': torch.as_tensor(prior_box, device=self.device),
                              'template_source_frame': torch.tensor(self.template_source_frame, device=self.device),
                              'template_source_box': torch.as_tensor(self.template_source_box, device=self.device),
                              'memory_target_commit': (verified & write)[0],
                              'memory_write_rates': memory_rates[0],
                              'target_source_indices': target_sources[0].reshape(7, 5, 2),
                              'forecast_means': distribution['means'][0],
                              'forecast_deviations': distribution['deviations'][0],
                              'forecast_reference': distribution['reference'][0],
                              'forecast_log_weights': distribution['log_weights'][0]}
        # Preserve actual double-precision output box, not its float32 model input copy.
        self.last_decision['boxes_xyxy'][region, choice] = torch.as_tensor(branch.box, device=self.device)
        return branch.box.copy()
