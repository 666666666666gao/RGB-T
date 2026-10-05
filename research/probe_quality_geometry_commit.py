"""Query-only causal prototype; parent policy and future frames remain unchanged.

Prepared for TRAIN action comparisons, not enabled in the main evaluator.
Quality-only selection is known to harm some saved states; these controls label
its output/appearance/geometry consequences rather than claim a better tracker.
"""
import torch

from .probe_geometry_commit import private_state
from .recoverability_tracker import RecoverabilityTracker

MODES = ('raw', 'quality_regular', 'quality_paused', 'quality_paused_geometry_keep')


def quality_extra_choice(data, quality_logits, baseline):
    """Preserve observed C's local choice; only an executed extra may replace it."""
    valid = data['valid'].flatten(1)
    rows = torch.arange(len(valid), device=valid.device)
    indices = torch.arange(35, device=valid.device)
    score = quality_logits.sigmoid().flatten(1) - .01 * (indices >= 5)[None]
    available = valid & (indices >= 5)[None]
    available[rows, baseline] = True
    best = score.masked_fill(~available, -torch.inf).argmax(1)
    return torch.where(score[rows, best] > score[rows, baseline] + .03, best, baseline)


def use_baseline_geometry(tracker, baseline):
    """Copy this frame's baseline geometry, leaving output and appearance alone."""
    assert tracker.frame == baseline.frame == tracker.history[-1][1] == baseline.history[-1][1]
    box, frame, quality = baseline.history[-1]
    tracker.branch.search_box = baseline.branch.search_box.copy()
    tracker.history[-1] = (box.copy(), frame, quality)


class QualityGeometryQueryTracker(RecoverabilityTracker):
    def __init__(self, *args, query, mode, **kwargs):
        assert mode in MODES
        super().__init__(*args, **kwargs)
        assert self.policy == 'learned' and self.search_enabled and self.safe_writes
        assert self.write_verification == 'action' and self.search_value == 'gross'
        self.query, self.probe_mode = query, mode
        self.query_record = None

    def original_inputs(self, observation, distribution, history):
        self.current_observations = {}
        self.query_data = super().original_inputs(observation, distribution, history)
        return self.query_data

    def insert_region(self, data, region, observation):
        super().insert_region(data, region, observation)
        self.current_observations[region] = observation

    def accept_observation(self, image, observation, pause):
        if self.frame != self.query:
            return super().accept_observation(image, observation, pause)
        candidates, _, boxes, choice = observation
        region = next(r for r, value in self.current_observations.items() if value[0] is candidates)
        baseline_index = region * 5 + choice
        chosen = baseline_index
        if self.probe_mode != 'raw':
            output = self.modules.decide(self.query_data, self.identity_anchor, self.identity_memory)
            chosen = int(quality_extra_choice(self.query_data, output['quality_logits'],
                                              torch.tensor([baseline_index], device=self.device))[0])
        q_region, q_choice = divmod(chosen, 5)
        q_candidates, q_quality, q_boxes, _ = self.current_observations[q_region]
        actual_pause = pause if self.probe_mode == 'raw' else self.probe_mode != 'quality_regular'
        reference = None
        if self.probe_mode == 'quality_paused_geometry_keep':
            reference = private_state(self)
            # Replay C's actual current acceptance on private state, including its
            # real clipping/provider rules. This does not change the real tracker.
            RecoverabilityTracker.accept_observation(reference, image, observation, pause)
        result = super().accept_observation(image, (q_candidates, q_quality, q_boxes, q_choice), actual_pause)
        if reference is not None:
            use_baseline_geometry(self, reference)
        self.query_record = {'baseline_index': baseline_index, 'chosen_index': chosen,
                             'baseline_raw_box': boxes[choice].copy(),
                             'pause': bool(actual_pause), 'mode': self.probe_mode,
                             'output_box': self.branch.box.copy(),
                             'search_reference': self.branch.search_box.copy(),
                             'motion_observation': self.history[-1][0].copy(),
                             'motion_frame': self.history[-1][1],
                             'legal_write_pair': bool(q_candidates['raw_score'][0, q_choice] > .84)}
        return result

    @torch.inference_mode()
    def step(self, image):
        box = super().step(image)
        if self.frame == self.query:
            note = self.query_record
            previous, chosen = note['baseline_index'], note['chosen_index']
            keep = int(self.last_decision['original_choice'])
            self.stats['changed_candidate_indices'] += int(chosen != keep) - int(previous != keep)
            # Core step emitted C's bookkeeping; restore that unselected slot and
            # identify the actual replacement/output before saving diagnostics.
            old_region, old_choice = divmod(previous, 5)
            region, choice = divmod(chosen, 5)
            self.last_decision['boxes_xyxy'][old_region, old_choice] = torch.as_tensor(note['baseline_raw_box'], device=self.device)
            self.last_decision['boxes_xyxy'][region, choice] = torch.as_tensor(box, device=self.device)
            for key, value in (('choice', chosen), ('region', region), ('pause', note['pause'])):
                self.last_decision[key] = torch.tensor(value, device=self.device)
        return box
