"""C extension: jointly choose a bounded alternative and its geometric/appearance commit.

A's bounded identity state and B's causal motion/search remain active. Current
output is distinct from geometric observation. No GT enters these decisions.
"""
import numpy as np
import torch
from torch import nn

from .probe_geometry_commit import private_state
from .probe_quality_geometry_commit import use_baseline_geometry
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import relative_geometry

ACTION_NAMES = ('accept_regular', 'accept_paused', 'output_only_keep_C_geometry')
FEATURES = 917


@torch.no_grad()
def decision_features(modules, data, anchor, memory, baseline, baseline_pause):
    inputs, _, _, support, _, _ = modules.candidate_inputs(data, anchor, memory)
    rows = torch.arange(len(baseline), device=baseline.device)
    reference = inputs[rows, baseline]
    boxes = data['history_boxes'][:, -8:].float()
    valid = data['history_valid'][:, -8:]
    history = torch.cat((relative_geometry(boxes, boxes[:, -1]),
                         ((data['history_frames'][:, -8:] - data['history_frames'][:, -1:]) / 8)[..., None],
                         data['history_quality'][:, -8:].float()[..., None], valid.float()[..., None]), -1)
    history *= valid[..., None]
    context = torch.cat((history.flatten(1), data['motion_means'].float().flatten(1),
                         data['motion_log_weights'].float(), baseline_pause.float()[:, None]), -1)
    region = torch.eye(7, device=inputs.device)[None, :, None].expand(len(inputs), -1, 5, -1).reshape(len(inputs), 35, 7)
    features = torch.cat((inputs, reference[:, None].expand(-1, 35, -1), region,
                          context[:, None].expand(-1, 35, -1)), -1)
    assert features.shape[1:] == (35, FEATURES) and torch.isfinite(features).all()
    output = modules.decide(data, anchor, memory)
    candidate_valid = data['valid'].flatten(1)
    quality = output['quality_logits'].sigmoid().flatten(1).masked_fill(~candidate_valid, -torch.inf)
    identity = (support[..., 0].mean(-1) - .5 * support[..., 1].clamp(min=0).mean(-1))
    identity = identity.masked_fill(~candidate_valid, -torch.inf)
    proposals = torch.zeros_like(candidate_valid)
    proposals[rows, baseline] = True
    proposals[rows, quality.argmax(1)] = True
    proposals[rows, identity.argmax(1)] = True
    legal = (proposals & candidate_valid)[..., None].expand(-1, -1, 3).clone()
    legal[..., 1] &= data['raw_score'].flatten(1) > .84
    legal[rows, baseline, 2] = False  # Same geometry; baseline pause already covers this action.
    assert legal[rows, baseline, baseline_pause.long()].all()
    return features, legal


class SelectiveStateCommitHead(nn.Module):
    def __init__(self, architecture='mlp'):
        super().__init__()
        assert architecture in ('mlp', 'linear')
        self.net = (nn.Sequential(nn.LayerNorm(FEATURES), nn.Linear(FEATURES, 3))
                    if architecture == 'linear' else
                    nn.Sequential(nn.LayerNorm(FEATURES), nn.Linear(FEATURES, 128), nn.GELU(), nn.Linear(128, 3)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features)


def choose_action(scores, legal, baseline, baseline_pause, threshold=.03):
    rows = torch.arange(len(baseline), device=baseline.device)
    keep = baseline * 3 + baseline_pause.long()
    values = scores.masked_fill(~legal, -torch.inf).flatten(1)
    best_value, chosen = values.max(1)
    return torch.where(best_value > values[rows, keep] + threshold, chosen, keep)


class SelectiveStateCommitTracker(RecoverabilityTracker):
    def __init__(self, *args, state_commit_head, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.policy == 'learned' and self.search_enabled and self.safe_writes
        assert self.write_verification == 'action' and self.search_value == 'gross'
        self.state_commit_head = state_commit_head.eval().requires_grad_(False)
        self.stats.update(state_commit_interventions=0, state_geometry_holds=0)

    def original_inputs(self, observation, distribution, history):
        self.current_observations = {}
        self.commit_data = super().original_inputs(observation, distribution, history)
        return self.commit_data

    def insert_region(self, data, region, observation):
        super().insert_region(data, region, observation)
        self.current_observations[region] = observation

    def accept_observation(self, image, observation, pause):
        candidates, _, boxes, choice = observation
        region = next(r for r, value in self.current_observations.items() if value[0] is candidates)
        baseline = torch.tensor([region * 5 + choice], device=self.device)
        baseline_pause = torch.tensor([pause], device=self.device)
        features, legal = decision_features(self.modules, self.commit_data, self.identity_anchor,
                                           self.identity_memory, baseline, baseline_pause)
        selected = int(choose_action(self.state_commit_head(features), legal, baseline, baseline_pause)[0])
        chosen, mode = divmod(selected, 3)
        q_region, q_choice = divmod(chosen, 5)
        q_candidates, q_quality, q_boxes, _ = self.current_observations[q_region]
        reference = None
        if mode == 2:
            reference = private_state(self)
            RecoverabilityTracker.accept_observation(reference, image, observation, pause)
        result = super().accept_observation(image, (q_candidates, q_quality, q_boxes, q_choice), mode > 0)
        if reference is not None:
            use_baseline_geometry(self, reference)
        self.commit_note = {'baseline': int(baseline[0]), 'choice': chosen, 'mode': mode,
                            'baseline_raw_box': boxes[choice].copy(), 'pause': mode > 0}
        self.stats['state_commit_interventions'] += int(chosen != int(baseline[0]) or (mode > 0) != pause)
        self.stats['state_geometry_holds'] += int(mode == 2)
        return result

    @torch.inference_mode()
    def step(self, image):
        box = super().step(image)
        note = self.commit_note
        previous, chosen = note['baseline'], note['choice']
        keep = int(self.last_decision['original_choice'])
        self.stats['changed_candidate_indices'] += int(chosen != keep) - int(previous != keep)
        r0, c0 = divmod(previous, 5)
        r1, c1 = divmod(chosen, 5)
        self.last_decision['boxes_xyxy'][r0, c0] = torch.as_tensor(note['baseline_raw_box'], device=self.device)
        self.last_decision['boxes_xyxy'][r1, c1] = torch.as_tensor(box, device=self.device)
        for key, value in [('choice', chosen), ('region', r1), ('pause', note['pause']),
                           ('state_commit_action', note['mode'])]:
            self.last_decision[key] = torch.tensor(value, device=self.device)
        self.last_decision['committed_search_reference'] = torch.as_tensor(self.branch.search_box, device=self.device)
        self.last_decision['committed_motion_observation'] = torch.as_tensor(self.history[-1][0], device=self.device)
        return box
