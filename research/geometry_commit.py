"""Learn query-only search and appearance commits from causal old4 state.

The current output and motion-history observation stay unchanged. Velocity is
a next-search reference, not an asserted target identity or permanent freeze.
"""
import numpy as np
import torch
from torch import nn

from .probe_geometry_commit import next_search_prediction
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import relative_geometry
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider

ACTION_NAMES = ('raw', 'pause', 'search_velocity', 'pause_search_velocity')


def selected_index(data, query_box):
    boxes = data['image_boxes'].reshape(len(data['valid']), 35, 4)
    target = torch.as_tensor(query_box, dtype=boxes.dtype, device=boxes.device).reshape(-1, 1, 4)
    matches = data['valid'].reshape(-1, 35) & (boxes == target).all(-1)
    assert (matches.sum(1) == 1).all()
    return matches.long().argmax(1)


@torch.no_grad()
def commit_features(modules, data, anchor, memory, choice):
    inputs = modules.candidate_inputs(data, anchor, memory)[0]
    rows = torch.arange(len(choice), device=choice.device)
    boxes = data['history_boxes'].float()
    relative = relative_geometry(boxes, boxes[:, -1])
    time = (data['history_frames'] - data['history_frames'][:, -1:]).float() / 8
    valid = data['history_valid']
    history = torch.cat((relative, time[..., None], data['history_quality'].float()[..., None],
                         valid.float()[..., None]), -1) * valid[..., None]
    features = torch.cat((inputs[rows, choice], history.flatten(1)), -1)
    assert features.shape[1] == 463 and torch.isfinite(features).all()
    return features


class GeometryCommitHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(463), nn.Linear(463, 64), nn.GELU(), nn.Linear(64, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features)


def choose_commit(scores, keep, raw_write_eligible, threshold=.03):
    rows = torch.arange(len(scores), device=scores.device)
    legal = torch.ones_like(scores, dtype=torch.bool)
    legal[:, 1] = legal[:, 3] = raw_write_eligible
    assert legal[rows, keep].all()
    values, choice = scores.masked_fill(~legal, -torch.inf).max(1)
    return torch.where(values > scores[rows, keep] + threshold, choice, keep)


class GeometryCommitTracker(RecoverabilityTracker):
    def __init__(self, *args, commit_head, **kwargs):
        super().__init__(*args, **kwargs)
        self.commit_head = commit_head.eval().requires_grad_(False)
        self.stats.update(geometry_commit_interventions=0, appearance_commit_overrides=0)

    def original_inputs(self, observation, distribution, history):
        self.commit_data = super().original_inputs(observation, distribution, history)
        return self.commit_data

    def accept_observation(self, image, observation, pause):
        # The initialized tracker really has one past box on its first frame.
        # Two past centers are required by the defined velocity-search action.
        if len(self.history) < 2:
            return super().accept_observation(image, observation, pause)
        candidates, _, boxes, choice = observation
        index = selected_index(self.commit_data, boxes[choice])
        features = commit_features(self.modules, self.commit_data, self.identity_anchor, self.identity_memory, index)
        keep = torch.tensor([int(pause)], device=self.device)
        scores = self.commit_head(features)
        action = int(choose_commit(scores, keep, candidates['raw_score'][:, choice] > .84)[0])
        search_box = self.branch.search_box.copy()
        prediction = next_search_prediction(self.history, search_box, self.frame + 1) if action >= 2 else None
        actual_pause = bool(action % 2)
        result = super().accept_observation(image, observation, actual_pause)
        if action >= 2:
            provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
            provider.initialize(search_box)
            provider.update(0., prediction, np.array((image.shape[-1], image.shape[-2])))
            self.branch.search_box = provider.cached_bbox.copy()
            self.stats['geometry_commit_interventions'] += 1
        self.stats['appearance_commit_overrides'] += int(actual_pause != pause)
        self.commit_action = action
        return result

    @torch.inference_mode()
    def step(self, image):
        self.commit_action = None
        box = super().step(image)
        if self.commit_action is None:
            self.commit_action = int(self.last_decision['pause'])
        self.last_decision['commit_action'] = torch.tensor(self.commit_action, device=self.device)
        self.last_decision['pause'] = torch.tensor(bool(self.commit_action % 2), device=self.device)
        return box
