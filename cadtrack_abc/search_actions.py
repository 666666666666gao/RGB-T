"""Train search against the action selector's achievable matched consequences.

Search sees only the local summary and three frozen motion hypotheses. The
three cached extra contexts are independent training labels, never joint inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .counterfactual import ACTION_MODES, advance_candidate
from .data import clip_boxes, crop_batch
from .future_actions import CADTrackFuturePolicy, action_features, context_baseline
from .model import candidates
from .ported_modules import geometry


def selected_action(output, baseline, threshold):
    scores = output['scores'].flatten(1)
    value, index = scores.max(1)
    return torch.where(value > threshold, index, baseline * len(ACTION_MODES))


class SearchActionHead(nn.Module):
    """Values for three probability-ordered motion regions, relative to local."""
    def __init__(self, hidden=128):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(295), nn.Linear(295, hidden), nn.GELU(), nn.Linear(hidden, 6))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[3:] = -3.

    def forward(self, features):
        value, harm = self.network(features).split(3, -1)
        return {'advantage': value, 'harm_logits': harm, 'scores': value - .1 * harm.sigmoid()}


def search_objective(output, achieved_utility):
    """Current selector's legal action utility; no oracle candidate maximization."""
    utility = achieved_utility.detach()
    difference = utility[:, 1:] - utility[:, :1]
    value_loss = F.smooth_l1_loss(output['advantage'], difference)
    harm_loss = F.binary_cross_entropy_with_logits(output['harm_logits'], (difference < -.05).float())
    score = torch.cat((torch.zeros_like(output['scores'][:, :1]), output['scores']), 1)
    gap = utility[:, :, None] - utility[:, None, :]
    ranked = gap > .05
    margin = F.relu(.03 - (score[:, :, None] - score[:, None, :]))
    ranking = (margin * gap.clamp(min=0) * ranked).sum() / ranked.sum().clamp(min=1)
    return value_loss + .2 * harm_loss + ranking


class CADTrackSearchActionPolicy(CADTrackFuturePolicy):
    """Native A and motion geometry plus trained search B / six-action C.

    The search decision precedes the extra image forward. Final C sees local
    candidates or exactly one executed region; advances the state once.
    """
    def __init__(self, checkpoint, threshold=.03):
        super().__init__(checkpoint, threshold)
        self.search_action = SearchActionHead()

    @torch.inference_mode()
    def step(self, images, frames, state, training_candidates=False):
        assert len(images) == 1 and not training_candidates
        modules = self.abc
        motion = modules.motion(state['history'], state['frames'], state['valid'], state['quality'],
                                state['anchor'], state['memory'])
        patch, transform = crop_batch(images, state['box'], 4., 256, self.device)
        local = candidates(self.base(state['templates'], patch), transform, state['box'])
        encoded = modules.encode(local, state, motion, 0.)
        local_scored = modules.score(encoded, local)
        summary = torch.cat((encoded['tokens'].mean(1), encoded['tokens'].max(1).values,
                             motion['means'].flatten(1), motion['log_weights']), -1)
        search = self.search_action(summary)
        value, rank = search['scores'].max(1)
        extra = int(value[0] > self.threshold + .01)
        pair, scored = local, local_scored
        mode_order = motion['log_weights'][0].argsort(descending=True, stable=True)
        if extra:
            target = motion['means'][:, mode_order[rank[0]], 0]
            center, size = geometry(motion['reference'])
            center = center + target[:, :2] * size
            size = size * target[:, 2:].clamp(-1.3863, 1.3863).exp()
            reference = clip_boxes(torch.cat((center - size / 2, center + size / 2), -1), images)
            patch, transform = crop_batch(images, reference, 6., 256, self.device)
            additional = candidates(self.base(state['templates'], patch), transform, reference)
            additional_encoded = modules.encode(additional, state, motion, 1.)
            pair = {key: torch.cat((local[key], additional[key]), 1) for key in local}
            combined = {key: torch.cat((encoded[key], additional_encoded[key]), 1) for key in encoded}
            scored = modules.score(combined, pair)
        baseline = context_baseline(dict(pair, **scored), self.threshold)
        future = self.future_action(action_features(dict(pair, **scored), motion, state['age'], baseline), baseline)
        action = int(selected_action(future, baseline, self.threshold)[0])
        candidate, mode_index = divmod(action, len(ACTION_MODES))
        query = {'distribution': motion, 'candidates': [local], 'scored': [local_scored]}
        if extra:
            query['candidates'].append(pair)
            query['scored'].append(scored)
        trace, advanced = advance_candidate(self, images, frames, state, query, extra, candidate, ACTION_MODES[mode_index])
        raw, risk = pair['raw'][:, candidate], scored['risk'][:, candidate].sigmoid()
        write = (advanced['age'] % 5 == 0) & (raw > .65) & (risk < .5)
        scores = future['scores'].flatten(1)
        intervened = action != int(baseline[0]) * len(ACTION_MODES)
        return dict(trace, choice=torch.tensor([candidate], device=self.device),
                    local=local, local_scored=local_scored, pair=pair, scored=scored, distribution=motion,
                    write=write, geometry_reference=advanced['box'], keep_box=clip_boxes(local['boxes'][:, 0], images),
                    selected_quality=scored['quality'][:, candidate].sigmoid(),
                    future_mode=ACTION_MODES[mode_index], future_net_advantage=float(scores[0, action]) if intervened else 0.,
                    search_region_rank=int(rank[0]) if extra else -1,
                    search_region_mode=int(mode_order[rank[0]]) if extra else -1,
                    search_net_gain=float(value[0])), advanced
