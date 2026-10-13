"""Future-action C head and causal runtime, separate from the running CAD fit.

The head reads frozen current/historical evidence. Only objective() reads the
matched continuation labels. Runtime reuses the native observed visual budget.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .counterfactual import ACTION_MODES, advance_candidate, clone_state
from .model import CADTrackABC
from .ported_modules import relative_geometry


def context_baseline(context, threshold):
    """Native keep/reselect rule within one legally observed context."""
    scores = context['score'].clone()
    if scores.shape[1] == 10:
        scores[:, 5:] -= .01
    best = scores.argmax(-1)
    value = scores.gather(1, best[:, None]).squeeze(1)
    return torch.where(value > scores[:, 0] + threshold, best, torch.zeros_like(best))


def action_features(context, distribution, age, baseline):
    """No GT/future images: a candidate, native reference and causal commit phase."""
    tokens = context['tokens']
    rows = torch.arange(len(tokens), device=tokens.device)
    reference = tokens[rows, baseline]
    probabilities = context['gate'].softmax(-1)[..., 0]
    phase = ((age + 1) % 5 == 0).to(tokens.dtype)[:, None, None].expand(-1, tokens.shape[1], -1)
    trust = distribution['history_trust'][:, None, None].expand_as(phase)
    coordinates = relative_geometry(context['boxes'], distribution['reference'])
    features = torch.cat((tokens, tokens - reference[:, None], context['raw'][..., None],
                          context['hann'][..., None], context['quality'].sigmoid()[..., None],
                          context['risk'].sigmoid()[..., None], context['geometry'].sigmoid()[..., None],
                          probabilities, context['support'].flatten(-2), phase, coordinates, trust), -1)
    assert features.shape[-1] == 275
    return features.detach()


class FutureActionHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(275), nn.Linear(275, hidden), nn.GELU(),
                                     nn.Linear(hidden, len(ACTION_MODES) * 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[len(ACTION_MODES):] = -3.

    def forward(self, features, baseline):
        prediction = self.network(features)
        value, harm_logits = prediction.split(len(ACTION_MODES), -1)
        rows = torch.arange(len(features), device=features.device)
        reference = value[rows, baseline, 0]
        reference_harm = harm_logits[rows, baseline, 0].sigmoid()
        advantage = value - reference[:, None, None]
        scores = advantage - .1 * (harm_logits.sigmoid() - reference_harm[:, None, None]).clamp(min=0)
        return {'advantage': advantage, 'harm_logits': harm_logits, 'scores': scores}


def objective(output, utilities, baseline):
    """Within the same observed context, improve upon the actual native action."""
    rows = torch.arange(len(utilities), device=utilities.device)
    reference = utilities[rows, baseline, 0]
    delta = utilities - reference[:, None, None]
    value_loss = F.smooth_l1_loss(output['advantage'], delta)
    harm_loss = F.binary_cross_entropy_with_logits(output['harm_logits'], (delta < -.05).float())
    values, scores = utilities.flatten(1), output['scores'].flatten(1)
    gap = values[:, :, None] - values[:, None, :]
    ranked = gap > .05
    margin = F.relu(.03 - (scores[:, :, None] - scores[:, None, :]))
    ranking = (margin * gap.clamp(min=0) * ranked).sum() / ranked.sum().clamp(min=1)
    return value_loss + ranking + .2 * harm_loss, {
        'relative_future_value': value_loss.detach(), 'action_ranking': ranking.detach(),
        'future_harm': harm_loss.detach()}


class CADTrackFuturePolicy(CADTrackABC):
    """Adds future C decisions to a separately loaded, frozen native A/B/C teacher.

    A checkpoint loader must strictly load abc and future_action independently.
    This class is not used by the currently running native 20-epoch pipeline.
    """
    def __init__(self, checkpoint, threshold=.03):
        super().__init__(checkpoint, threshold)
        self.abc.requires_grad_(False).eval()
        self.future_action = FutureActionHead()

    def train(self, mode=True):
        super().train(mode)
        self.abc.eval()
        return self

    @torch.inference_mode()
    def step(self, images, frames, state, training_candidates=False):
        assert len(images) == 1 and not training_candidates
        prior = clone_state(state)
        observation, native_state = super().step(images, frames, clone_state(prior))
        extra = int(observation['extra_forwards'] > 0)
        assert observation['extra_forwards'] == extra
        candidates = observation['pair'] if extra else observation['local']
        scored = observation['scored'] if extra else observation['local_scored']
        context = dict(candidates, **scored)
        baseline = observation['choice']
        features = action_features(context, observation['distribution'], prior['age'], baseline)
        future = self.future_action(features, baseline)
        scores = future['scores'][0]
        best = int(scores.flatten().argmax())
        candidate, mode_index = divmod(best, len(ACTION_MODES))
        intervene = bool(scores.flatten()[best] > self.threshold)
        if not intervene:
            observation.update(future_mode='policy', future_net_advantage=0.)
            return observation, native_state
        mode = ACTION_MODES[mode_index]
        query = {'distribution': observation['distribution'],
                 'candidates': [observation['local']], 'scored': [observation['local_scored']]}
        if extra:
            query['candidates'].append(observation['pair'])
            query['scored'].append(observation['scored'])
        trace, advanced = advance_candidate(self, images, frames, prior, query, extra, candidate, mode)
        raw = candidates['raw'][:, candidate]
        risk = scored['risk'][:, candidate].sigmoid()
        write = (advanced['age'] % 5 == 0) & (raw > .65) & (risk < .5)
        observation.update(box=trace['box'], choice=torch.tensor([candidate], device=self.device),
                           geometry_commit=trace['geometry_commit'], write=write,
                           template_commit=trace['template_commit'], target_memory_commit=trace['target_memory_commit'],
                           geometry_reference=advanced['box'], selected_quality=scored['quality'][:, candidate].sigmoid(),
                           future_mode=mode, future_net_advantage=float(scores[candidate, mode_index]))
        return observation, advanced
