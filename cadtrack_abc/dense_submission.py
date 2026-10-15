"""Candidate evidence and six graded actions for the isolated dense R3 path.

R1 head architectures are reused. Current-IoU proxy heads are distinguished
from future harm supervision; future labels are used only by the objectives.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .counterfactual import ACTION_MODES
from .future_actions import FutureActionHead, action_features
from .search_actions import SearchActionHead, selected_action
from .ported_modules import relative_geometry
from .dense_temporal import TrustedMotion


class DenseCandidateEvidence(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fused = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.embedding = nn.Sequential(nn.Linear(hidden * 3 + 15, hidden), nn.GELU())
        self.relations = nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
        self.normalization = nn.LayerNorm(hidden)
        self.proxy = nn.Linear(hidden, 3)
        nn.init.zeros_(self.proxy.weight)
        nn.init.zeros_(self.proxy.bias)
        with torch.no_grad():
            self.proxy.bias[1], self.proxy.bias[2] = -3., 3.

    def encode(self, candidate, appearance, distribution, source):
        coordinates = relative_geometry(candidate['boxes'], distribution['reference'])
        density = TrustedMotion.log_probability(distribution, coordinates).clamp(-20, 10)
        trust = distribution['history_trust'][:, None].expand_as(density)
        inputs = torch.cat((self.fused(candidate['fused']), appearance['descriptor'].flatten(-2),
            appearance['support'].flatten(-2), coordinates, candidate['raw'][..., None],
            candidate['hann'][..., None], density[..., None], trust[..., None],
            torch.full_like(density[..., None], source)), -1)
        return dict(tokens=self.embedding(inputs), descriptors=appearance['descriptor'],
            gate=appearance['gate'], support=appearance['support'])

    def contextualize(self, candidate, encoded):
        tokens = encoded['tokens']
        context = self.relations(tokens, tokens, tokens, need_weights=False)[0]
        tokens = self.normalization(tokens + context)
        proxy = self.proxy(tokens)
        return dict(encoded, tokens=tokens,
            quality=torch.logit(candidate['raw'].clamp(1e-5, 1 - 1e-5)) + proxy[..., 0],
            risk=proxy[..., 1], geometry=proxy[..., 2])


class DenseDecisionHeads(nn.Module):
    def __init__(self, threshold=.03):
        super().__init__()
        self.threshold = threshold
        self.future_action = FutureActionHead()
        self.search_action = SearchActionHead()

    def search(self, context, distribution):
        features = torch.cat((context['tokens'].mean(1), context['tokens'].max(1).values,
            distribution['means'].flatten(1), distribution['log_weights']), -1)
        return self.search_action(features.detach()), features.detach()

    def choose(self, candidate, context, distribution, age):
        baseline = torch.zeros(len(candidate['raw']), device=candidate['raw'].device, dtype=torch.long)
        output = self.future_action(action_features(dict(candidate, **context), distribution, age, baseline), baseline)
        chosen = selected_action(output, baseline, self.threshold)
        return chosen // len(ACTION_MODES), chosen % len(ACTION_MODES), output


def submission_masks(choice, mode, geometry_logits):
    # Local keep is the original R2 geometry policy. Alternative policy actions
    # use the learned localization proxy; explicit geometry actions override it.
    geometric = (choice == 0) | (geometry_logits.sigmoid() >= .5)
    geometric = torch.where(mode == ACTION_MODES.index('defer_geometry'), False, geometric)
    geometric = torch.where(mode == ACTION_MODES.index('accept_geometry'), True, geometric)
    # R3 defer_geometry quarantines this observation: geometry AND identity
    # remain uncommitted. R1 consequences (which allowed identity here) cannot
    # be reused for this state/action definition.
    normal = (mode == 0) | (mode == 3)
    permit = torch.stack((normal | (mode == 4), normal | (mode == 5)), -1)
    return geometric, permit


def proxy_objective(context, quality, valid):
    weight = valid[:, None].float()
    denominator = weight.sum().clamp(min=1) * quality.shape[1]
    loss = (F.binary_cross_entropy_with_logits(context['quality'], quality, reduction='none') * weight).sum() / denominator
    risk = (F.binary_cross_entropy_with_logits(context['risk'], (quality < .2).float(), reduction='none') * weight).sum() / denominator
    geometric = (F.binary_cross_entropy_with_logits(context['geometry'], (quality >= .5).float(), reduction='none') * weight).sum() / denominator
    return loss + .2 * (risk + geometric)


def consequence_objective(output, utility, future_mean_iou, future_lost_fraction, baseline):
    """Advantage uses realized utility; harm explicitly uses future degradation."""
    rows = torch.arange(len(utility), device=utility.device)
    reference = utility[rows, baseline, 0]
    delta = utility - reference[:, None, None]
    reference_future = future_mean_iou[rows, baseline, 0]
    reference_lost = future_lost_fraction[rows, baseline, 0]
    harmful = (future_mean_iou < reference_future[:, None, None] - .05)
    harmful |= future_lost_fraction > reference_lost[:, None, None] + .05
    value = F.smooth_l1_loss(output['advantage'], delta)
    harm = F.binary_cross_entropy_with_logits(output['harm_logits'], harmful.float())
    values, scores = utility.flatten(1), output['scores'].flatten(1)
    gap = values[:, :, None] - values[:, None, :]
    ranked = gap > .05
    margin = F.relu(.03 - (scores[:, :, None] - scores[:, None, :]))
    ranking = (margin * gap.clamp(min=0) * ranked).sum() / ranked.sum().clamp(min=1)
    return value + ranking + .2 * harm


def search_consequence_objective(output, utility, future_mean_iou, future_lost_fraction):
    difference = utility[:, 1:] - utility[:, :1]
    harmful = (future_mean_iou[:, 1:] < future_mean_iou[:, :1] - .05)
    harmful |= future_lost_fraction[:, 1:] > future_lost_fraction[:, :1] + .05
    value = F.smooth_l1_loss(output['advantage'], difference)
    harm = F.binary_cross_entropy_with_logits(output['harm_logits'], harmful.float())
    score = torch.cat((torch.zeros_like(output['scores'][:, :1]), output['scores']), 1)
    gap = utility[:, :, None] - utility[:, None, :]
    ranked = gap > .05
    margin = F.relu(.03 - (score[:, :, None] - score[:, None, :]))
    ranking = (margin * gap.clamp(min=0) * ranked).sum() / ranked.sum().clamp(min=1)
    return value + ranking + .2 * harm
