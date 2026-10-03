"""Coupled A memory, B identity-conditioned futures, and C candidate value.

Forward receives past observations and current candidates only. Ground truth
and future observations occur exclusively in temporal_objective.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_learning import CandidateQualityHead

DECISION_FIELDS = ('features', 'evidence', 'raw_score', 'boxes', 'modality_features',
                   'anchor_features', 'image_boxes', 'valid', 'c1_quality',
                   'history_descriptors', 'history_evidence', 'history_quality',
                   'history_boxes', 'history_frames', 'history_write', 'history_valid')


def geometry(boxes):
    size = (boxes[..., 2:] - boxes[..., :2]).clamp(min=10.)
    return (boxes[..., :2] + boxes[..., 2:]) / 2, size


def relative_geometry(boxes, reference):
    center, size = geometry(boxes)
    origin, scale = geometry(reference)
    return torch.cat(((center - origin.unsqueeze(-2)) / scale.unsqueeze(-2),
                      torch.log(size / scale.unsqueeze(-2))), -1)


class DiscriminativeMemory(nn.Module):
    def __init__(self, hidden=128, slots=4):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.slot_keys = nn.Parameter(torch.randn(slots, 2, hidden) * .02)
        self.gate = nn.Sequential(nn.Linear(hidden + 4, 64), nn.GELU(), nn.Linear(64, 1))
        self.slots = slots

    def encode(self, features):
        return F.normalize(self.projection(features), dim=-1)

    def initialize(self, anchor):
        return anchor.unsqueeze(1).expand(-1, self.slots, -1, -1).clone()

    def update(self, memory, descriptor, evidence, quality, write):
        rgb = evidence[:, [0, 2, 3]]
        tir = evidence[:, [0, 4, 5]]
        observations = torch.stack((rgb, tir), 1)
        inputs = torch.cat((descriptor, observations,
                            quality[:, None, None].expand(-1, 2, 1)), -1)
        trust = self.gate(inputs).sigmoid().squeeze(-1) * write[:, None]
        assignment = torch.einsum('bmd,smd->bsm', descriptor, self.slot_keys).softmax(1)
        rate = assignment * trust[:, None]
        return memory * (1 - rate[..., None]) + descriptor[:, None] * rate[..., None]

    @staticmethod
    def support(memory, anchor, descriptors):
        sources = F.normalize(torch.cat((anchor[:, None], memory), 1), dim=-1)
        return torch.einsum('bsmd,bkmd->bksm', sources, descriptors).max(2).values


class IdentityMotion(nn.Module):
    def __init__(self, hidden=128, history=8, modes=3, horizon=3):
        super().__init__()
        self.history, self.modes, self.horizon = history, modes, horizon
        self.encoder = nn.Sequential(nn.Linear(history * 6 + hidden * 2 + 1, hidden), nn.GELU(),
                                     nn.Linear(hidden, modes * horizon * 8 + modes))
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        with torch.no_grad():
            parameters = self.encoder[-1].bias[:modes * horizon * 8].view(modes, horizon, 8)
            parameters[..., 4:] = -1.5
            parameters[0, :, 0] = -.3
            parameters[2, :, 0] = .3

    def forward(self, boxes, frames, valid, quality, anchor, memory):
        boxes, frames = boxes[:, -self.history:], frames[:, -self.history:]
        valid, quality = valid[:, -self.history:], quality[:, -self.history:]
        reference = boxes[:, -1]
        values = relative_geometry(boxes, reference)
        time = (frames - frames[:, -1:]).float() / self.history
        observed = torch.cat((values, time[..., None], valid.float()[..., None]), -1)
        observed = observed * valid[..., None]
        trust = (quality * valid).sum(1) / valid.sum(1).clamp(min=1)
        inputs = torch.cat((observed.flatten(1), anchor.mean(1), memory.mean((1, 2)), trust[:, None]), -1)
        result = self.encoder(inputs)
        weights = result[:, -self.modes:].log_softmax(-1)
        parameters = result[:, :-self.modes].view(-1, self.modes, self.horizon, 8)
        center, scale = geometry(boxes)
        velocity = (center[:, -1] - center[:, -2]) / scale[:, -1]
        interval = (frames[:, -1] - frames[:, -2]).clamp(min=1)
        velocity = velocity / interval[:, None] * valid[:, -2, None]
        base = torch.zeros_like(parameters[..., :4])
        steps = torch.arange(1, self.horizon + 1, device=boxes.device)
        base[..., :2] = velocity[:, None, None] * steps[None, None, :, None]
        means = parameters[..., :4] + base
        deviations = F.softplus(parameters[..., 4:]) + .05
        return {'log_weights': weights, 'means': means, 'deviations': deviations,
                'reference': reference, 'history_trust': trust}

    @staticmethod
    def log_probability(distribution, coordinates, step=0):
        means = distribution['means'][:, :, step]
        deviation = distribution['deviations'][:, :, step]
        distance = (coordinates[:, :, None] - means[:, None]) / deviation[:, None]
        components = -.5 * (distance.square() + math.log(2 * math.pi)).sum(-1)
        components -= deviation.log().sum(-1)[:, None]
        return torch.logsumexp(components + distribution['log_weights'][:, None], -1)


class TemporalModules(nn.Module):
    def __init__(self, c1_checkpoint, slots=4, motion_history=8, modes=3, horizon=3):
        super().__init__()
        hidden = c1_checkpoint['args']['hidden']
        self.c1 = CandidateQualityHead(hidden)
        self.c1.load_state_dict(c1_checkpoint['head'], strict=True)
        self.c1.requires_grad_(False).eval()
        self.memory = DiscriminativeMemory(hidden, slots)
        self.memory.projection.load_state_dict(self.c1.projection.state_dict(), strict=True)
        self.motion = IdentityMotion(hidden, motion_history, modes, horizon)
        # Features, original evidence, two identity supports, motion density/trust.
        self.selector = nn.Sequential(nn.Linear(hidden + 14, hidden), nn.GELU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.c1.eval()
        return self

    def history_memory(self, data):
        anchor = self.memory.encode(data['anchor_features'].float())
        memory = self.memory.initialize(anchor)
        descriptors = self.memory.encode(data['history_descriptors'].float())
        for frame in range(descriptors.shape[1]):
            memory = self.memory.update(memory, descriptors[:, frame],
                                        data['history_evidence'][:, frame].float(),
                                        data['history_quality'][:, frame].float(),
                                        data['history_write'][:, frame] & data['history_valid'][:, frame])
        return anchor, memory, descriptors

    def decide(self, candidates, anchor, memory, history_boxes, history_frames,
               history_valid, history_quality):
        descriptors = self.memory.encode(candidates['modality_features'].float())
        support = self.memory.support(memory, anchor, descriptors)
        distribution = self.motion(history_boxes.float(), history_frames, history_valid,
                                   history_quality.float(), anchor, memory)
        coordinates = relative_geometry(candidates['image_boxes'].float(), distribution['reference'])
        motion_support = self.motion.log_probability(distribution, coordinates).clamp(-20, 10)
        with torch.no_grad():
            c1_quality = candidates['c1_quality'].float()
            c1_logits = torch.logit(c1_quality.clamp(1e-5, 1 - 1e-5))
            c1_features = self.c1.projection(candidates['features'].float())
        trust = distribution['history_trust'][:, None].expand_as(motion_support)
        inputs = torch.cat((c1_features, candidates['evidence'].float(), support,
                            motion_support[..., None], trust[..., None]), -1)
        residual = self.selector(inputs)
        current_logits, future_logits = c1_logits + residual[..., 0], c1_logits + residual[..., 1]
        cost = residual[..., 2]
        reconstructed = c1_logits.sigmoid()
        combined = c1_quality + .7 * (current_logits.sigmoid() - reconstructed) + .3 * (future_logits.sigmoid() - reconstructed)
        combined -= .1 * cost.clamp(min=0)
        scores = candidates['evidence'][..., 1] + (combined - candidates['raw_score']) * .55
        return {'scores': scores, 'current_logits': current_logits, 'future_logits': future_logits,
                'cost': cost, 'quality': combined, 'identity_support': support,
                'distribution': distribution, 'descriptors': descriptors,
                'c1_scores': candidates['evidence'][..., 1] + (c1_quality - candidates['raw_score']) * .55}

    def forward(self, data):
        anchor, memory, past = self.history_memory(data)
        result = self.decide(data, anchor, memory, data['history_boxes'], data['history_frames'],
                             data['history_valid'], data['history_quality'])
        result.update(anchor=anchor, memory=memory, past_descriptors=past)
        return result


def temporal_objective(model, output, data, rank_gap=.1):
    valid = data['valid']
    current, future = data['current_iou'].float(), data['future_iou'].float().mean(-1)
    cost = data['wrong_update_fraction'].float()
    current_loss = F.binary_cross_entropy_with_logits(output['current_logits'][valid], current[valid])
    future_loss = F.mse_loss(output['future_logits'].sigmoid()[valid], future[valid])
    cost_loss = F.mse_loss(output['cost'][valid], cost[valid])
    utility = .7 * current + .3 * future - .1 * cost
    differences = utility[:, :, None] - utility[:, None, :]
    pairs = valid[:, :, None] & valid[:, None, :] & (differences > rank_gap)
    score_differences = output['scores'][:, :, None] - output['scores'][:, None, :]
    ranking = (F.softplus(-score_differences) * differences.clamp(min=0) * pairs).sum() / pairs.sum().clamp(min=1)

    # Teacher uses only past target-quality labels; none enters model.forward.
    past = output['past_descriptors'].detach()
    teacher_sources = torch.cat((output['anchor'][:, None].detach(), past), 1)
    trustworthy = data['history_valid'] & (data['history_iou'] >= .5)
    trustworthy = torch.cat((torch.ones_like(trustworthy[:, :1]), trustworthy), 1)
    teacher = torch.einsum('bsmd,bkmd->bksm', teacher_sources, output['descriptors'].detach())
    teacher = teacher.masked_fill(~trustworthy[:, None, :, None], -torch.inf).max(2).values.mean(-1)
    support = output['identity_support'].mean(-1)
    teacher_gap = teacher[:, :, None] - teacher[:, None, :]
    student_gap = support[:, :, None] - support[:, None, :]
    memory_loss = ((student_gap - teacher_gap).square() * pairs).sum() / pairs.sum().clamp(min=1)
    identity_pairs = valid[:, :, None] & valid[:, None, :] & ((current[:, :, None] - current[:, None, :]) > .5)
    identity_loss = (F.relu(.2 - student_gap) * identity_pairs).sum() / identity_pairs.sum().clamp(min=1)

    target_motion = relative_geometry(data['motion_targets'].float(), output['distribution']['reference'])
    distribution = output['distribution']
    distance = (target_motion[:, None] - distribution['means']) / distribution['deviations']
    components = -.5 * (distance.square() + math.log(2 * math.pi)).sum((-1, -2))
    components -= distribution['deviations'].log().sum((-1, -2))
    motion_loss = -torch.logsumexp(components + distribution['log_weights'], -1).mean() / model.motion.horizon
    loss = .7 * current_loss + .3 * future_loss + .1 * cost_loss + ranking + .1 * (memory_loss + identity_loss + motion_loss)
    parts = {name: float(value.detach()) for name, value in
             (('current', current_loss), ('future', future_loss), ('cost', cost_loss),
              ('ranking', ranking), ('memory', memory_loss), ('identity', identity_loss), ('motion', motion_loss))}
    return loss, parts
