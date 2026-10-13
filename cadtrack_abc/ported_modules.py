"""A/B implementations ported from research/temporal_modules.py and recoverability_modules.py.
CAD features use a newly initialized projection; no GOLA head checkpoint is loaded.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def geometry(boxes):
    size = (boxes[..., 2:] - boxes[..., :2]).clamp(min=10.)
    return (boxes[..., :2] + boxes[..., 2:]) / 2, size


def relative_geometry(boxes, reference):
    center, size = geometry(boxes)
    origin, scale = geometry(reference)
    return torch.cat(((center - origin.unsqueeze(-2)) / scale.unsqueeze(-2),
                      torch.log(size / scale.unsqueeze(-2))), -1)


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


class InstanceMemory(nn.Module):
    """Four slots per sensor: two target, one non-target, one uncertain."""
    def __init__(self, projection, hidden=128):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.projection.load_state_dict(projection.state_dict(), strict=True)
        self.gate = nn.Sequential(nn.Linear(hidden + 4, 64), nn.GELU(), nn.Linear(64, 3))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        with torch.no_grad():
            self.gate[-1].bias[0] = 1.  # Initial target probability .576 > verification threshold.
        self.target_keys = nn.Parameter(torch.randn(2, 2, hidden) * .02)

    def encode(self, features):
        return F.normalize(self.projection(features.float()), dim=-1)

    def initialize(self, anchor):
        return torch.cat((anchor[:, None].expand(-1, 2, -1, -1).clone(),
                          torch.zeros_like(anchor[:, None]).expand(-1, 2, -1, -1).clone()), 1)

    def gate_logits(self, descriptor, evidence, quality, anchor):
        point_anchor = evidence[..., [2, 4]]
        roi_anchor = (descriptor * anchor).sum(-1)
        cues = torch.stack((evidence[..., 0, None].expand_as(roi_anchor),
                            quality[..., None].expand_as(roi_anchor), point_anchor, roi_anchor), -1)
        return self.gate(torch.cat((descriptor, cues), -1))

    def write_rates(self, descriptor, probability, observed, commit):
        assignment = torch.einsum('bmd,smd->bsm', descriptor, self.target_keys).softmax(1)
        target_rate = assignment * probability[..., 0][:, None] * commit[:, None]
        other_rate = probability[..., 1:].permute(0, 2, 1) * observed[:, None, None]
        category = probability.argmax(-1)
        context_class = torch.stack(((category == 1) & (probability[..., 1] >= .5), category == 2), 1)
        other_rate *= context_class
        return torch.cat((target_rate, other_rate), 1)

    def update(self, memory, descriptor, probability, observed, commit):
        rates = self.write_rates(descriptor, probability, observed, commit)
        return memory * (1 - rates[..., None]) + descriptor[:, None] * rates[..., None]

    @staticmethod
    def support(memory, anchor, descriptor):
        target = F.normalize(torch.cat((anchor[:, None], memory[:, :2]), 1), dim=-1)
        positive = torch.einsum('bsmd,bkmd->bksm', target, descriptor).max(2).values
        other = F.normalize(memory[:, 2:], dim=-1)
        context = torch.einsum('bsmd,bkmd->bkms', other, descriptor)
        return torch.cat((positive[..., None], context), -1)

    def history(self, data):
        anchor = self.encode(data['anchor_features'])
        past = self.encode(data['history_instance_descriptors'])
        memory = self.initialize(anchor)
        logits = []
        for frame in range(past.shape[1]):
            gate = self.gate_logits(past[:, frame], data['history_evidence'][:, frame].float(),
                                    data['history_quality'][:, frame].float(), anchor)
            observed = data['history_valid'][:, frame] & (data['history_frames'][:, frame] > 0)
            probability = gate.softmax(-1)
            verified = probability[..., 0] >= .5
            commit = observed[:, None] & data['history_write'][:, frame, None] & verified
            memory = self.update(memory, past[:, frame], probability, observed, commit)
            logits.append(gate)
        return anchor, memory, past, torch.stack(logits, 1)


