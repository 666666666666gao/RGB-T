"""R3 temporal components; the running R2 H/HA implementation is unchanged.

Geometry history contains accepted observations only. Clock time still advances
when C defers geometry, so the forecast moves and its uncertainty grows.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .ported_modules import IdentityMotion, geometry, relative_geometry


class TrustedMotion(nn.Module):
    def __init__(self, hidden=128, history=8, modes=3, horizon=3):
        super().__init__()
        self.history, self.modes, self.horizon = history, modes, horizon
        self.encoder = nn.Sequential(nn.Linear(history * 7 + hidden * 2 + 2, hidden),
            nn.GELU(), nn.Linear(hidden, modes * horizon * 8 + modes))
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        with torch.no_grad():
            bias = self.encoder[-1].bias[:-modes].view(modes, horizon, 8)
            bias[..., 4:] = -1.5
            bias[0, :, 0] = -.3
            bias[2, :, 0] = .3

    def forward(self, boxes, frames, valid, quality, anchor, slots, current_frame):
        reference = boxes[:, -1]
        gap = current_frame - frames[:, -1]
        time = (frames - frames[:, -1:]).float() / self.history
        observed = torch.cat((relative_geometry(boxes, reference), time[..., None],
            quality[..., None], valid.float()[..., None]), -1) * valid[..., None]
        trust = (quality * valid).sum(1) / valid.sum(1).clamp(min=1)
        features = torch.cat((observed.flatten(1), anchor.mean(1), slots.mean((1, 2)),
            gap.float().log1p()[:, None], trust[:, None]), -1)
        prediction = self.encoder(features)
        parameters = prediction[:, :-self.modes].view(-1, self.modes, self.horizon, 8)
        center, size = geometry(boxes)
        interval = (frames[:, -1] - frames[:, -2]).clamp(min=1)
        velocity = (center[:, -1] - center[:, -2]) / size[:, -1] / interval[:, None]
        velocity = velocity * valid[:, -2, None]
        offsets = gap[:, None] + torch.arange(self.horizon, device=boxes.device)[None]
        base = torch.zeros_like(parameters[..., :4])
        base[..., :2] = velocity[:, None, None] * offsets[:, None, :, None]
        deviation = (F.softplus(parameters[..., 4:]) + .05).clamp(max=2.)
        deviation = deviation * offsets[:, None, :, None].float().sqrt()
        return dict(reference=reference, means=parameters[..., :4] + base,
            deviations=deviation, log_weights=prediction[:, -self.modes:].log_softmax(-1),
            history_trust=trust, observation_gap=gap)

    log_probability = staticmethod(IdentityMotion.log_probability)


class MotionGridPrompt(nn.Module):
    """Soft, zero-initialized prior before the existing Center head."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(7, 768, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features, distribution, transform, crop_reference):
        _, _, height, width = features.shape
        y, x = torch.meshgrid((torch.arange(height, device=features.device) + .5) / height,
            (torch.arange(width, device=features.device) + .5) / width, indexing='ij')
        grid = torch.stack((x, y), -1)[None]
        crop_center = crop_reference[:, :2] + crop_reference[:, 2:] / 2
        side = transform[:, 2]
        points = crop_center[:, None, None] - side[:, None, None, None] / 2
        points = points + grid * side[:, None, None, None]
        origin, scale = geometry(distribution['reference'])
        mean = origin[:, None] + distribution['means'][:, :, 0, :2] * scale[:, None]
        deviation = distribution['deviations'][:, :, 0, :2] * scale[:, None]
        distance = (points[:, None] - mean[:, :, None, None]) / deviation[:, :, None, None]
        prior = torch.exp(-.5 * distance.square().sum(-1))
        prior = prior * distribution['log_weights'].exp()[:, :, None, None]
        # No normalization over the crop: outside modes keep low in-crop mass.
        log_size = (distribution['means'][:, :, 0, 2:] *
            distribution['log_weights'].exp()[..., None]).sum(1)
        trust = distribution['history_trust']
        extra = torch.cat((log_size, trust[:, None],
            distribution['observation_gap'].float().log1p()[:, None]), -1)
        extra = extra[:, :, None, None].expand(-1, -1, height, width)
        return features + self.projection(torch.cat((prior, extra), 1)) * trust[:, None, None, None]


def motion_objective(distribution, future_boxes, valid):
    target = relative_geometry(future_boxes, distribution['reference'])
    distance = (target[:, None] - distribution['means']) / distribution['deviations']
    density = -.5 * (distance.square() + math.log(2 * math.pi)).sum(-1)
    density -= distribution['deviations'].log().sum(-1)
    density = (density * valid[:, None]).sum(-1) + distribution['log_weights']
    count = valid.sum(-1)
    loss = -torch.logsumexp(density, -1) / count.clamp(min=1)
    return (loss * (count > 0)).sum() / (count > 0).sum().clamp(min=1)


def accept_geometry_history(state, boxes, frame, quality, commit):
    """A deferred output never enters trusted position/velocity history."""
    replacements = dict(
        history=torch.cat((state['history'][:, 1:], boxes.detach()[:, None]), 1),
        frames=torch.cat((state['frames'][:, 1:], frame[:, None]), 1),
        valid=torch.cat((state['valid'][:, 1:], torch.ones_like(commit[:, None])), 1),
        quality=torch.cat((state['quality'][:, 1:], quality.detach()[:, None]), 1))
    return {name: torch.where(commit.reshape(-1, *([1] * (value.ndim - 1))), value, state[name])
            for name, value in replacements.items()}
