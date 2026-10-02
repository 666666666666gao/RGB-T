"""C3 short-rollout utility head; future images/GT are training labels only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def motion_evidence(history, boxes, window):
    history = np.asarray(history)
    centers = (history[:, :2] + history[:, 2:]) / 2
    scale = np.maximum(history[-1, 2:] - history[-1, :2], 10.)
    # A single observation has no measured velocity; its history length marks this.
    velocity = np.diff(centers, axis=0) / scale if len(history) > 1 else np.zeros((1, 2))
    candidate_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    offset = (candidate_centers - centers[-1]) / scale
    relative_size = np.log(np.maximum(boxes[:, 2:] - boxes[:, :2], 1.) / scale)
    past = np.concatenate((velocity.mean(0), velocity[-1], [len(history) / window]))
    return np.concatenate((np.broadcast_to(past, (len(boxes), 5)), offset, relative_size), axis=1).astype(np.float32)


class BranchUtilityHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.scorer = nn.Sequential(nn.Linear(hidden + 19, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, features, evidence, motion, c1_quality):
        residual = self.scorer(torch.cat((self.projection(features), evidence, motion), -1)).squeeze(-1)
        return c1_quality + residual


def utility_selection_scores(prediction, evidence):
    # Match C1's .45 Hann policy when the utility residual is zero.
    return evidence[..., 1] + (prediction - evidence[..., 0]) * (1 - .45)


def utility_objective(prediction, target, valid, rank_scores, rank_weight=1., rank_gap=.1):
    regression = F.mse_loss(prediction[valid], target[valid])
    difference = target.unsqueeze(2) - target.unsqueeze(1)
    pair = valid.unsqueeze(2) & valid.unsqueeze(1) & (difference > rank_gap)
    ranking = F.softplus(-(rank_scores.unsqueeze(2) - rank_scores.unsqueeze(1))) * difference.clamp(min=0)
    return regression + rank_weight * (ranking * pair).sum() / pair.sum().clamp(min=1)
