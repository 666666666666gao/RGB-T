"""Candidate relations within the actual local-plus-one-region visual budget."""
import torch
from torch import nn


class CandidateRelations(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # Existing candidate inputs: fused, two ROI descriptors and23 cues.
        self.embedding = nn.Sequential(nn.LayerNorm(hidden * 3 + 30),
                                       nn.Linear(hidden * 3 + 30, hidden), nn.GELU())
        self.attention = nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
        self.normalization = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 7)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs, valid):
        batch = len(inputs)
        source = torch.eye(7, device=inputs.device)[None, :, None].expand(batch, -1, 5, -1)
        tokens = self.embedding(torch.cat((inputs.reshape(batch, 7, 5, -1), source), -1))
        local = tokens[:, 0]
        local_context = self.attention(local, local, local, key_padding_mask=~valid[:, 0],
                                       need_weights=False)[0]
        local_result = self.normalization(local + local_context)
        # Each extra candidate sees only local candidates and its OWN region.
        # Six cached alternatives never attend to one another: deployment sees
        # at most one of them. The local result does not read unexecuted regions.
        extras = tokens[:, 1:].reshape(batch * 6, 5, -1)
        pairs = torch.cat((local[:, None].expand(-1, 6, -1, -1), tokens[:, 1:]), 2)
        padding = ~torch.cat((valid[:, :1].expand(-1, 6, -1), valid[:, 1:]), 2)
        extra_context = self.attention(extras, pairs.reshape(batch * 6, 10, -1),
                                       pairs.reshape(batch * 6, 10, -1),
                                       key_padding_mask=padding.reshape(batch * 6, 10),
                                       need_weights=False)[0]
        extra_result = self.normalization(extras + extra_context).reshape(batch, 30, -1)
        return self.output(torch.cat((local_result, extra_result), 1))
