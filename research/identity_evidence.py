"""Fixed trained identity evidence in the existing causal ABC decision.

This tests transfer of the paired projection to continuous tracking. Original
A memory, B motion/search and C write/risk rules stay loaded and active. The
new projection reads jointly attended streams, not independent sensors.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .identity_representation_probe import first_context
from .recoverability_modules import RecoverabilityModules
from .recoverability_tracker import RecoverabilityTracker


class IdentityEvidenceModules(RecoverabilityModules):
    def configure_identity(self, path, weight):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['args']['source'] == 'encoded'
        assert checkpoint['epoch'] == 9 and weight >= 0
        device = next(self.parameters()).device
        self.identity_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU()).to(device)
        self.identity_projection.load_state_dict(checkpoint['projection'], strict=True)
        self.identity_projection.eval().requires_grad_(False)
        self.identity_weight = weight
        self.identity_checkpoint_epoch = checkpoint['epoch']
        assert not self.post_search_bidirectional

    def identity_encode(self, features):
        # Exactly the FP16 cache round-trip followed by F32 projection in fit.
        return F.normalize(self.identity_projection(features.half().float()), dim=-1)

    def decide(self, data, anchor, memory):
        output = super().decide(data, anchor, memory)
        descriptor = self.identity_encode(data['encoded_instance_features'])
        support = (descriptor * data['encoded_identity_anchor'][:, None, None]).sum(-1).mean(-1)
        keep = support[torch.arange(len(support), device=support.device), 0, data['original_choice']]
        residual = self.identity_weight * (support - keep[:, None, None])
        if self.identity_weight:
            output['scores'] = output['scores'] + residual[..., None]
        # Search heads, write verification and regular/pause score differences
        # are unchanged. Invalid/unexecuted candidates remain masked by C.
        output['semantic_identity_support'] = support
        output['semantic_identity_residual'] = residual
        self.last_identity_support = support
        self.last_identity_residual = residual
        return output


class IdentityEvidenceTracker(RecoverabilityTracker):
    observation_fields = RecoverabilityTracker.observation_fields + ('encoded_instance_features',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.autocast('cuda', dtype=torch.float16):
            initial = self.extractor(first_context(self.anchor[None], self.anchor_mask))
        # Query-conditioned template tokens never replace this protected anchor.
        self.encoded_identity_anchor = self.modules.identity_encode(initial['encoded_template_features'])

    def original_inputs(self, observation, distribution, history):
        data = super().original_inputs(observation, distribution, history)
        data['encoded_identity_anchor'] = self.encoded_identity_anchor
        return data

    @torch.inference_mode()
    def step(self, image):
        result = super().step(image)
        self.last_decision['semantic_identity_support'] = self.modules.last_identity_support[0]
        self.last_decision['semantic_identity_residual'] = self.modules.last_identity_residual[0]
        return result
