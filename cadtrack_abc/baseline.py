from pathlib import Path
import sys

import torch
from torch import nn

NATIVE = Path(__file__).parent / 'vendor' / 'CADTrack'
sys.path.insert(0, str(NATIVE))
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.cadtrack.vit_care import vit_base_patch16_224
from lib.models.layers.head import build_box_head, conv


class CleanCADTrack(nn.Module):
    """Two independent shared-weight ViT streams plus the pretrained fusion/head.

    CDA, CSS, MCI and CAM are absent, as in the supplied checkpoint. All native
    parameter names are retained and loading must be strict.
    """
    def __init__(self, checkpoint):
        super().__init__()
        update_config_from_file(str(NATIVE / 'experiments/cadtrack/cadtrack.yaml'))
        self.backbone = vit_base_patch16_224('', drop_path_rate=0., add_cls_token=True)
        self.backbone.finetune_track(cfg, patch_start_index=1)
        self.box_head = build_box_head(cfg, 768)
        self.decode_fuse_search = conv(1536, 768)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.load_state_dict(payload['net'], strict=True)
        self.loaded_epoch = payload['epoch']
        self.loaded_tensors = len(payload['net'])
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, templates, search):
        tokens, _, lengths = self.backbone(templates, search, track_query_before=None)
        rgb, tir = tokens.chunk(2, dim=1)
        rgb_x, tir_x = rgb[:, -256:], tir[:, -256:]
        fused = self.decode_fuse_search(torch.cat((rgb_x, tir_x), -1).transpose(1, 2).reshape(-1, 1536, 16, 16))
        score, _, size, offset = self.box_head(fused)
        return {'score': score, 'size': size, 'offset': offset, 'fused': fused,
                'modality': torch.stack((rgb_x, tir_x), 1).transpose(-1, -2).reshape(-1, 2, 768, 16, 16),
                'anchor_tokens': torch.stack((rgb[:, 1:65], tir[:, 1:65]), 1).transpose(-1, -2).reshape(-1, 2, 768, 8, 8)}
