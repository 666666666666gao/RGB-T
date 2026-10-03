"""C1: learn candidate quality/ranking; bounded online recovery is a later stage.

Uses the complete pretrained GOLA-B, frozen throughout. No future observations
or GT features are passed to the learned scorer. GT only supplies crop sampling
and training/held-out diagnostic labels, as in supervised tracker training.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped
from trackit.core.utils.siamfc_cropping import (
    prepare_siamfc_cropping_with_augmentation, apply_siamfc_cropping,
    apply_siamfc_cropping_to_boxes,
)
from trackit.core.utils.bbox_mask_gen import get_foreground_bounding_box
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.models.backbone.dinov2.builder import build_dino_v2_backbone
from trackit.models.methods.GOLA.gola_full_finetune import GOLABaseline_DINOv2


def box_iou(boxes, target):
    lt = torch.maximum(boxes[..., :2], target[..., :2])
    rb = torch.minimum(boxes[..., 2:], target[..., 2:])
    intersection = (rb - lt).clamp(min=0).prod(-1)
    area = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0).prod(-1)
    target_area = (target[..., 2:] - target[..., :2]).clamp(min=0).prod(-1)
    return intersection / (area + target_area - intersection).clamp(min=1e-8)


def crop_frame(frame, context_box, output_size, area_factor, scale_jitter, translation_jitter, rng):
    images = [torch.from_numpy(np.array(Image.open(p).convert('RGB'), copy=True)).permute(2, 0, 1)
              for p in frame.get_image_path()]
    image = torch.cat(images, dim=0).float()
    size = np.array((output_size, output_size))
    params, _ = prepare_siamfc_cropping_with_augmentation(
        context_box, area_factor, size, scale_jitter, translation_jitter, rng)
    crop, _, params = apply_siamfc_cropping(image, size, params, 'bilinear', False)
    crop = get_dataset_norm_stats_transform('mm', inplace=True)(crop / 255.)
    return crop, params


def foreground_mask(box, params):
    bounds = get_foreground_bounding_box(box, params, (14., 14.)).clip(0, 8)
    mask = torch.zeros((8, 8), dtype=torch.long)
    mask[bounds[1]:bounds[3], bounds[0]:bounds[2]] = 1
    return mask


class CandidateTrainingData(torch.utils.data.Dataset):
    def __init__(self, root, cache, sequence_indices, samples, seed, max_gap,
                 contamination_probability, translation_jitter):
        self.data = MultiModalObjectTrackingDataset_MemoryMapped.load(root, cache)
        self.sequence_indices = sequence_indices
        self.samples, self.seed, self.max_gap = samples, seed, max_gap
        self.contamination_probability = contamination_probability
        self.translation_jitter = translation_jitter
        self.epoch = 0
        self.valid_frames = {}
        for i in sequence_indices:
            boxes = self.data[i].get_all_bounding_boxes()
            valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
            self.valid_frames[i] = np.flatnonzero(valid)
        # A training sample requires a valid first-frame identity and a past frame.
        self.sequence_indices = [i for i in sequence_indices
                                 if len(self.valid_frames[i]) >= 2 and self.valid_frames[i][0] == 0]
        assert self.sequence_indices

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, index)))
        sequence_index = self.sequence_indices[int(rng.integers(len(self.sequence_indices)))]
        seq = self.data[sequence_index]
        valid = self.valid_frames[sequence_index]
        t_pos = int(rng.integers(1, len(valid)))
        current = int(valid[t_pos])
        previous = int(valid[int(rng.integers(max(0, t_pos - self.max_gap), t_pos))])
        z_box = seq[0].get_bounding_box().copy()
        d_box = seq[previous].get_bounding_box().copy()
        z, z_params = crop_frame(seq[0], z_box, 112, 2., 0., 0., rng)
        contaminated = bool(rng.random() < self.contamination_probability)
        d_translation = 1.5 if contaminated else 0.
        d, d_params = crop_frame(seq[previous], d_box, 112, 2., 0., d_translation, rng)
        # A contaminated template is treated as if the tracker believed the crop
        # centre was its target; GT is not supplied as its foreground mask.
        d_mask = (foreground_mask(d_box, d_params) if not contaminated
                  else torch.nn.functional.pad(torch.ones((4, 4), dtype=torch.long), (2, 2, 2, 2)))
        x, x_params = crop_frame(seq[current], d_box, 224, 4., .25, self.translation_jitter, rng)
        gt = apply_siamfc_cropping_to_boxes(seq[current].get_bounding_box().copy(), x_params) / 224.
        return {'z': z, 'd': d, 'x': x, 'z_feat_mask': foreground_mask(z_box, z_params),
                'd_feat_mask': d_mask, 'gt': torch.tensor(gt, dtype=torch.float32),
                'contaminated': contaminated}


class FrozenCandidateExtractor(nn.Module):
    def __init__(self, checkpoint, candidates=5, window_penalty=.45, nms_iou=.7, proposal_policy='peaks'):
        super().__init__()
        assert candidates >= 2
        assert proposal_policy in ('peaks', 'dense_hann5', 'dense_raw5')
        backbone = build_dino_v2_backbone('ViT-B/14', load_pretrained=True, acc='none')
        self.base = GOLABaseline_DINOv2(backbone, (8, 8), (16, 16))
        result = self.base.load_state_dict_from_file(checkpoint)
        # Backbone parameters come from DINOv2; checkpoint contains GOLA adapters/head.
        # Check the actual load result so a wrong checkpoint cannot silently skip the head.
        assert not result.unexpected_keys, result.unexpected_keys
        assert not any(k.startswith(('head.', 'token_type_embed')) for k in result.missing_keys), result.missing_keys
        self.load_receipt = {'checkpoint': checkpoint, 'missing_keys_from_adapter_checkpoint': result.missing_keys,
                             'unexpected_keys': result.unexpected_keys, 'dinov2_pretrained': True}
        self.base.requires_grad_(False)
        self.base.eval()
        self.candidates, self.window_penalty, self.nms_iou = candidates, window_penalty, nms_iou
        self.proposal_policy = proposal_policy
        window = torch.outer(torch.hann_window(16, periodic=False), torch.hann_window(16, periodic=False))
        self.register_buffer('window', window.flatten())

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, batch):
        return self.extract_with_output(batch)[0]

    @torch.no_grad()
    def extract_with_output(self, batch):
        z, d, x = batch['z'], batch['d'], batch['x']
        zm, dm = batch['z_feat_mask'], batch['d_feat_mask']
        zv, zi = self.base._z_feat(z, zm)
        dv, di = self.base._d_feat(d, dm)
        xv, xi = self.base._x_feat(x)
        fused = self.base._fusion(zv, xv, zi, xi, dv, di)
        output = self.base.head(fused)
        raw = output['score_map'].float().sigmoid().flatten(1)
        ranked = raw * (1 - self.window_penalty) + self.window * self.window_penalty
        boxes = output['boxes'].flatten(1, 2).float()
        indices = torch.zeros((x.shape[0], self.candidates), dtype=torch.long, device=x.device)
        valid = torch.zeros_like(indices, dtype=torch.bool)
        peaks = output['score_map'] == F.max_pool2d(output['score_map'].unsqueeze(1), 3, 1, 1).squeeze(1)
        if self.proposal_policy != 'peaks':
            peaks = torch.ones_like(peaks, dtype=torch.bool)
        proposal_scores = raw if self.proposal_policy == 'dense_raw5' else ranked
        for b in range(x.shape[0]):
            # Always retain the original Hann winner; dense controls only relax peaks.
            winner = int(ranked[b].argmax())
            order = proposal_scores[b].masked_fill(~peaks[b].flatten(), -torch.inf).argsort(descending=True)
            selected = [winner]
            for candidate in order.tolist():
                if not bool(peaks[b].flatten()[candidate]):
                    break
                if all(max(abs(candidate // 16 - j // 16), abs(candidate % 16 - j % 16)) >= 2
                       for j in selected) and bool((box_iou(boxes[b, selected], boxes[b, candidate]) < self.nms_iou).all()):
                    selected.append(candidate)
                if len(selected) == self.candidates:
                    break
            indices[b, :len(selected)] = torch.tensor(selected, device=x.device)
            valid[b, :len(selected)] = True
        gather = indices.unsqueeze(-1).expand(-1, -1, fused.shape[-1])
        features = fused.gather(1, gather).float()
        selected_boxes = boxes.gather(1, indices.unsqueeze(-1).expand(-1, -1, 4))
        # These identity descriptors use pre-attention patch features separately
        # for each sensor. Jointly attended tokens are never called RGB/TIR evidence.
        sims, modality_features, anchor_features = [], [], []
        for channel in (slice(0, 3), slice(3, 6)):
            z_raw = self.base.patch_embed(z[:, channel])
            d_raw = self.base.patch_embed(d[:, channel])
            x_raw = self.base.patch_embed(x[:, channel])
            z_weights, d_weights = zm.flatten(1).float(), dm.flatten(1).float()
            z_anchor = (z_raw * z_weights.unsqueeze(-1)).sum(1) / z_weights.sum(1, keepdim=True).clamp(min=1)
            d_anchor = (d_raw * d_weights.unsqueeze(-1)).sum(1) / d_weights.sum(1, keepdim=True).clamp(min=1)
            candidate_raw = x_raw.gather(1, gather)
            modality_features.append(candidate_raw.float())
            anchor_features.append(z_anchor.float())
            sims.extend((F.cosine_similarity(candidate_raw, z_anchor.unsqueeze(1), dim=-1),
                         F.cosine_similarity(candidate_raw, d_anchor.unsqueeze(1), dim=-1)))
        selected_raw = raw.gather(1, indices)
        evidence = torch.cat((selected_raw.unsqueeze(-1), ranked.gather(1, indices).unsqueeze(-1),
                              torch.stack(sims, dim=-1), selected_boxes), dim=-1)
        return {'features': features, 'evidence': evidence, 'raw_score': selected_raw,
                'boxes': selected_boxes, 'valid': valid,
                'modality_features': torch.stack(modality_features, dim=2),
                'anchor_features': torch.stack(anchor_features, dim=1)}, output


class CandidateQualityHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.scorer = nn.Sequential(nn.Linear(hidden + 10, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, candidates):
        residual = self.scorer(torch.cat((self.projection(candidates['features']), candidates['evidence']), -1)).squeeze(-1)
        return torch.logit(candidates['raw_score'].clamp(1e-5, 1 - 1e-5)) + residual


def selection_scores(logits, candidates, window_penalty):
    # Retain GOLA's original Hann penalty and exact initial winner ordering.
    return candidates['evidence'][..., 1] + (logits.sigmoid() - candidates['raw_score']) * (1 - window_penalty)


def candidate_objective(logits, candidates, gt, rank_weight, rank_gap):
    valid = candidates['valid']
    quality = box_iou(candidates['boxes'], gt.unsqueeze(1))
    quality_loss = F.binary_cross_entropy_with_logits(logits[valid], quality[valid])
    delta = quality.unsqueeze(2) - quality.unsqueeze(1)
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1) & (delta > rank_gap)
    ranking = F.softplus(-(logits.unsqueeze(2) - logits.unsqueeze(1))) * delta.clamp(min=0)
    rank_loss = (ranking * pair_mask).sum() / pair_mask.sum().clamp(min=1)
    return quality_loss + rank_weight * rank_loss, quality
