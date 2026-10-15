"""R2: frozen ViT extraction, trainable localization, and competitive memory prompts.

Tracking observes predictions only. Later annotations are consumed exclusively by
the separate loss functions and their causal, training-only compression teacher.
The original CleanCADTrack and R1 execution paths are unchanged.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .baseline import CleanCADTrack
from .data import crop_batch, xyxy, clip_boxes, iou
from .model import pool_rois
from .ported_modules import InstanceMemory
from lib.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import generate_heatmap


def dense_candidates(features, transform, reference, window):
    response = (features['score'] * window)[:, 0]
    suppressed = response.clone()
    axis = torch.arange(16, device=response.device)
    ys, xs = torch.meshgrid(axis, axis, indexing='ij')
    indices = []
    for _ in range(5):
        index = suppressed.flatten(1).argmax(-1)
        indices.append(index)
        near = (ys[None] - index[:, None, None] // 16).abs().maximum((xs[None] - index[:, None, None] % 16).abs()) <= 1
        suppressed = suppressed.masked_fill(near, -torch.inf)
    indices = torch.stack(indices, 1)
    gather = indices[:, None].expand(-1, 2, -1)
    sizes = features['size'].flatten(2).gather(2, gather).transpose(1, 2)
    offsets = features['offset'].flatten(2).gather(2, gather).transpose(1, 2)
    centers = (torch.stack((indices % 16, indices // 16), -1).float() + offsets) / 16
    normalized = torch.cat((centers - sizes / 2, centers + sizes / 2), -1)
    origin = reference[:, :2] + reference[:, 2:] / 2 - transform[:, 2:] / 2
    return dict(boxes=normalized * transform[:, None, 2:] + torch.cat((origin, origin), -1)[:, None],
        normalized=normalized, raw=features['score'][:, 0].flatten(1).gather(1, indices),
        hann=response.flatten(1).gather(1, indices),
        instance=pool_rois(features['modality'], normalized),
        fused=pool_rois(features['fused'][:, None], normalized).squeeze(2))


class CompetitiveMemoryPrompt(nn.Module):
    """Two target, one verified non-target and one uncertain slot per sensor."""
    def __init__(self, hidden=128):
        super().__init__()
        projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.memory = InstanceMemory(projection, hidden)
        self.output = nn.Linear(hidden * 3 + 3, 768)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def initialize(self, features):
        anchor = self.memory.encode(features)
        return dict(anchor=anchor, slots=self.memory.initialize(anchor))

    def forward(self, features, state):
        batch, sensors, channels, height, width = features.shape
        pixels = features.flatten(-2).permute(0, 3, 1, 2)
        query = self.memory.encode(pixels)
        positive = F.normalize(torch.cat((state['anchor'][:, None], state['slots'][:, :2]), 1), dim=-1)
        similarity = torch.einsum('bnmd,bsmd->bnms', query, positive)
        read = torch.einsum('bnms,bsmd->bnmd', (similarity * 10).softmax(-1), positive)
        context = F.normalize(state['slots'][:, 2:], dim=-1)
        other = context[:, 0, None].expand(-1, height * width, -1, -1)
        unknown = context[:, 1, None].expand_as(other)
        evidence = torch.stack((similarity.max(-1).values, (query * other).sum(-1),
                                (query * unknown).sum(-1)), -1)
        prompt = self.output(torch.cat((read, other, unknown, evidence), -1))
        prompt = prompt.permute(0, 2, 3, 1).reshape(batch, sensors, channels, height, width)
        return features + prompt

    def encode_candidates(self, candidate, state):
        descriptor = self.memory.encode(candidate['instance'])
        evidence = descriptor.new_zeros(*descriptor.shape[:2], 10)
        evidence[..., 0] = candidate['raw']
        anchor_similarity = (descriptor * state['anchor'][:, None]).sum(-1)
        evidence[..., 2], evidence[..., 4] = anchor_similarity.unbind(-1)
        gate = self.memory.gate_logits(descriptor, evidence, candidate['raw'], state['anchor'][:, None])
        support = self.memory.support(state['slots'], state['anchor'], descriptor)
        return dict(descriptor=descriptor, gate=gate, support=support)

    def advance(self, state, encoded, write):
        probability = encoded['gate'].softmax(-1)
        descriptor = encoded['descriptor']
        verified = probability[:, 0, :, 0] >= .5
        commit = write[:, None] & verified
        observed = torch.ones_like(write)
        rates = self.memory.write_rates(descriptor[:, 0], probability[:, 0], observed, commit)[:, :2]
        positive = state['slots'][:, :2] * (1 - rates[..., None]) + descriptor[:, None, 0] * rates[..., None]
        context = []
        category = probability.argmax(-1)
        rows = torch.arange(len(write), device=write.device)[:, None]
        sensors = torch.arange(2, device=write.device)[None]
        for label in (1, 2):
            eligible = category == label
            if label == 1:
                eligible = eligible & (probability[..., label] >= .5)
            confidence = probability[..., label] * eligible
            rate, index = confidence.max(1)
            value = descriptor[rows, index, sensors]
            old = state['slots'][:, label + 1]
            context.append(old * (1 - rate[..., None]) + value * rate[..., None])
        slots = torch.cat((positive, torch.stack(context, 1)), 1)
        return dict(anchor=state['anchor'], slots=slots), commit


class DenseCADTrack(nn.Module):
    def __init__(self, base: CleanCADTrack, use_memory: bool):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.base.decode_fuse_search.requires_grad_(True)
        self.base.box_head.requires_grad_(True)
        self.prompt = CompetitiveMemoryPrompt() if use_memory else None
        self.focal = FocalLoss()
        axis = .5 * (1 - torch.cos((2 * math.pi / 17) * torch.arange(1, 17).float()))
        self.register_buffer('window', axis.reshape(1, 1, -1, 1) * axis.reshape(1, 1, 1, -1), persistent=False)

    @property
    def device(self):
        return next(self.base.box_head.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        self.base.backbone.eval()
        return self

    @torch.no_grad()
    def extract(self, templates, search):
        tokens, _, _ = self.base.backbone(templates, search, track_query_before=None)
        rgb, tir = tokens.chunk(2, dim=1)
        rgb_x, tir_x = rgb[:, -256:], tir[:, -256:]
        return dict(rgb_x=rgb_x, tir_x=tir_x,
            modality=torch.stack((rgb_x, tir_x), 1).transpose(-1, -2).reshape(-1, 2, 768, 16, 16),
            anchor_tokens=torch.stack((rgb[:, 1:65], tir[:, 1:65]), 1).transpose(-1, -2).reshape(-1, 2, 768, 8, 8))

    def predict(self, templates, search, memory):
        features = self.extract(templates, search)
        if self.prompt is None:
            rgb_x, tir_x = features['rgb_x'], features['tir_x']
        else:
            prompted = self.prompt(features['modality'], memory)
            rgb_x, tir_x = prompted[:, 0].flatten(2).transpose(1, 2), prompted[:, 1].flatten(2).transpose(1, 2)
        fused = self.base.decode_fuse_search(torch.cat((rgb_x, tir_x), -1).transpose(1, 2).reshape(-1, 1536, 16, 16))
        score, _, size, offset = self.base.box_head(fused)
        return dict(score=score, size=size, offset=offset, fused=fused, modality=features['modality'])

    def initialize_memory(self, templates, search, boxes):
        features = self.extract(templates, search)
        side = torch.ceil(torch.sqrt(boxes[:, 2:].prod(-1)) * 2)
        size = boxes[:, 2:] / side[:, None]
        region = torch.cat((.5 - size / 2, .5 + size / 2), -1)[:, None]
        return self.prompt.initialize(pool_rois(features['anchor_tokens'], region).squeeze(1))

    def trainable_state(self):
        state = dict(fusion=self.base.decode_fuse_search.state_dict(), head=self.base.box_head.state_dict())
        if self.prompt is not None:
            state['prompt'] = self.prompt.state_dict()
        return state

    def load_trainable_state(self, state):
        assert set(state) == ({'fusion', 'head', 'prompt'} if self.prompt is not None else {'fusion', 'head'})
        self.base.decode_fuse_search.load_state_dict(state['fusion'], strict=True)
        self.base.box_head.load_state_dict(state['head'], strict=True)
        if self.prompt is not None:
            self.prompt.load_state_dict(state['prompt'], strict=True)

    def initialize_batch(self, images, boxes):
        template, _ = crop_batch(images, boxes, 2., 128, self.device)
        memory = None
        if self.prompt is not None:
            search, _ = crop_batch(images, boxes, 4., 256, self.device)
            memory = self.initialize_memory([template, template], search, boxes)
        return dict(templates=[template, template], box=boxes, age=0, memory=memory)

    def step_batch(self, images, state):
        patch, transform = crop_batch(images, state['box'], 4., 256, self.device)
        maps = self.predict(state['templates'], patch, state['memory'])
        candidate = dense_candidates(maps, transform, state['box'], self.window)
        box = clip_boxes(candidate['boxes'][:, 0], images).detach()
        age = state['age'] + 1
        write = (candidate['hann'][:, 0] > .65) & (age % 5 == 0)
        memory, encoded = state['memory'], None
        if self.prompt is not None:
            encoded = self.prompt.encode_candidates(candidate, memory)
            memory, commit = self.prompt.advance(memory, encoded, write)
        else:
            commit = write[:, None].expand(-1, 2)
        templates = list(state['templates'])
        if write.any():
            template, _ = crop_batch(images, box, 2., 128, self.device)
            mask = commit.repeat_interleave(3, -1)[:, :, None, None]
            templates[1] = torch.where(mask, template, templates[1])
        observation = dict(maps=maps, candidate=candidate, encoded=encoded, memory_before=state['memory'],
            reference=state['box'], transform=transform, box=box, template_commit=commit)
        return observation, dict(templates=templates, box=box, age=age, memory=memory)

    def localization_losses(self, observation, truth):
        valid = (truth[:, 2:] > 0).all(-1)
        transform, reference = observation['transform'], observation['reference']
        origin = reference[:, :2] + reference[:, 2:] / 2 - transform[:, 2:] / 2
        normalized = torch.cat(((truth[:, :2] - origin) / transform[:, 2:], truth[:, 2:] / transform[:, 2:]), -1)
        normalized = torch.where(valid[:, None], normalized, torch.zeros_like(normalized))
        heatmap = generate_heatmap(normalized.detach().cpu()[None], 256, 16)[0].to(self.device)
        maps = observation['maps']
        focal = self.focal(maps['score'][valid], heatmap[valid, None])
        # GT chooses the supervised regression cell only; tracking used prediction above.
        predicted = self.base.box_head.cal_bbox(heatmap[:, None], maps['size'], maps['offset'])
        predicted_xyxy = box_cxcywh_to_xyxy(predicted)
        target_xyxy = xyxy(normalized).clamp(0, 1)
        regression = valid & (heatmap.flatten(1).max(-1).values == 1)
        regression &= (target_xyxy[:, 2:] > target_xyxy[:, :2]).all(-1)
        weights = regression.float()
        giou, _ = generalized_box_iou(predicted_xyxy, target_xyxy)
        giou = ((1 - giou) * weights).sum() / weights.sum().clamp(min=1)
        l1 = (F.l1_loss(predicted_xyxy, target_xyxy, reduction='none').mean(-1) * weights).sum() / weights.sum().clamp(min=1)
        return dict(focal=focal, l1=l1, giou=giou), valid

    def memory_losses(self, observation, truth, valid, teacher_features, teacher_valid):
        encoded = observation['encoded']
        quality = iou(observation['candidate']['boxes'].detach(), xyxy(truth)[:, None])
        category = torch.where(quality >= .5, 0, torch.where(quality == 0, 1, 2)).long()
        gate = F.cross_entropy(encoded['gate'].flatten(0, 2), category[..., None].expand(-1, -1, 2).flatten(), reduction='none')
        gate = (gate.reshape(len(truth), -1, 2) * valid[:, None, None]).sum() / (valid.sum() * 10).clamp(min=1)
        pairs = (quality[:, :, None] - quality[:, None, :] > .1) & valid[:, None, None]
        support = encoded['support'][..., 0].mean(-1)
        gap = support[:, :, None] - support[:, None, :]
        ranking = (F.relu(.2 - gap) * pairs).sum() / pairs.sum().clamp(min=1)
        past = torch.cat(teacher_features, 1).detach()
        past_valid = torch.cat(teacher_valid, 1)
        teacher = torch.einsum('bsmd,bkmd->bksm', past, encoded['descriptor'].detach())
        teacher = teacher.masked_fill(~past_valid[:, None, :, None], -torch.inf).max(2).values.mean(-1)
        teacher_gap = teacher[:, :, None] - teacher[:, None, :]
        preservation = ((gap - teacher_gap).square() * pairs).sum() / pairs.sum().clamp(min=1)
        return dict(gate=gate, identity=ranking, preservation=preservation), quality[:, 0]

    def forward(self, clips, compute_loss=True):
        labels = torch.tensor([clip['boxes'].tolist() for clip in clips], device=self.device)
        state = self.initialize_batch([clip['images'][0] for clip in clips], labels[:, 0])
        steps = len(clips[0]['images']) - 1
        teacher_features, teacher_valid = [], []
        if self.prompt is not None:
            teacher_features = [state['memory']['anchor'][:, None].detach()]
            teacher_valid = [torch.ones(len(clips), 1, device=self.device, dtype=torch.bool)]
        losses, predictions, totals = [], [], {}
        for time in range(1, steps + 1):
            observation, state = self.step_batch([clip['images'][time] for clip in clips], state)
            predictions.append(observation['box'])
            if compute_loss:
                parts, valid = self.localization_losses(observation, labels[:, time])
                loss = parts['focal'] + 5 * parts['l1'] + 2 * parts['giou']
                if self.prompt is not None:
                    memory_parts, quality = self.memory_losses(observation, labels[:, time], valid, teacher_features, teacher_valid)
                    parts.update(memory_parts)
                    loss = loss + .2 * parts['gate'] + .1 * (parts['identity'] + parts['preservation'])
                    teacher_features.append(observation['encoded']['descriptor'][:, :1].detach())
                    teacher_valid.append(((quality >= .5) & valid)[:, None])
                losses.append(loss)
                for name, value in parts.items():
                    totals[name] = totals.get(name, 0.) + value.detach() / steps
        prediction = torch.stack(predictions, 1)
        valid = (labels[:, 1:steps + 1, 2:] > 0).all(-1)
        quality = iou(xyxy(prediction), xyxy(labels[:, 1:steps + 1]))
        sequence_iou = (quality * valid).sum(1) / valid.sum(1).clamp(min=1)
        return dict(loss=torch.stack(losses).mean() if compute_loss else prediction.new_zeros(()),
            parts=totals, predictions=prediction, sequence_iou=sequence_iou)
