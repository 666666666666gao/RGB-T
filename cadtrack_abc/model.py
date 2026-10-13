"""Online CAD adapter: bounded instance memory, multimodal motion and risk selection.

Only the initial box enters initialization. step() has no annotation argument.
Labels and future boxes are consulted separately in training_losses().
"""
import torch
from torch import nn
from torch.nn import functional as F

from .baseline import CleanCADTrack
from .data import crop_batch, xyxy, xywh, clip_boxes, iou
from .ported_modules import InstanceMemory, IdentityMotion, relative_geometry, geometry


def pool_rois(features, boxes):
    """3x3 spatial sampling of each normalized box; sensors remain separate."""
    batch, sensors, channels, height, width = features.shape
    centers = (boxes[..., :2] + boxes[..., 2:]) / 2
    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1 / height)
    axis = boxes.new_tensor([-.3333, 0., .3333])
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    lattice = torch.stack((xx, yy), -1).reshape(1, 1, 9, 2)
    grid = (centers[:, :, None] + sizes[:, :, None] * lattice) * 2 - 1
    count = boxes.shape[1]
    grid = grid[:, None].expand(-1, sensors, -1, -1, -1).reshape(batch * sensors, count * 9, 1, 2)
    sampled = F.grid_sample(features.flatten(0, 1), grid, align_corners=False, padding_mode='zeros')
    return sampled.reshape(batch, sensors, channels, count, 9).mean(-1).permute(0, 3, 1, 2)


def candidates(features, transform, reference):
    score = features['score'][:, 0]
    axis = torch.arange(1, 17, device=score.device, dtype=score.dtype)
    window = .5 * (1 - torch.cos(2 * torch.pi * axis / 17))
    hann = window[:, None] * window[None, :]
    response = score * hann
    suppressed = response.clone()
    ys, xs = torch.meshgrid(torch.arange(16, device=score.device), torch.arange(16, device=score.device), indexing='ij')
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
    centers = torch.stack((indices % 16, indices // 16), -1) / 16 + offsets / 16
    normalized = torch.cat((centers - sizes / 2, centers + sizes / 2), -1)
    # Native tracker maps relative to the preceding box center, half the crop side.
    origin = reference[:, :2] + reference[:, 2:] / 2 - transform[:, 2:] / 2
    boxes = normalized * transform[:, None, 2:] + torch.cat((origin, origin), -1)[:, None]
    return {'boxes': boxes, 'normalized': normalized,
            'raw': score.flatten(1).gather(1, indices),
            'hann': response.flatten(1).gather(1, indices),
            'instance': pool_rois(features['modality'], normalized),
            'fused': pool_rois(features['fused'][:, None], normalized).squeeze(2)}


class CADModules(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.memory = InstanceMemory(projection, hidden)
        self.motion = IdentityMotion(hidden=hidden, history=8, modes=3, horizon=3)
        self.fused = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.embedding = nn.Sequential(nn.Linear(hidden * 3 + 15, hidden), nn.GELU())
        self.relations = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.)
        self.normalization = nn.LayerNorm(hidden)
        self.action = nn.Linear(hidden, 4)
        # Local-only search decision: candidate summary + causal future modes.
        self.search = nn.Sequential(nn.Linear(hidden * 2 + 39, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.action.weight)
        nn.init.zeros_(self.action.bias)
        nn.init.zeros_(self.search[-1].weight)
        nn.init.zeros_(self.search[-1].bias)
        with torch.no_grad():
            self.action.bias[1] = -3.
            self.action.bias[2] = 3.
            self.search[-1].bias[0] = -3.

    def encode(self, candidate, state, distribution, source):
        descriptors = self.memory.encode(candidate['instance'])
        support = self.memory.support(state['memory'], state['anchor'], descriptors)
        coordinates = relative_geometry(candidate['boxes'], distribution['reference'])
        probability = self.motion.log_probability(distribution, coordinates).clamp(-20, 10)
        trust = distribution['history_trust'][:, None].expand_as(probability)
        inputs = torch.cat((self.fused(candidate['fused']), descriptors.flatten(-2), support.flatten(-2),
                            coordinates, candidate['raw'][..., None], candidate['hann'][..., None],
                            probability[..., None], trust[..., None], torch.full_like(probability[..., None], source)), -1)
        token = self.embedding(inputs)
        evidence = token.new_zeros(len(token), token.shape[1], 10)
        evidence[..., 0] = candidate['raw']
        evidence[..., 1] = candidate['hann']
        evidence[..., 2] = (descriptors[:, :, 0] * state['anchor'][:, None, 0]).sum(-1)
        evidence[..., 4] = (descriptors[:, :, 1] * state['anchor'][:, None, 1]).sum(-1)
        gate = self.memory.gate_logits(descriptors, evidence, candidate['raw'], state['anchor'][:, None])
        return {'tokens': token, 'descriptors': descriptors, 'support': support, 'gate': gate, 'evidence': evidence}

    def score(self, encoded, candidate):
        tokens = encoded['tokens']
        context = self.relations(tokens, tokens, tokens, need_weights=False)[0]
        action = self.action(self.normalization(tokens + context))
        quality = torch.logit(candidate['raw'].clamp(1e-5, 1 - 1e-5)) + action[..., 0]
        advantage = action[..., 3].tanh()
        risk = action[..., 1].sigmoid()
        score = candidate['hann'] + .55 * (quality.sigmoid() - candidate['raw'])
        score += .25 * (advantage - advantage[:, :1]) - .05 * (risk - risk[:, :1])
        return dict(encoded, quality=quality, risk=action[..., 1], geometry=action[..., 2], advantage=advantage, score=score)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('CADTrackABC owns the complete causal clip forward.')


class CADTrackABC(nn.Module):
    def __init__(self, checkpoint, threshold=.03):
        super().__init__()
        self.base = CleanCADTrack(checkpoint)
        self.abc = CADModules()
        self.threshold = threshold

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @property
    def device(self):
        return next(self.abc.parameters()).device

    def initialize(self, images, initial_boxes, frames):
        boxes = initial_boxes.to(self.device)
        template, _ = crop_batch(images, boxes, 2., 128, self.device)
        search, _ = crop_batch(images, boxes, 4., 256, self.device)
        with torch.no_grad():
            output = self.base([template, template], search)
        # First template target region respects target aspect ratio.
        side = torch.ceil(torch.sqrt(boxes[:, 2:].prod(-1)) * 2)
        wh = boxes[:, 2:] / side[:, None]
        normalized = torch.cat((.5 - wh / 2, .5 + wh / 2), -1)[:, None]
        anchor = self.abc.memory.encode(pool_rois(output['anchor_tokens'], normalized).squeeze(1))
        history = xyxy(boxes)[:, None].expand(-1, 8, -1).clone()
        times = frames.to(self.device)[:, None].expand(-1, 8).clone()
        valid = torch.zeros(len(boxes), 8, device=self.device, dtype=torch.bool)
        valid[:, -1] = True
        return {'templates': [template, template], 'box': boxes, 'anchor': anchor,
                'memory': self.abc.memory.initialize(anchor), 'history': history, 'frames': times,
                'valid': valid, 'quality': valid.float(), 'age': torch.zeros(len(boxes), device=self.device, dtype=torch.long)}

    def step(self, images, frames, state, training_candidates=False):
        modules = self.abc
        distribution = modules.motion(state['history'], state['frames'], state['valid'], state['quality'], state['anchor'], state['memory'])
        local_patch, transform = crop_batch(images, state['box'], 4., 256, self.device)
        local = candidates(self.base(state['templates'], local_patch), transform, state['box'])
        local_encoded = modules.encode(local, state, distribution, 0.)
        local_scored = modules.score(local_encoded, local)
        summary = torch.cat((local_encoded['tokens'].mean(1), local_encoded['tokens'].max(1).values,
                             distribution['means'].flatten(1), distribution['log_weights']), -1)
        search_logits = modules.search(summary)
        trigger = ((search_logits[:, 0].sigmoid() > .5) | (local_scored['quality'][:, 0].sigmoid() < .5)) & (search_logits[:, 1].tanh() > self.threshold + .01)
        pair, scored = local, local_scored
        extra_count = 0
        if training_candidates or trigger.any():
            rows = torch.arange(len(images), device=self.device)
            modes = distribution['log_weights'].argmax(-1)
            target = distribution['means'][rows, modes, 0].detach()
            center, size = geometry(distribution['reference'])
            center = center + target[:, :2] * size
            size = size * target[:, 2:].clamp(-1.3863, 1.3863).exp()
            extra_box = clip_boxes(torch.cat((center - size / 2, center + size / 2), -1), images)
            extra_patch, extra_transform = crop_batch(images, extra_box, 6., 256, self.device)
            extra = candidates(self.base(state['templates'], extra_patch), extra_transform, extra_box)
            pair = {key: torch.cat((local[key], extra[key]), 1) for key in local}
            extra_encoded = modules.encode(extra, state, distribution, 1.)
            encoded = {key: torch.cat((local_encoded[key], extra_encoded[key]), 1) for key in local_encoded}
            scored = modules.score(encoded, pair)
            extra_count = len(images)
        scores = scored['score'].clone()
        scores[:, :5] = torch.where(trigger[:, None], scores[:, :5], local_scored['score'])
        if scores.shape[1] == 10:
            scores[:, 5:] = scores[:, 5:].masked_fill(~trigger[:, None], -torch.inf) - .01
        winner = scores.argmax(-1)
        best = scores.gather(1, winner[:, None]).squeeze(1)
        choice = torch.where(best > scores[:, 0] + self.threshold, winner, torch.zeros_like(winner))
        rows = torch.arange(len(images), device=self.device)
        chosen_box = clip_boxes(pair['boxes'][rows, choice], images).detach()
        # Use exactly the context legally observed by this row's decision.
        selected_quality = scored['quality'][rows, choice]
        selected_risk = scored['risk'][rows, choice]
        selected_geometry = scored['geometry'][rows, choice]
        selected_gate = scored['gate'][rows, choice]
        selected_descriptor = scored['descriptors'][rows, choice]
        local_choice = choice.clamp(max=4)
        selected_quality = torch.where(trigger, selected_quality, local_scored['quality'][rows, local_choice])
        selected_risk = torch.where(trigger, selected_risk, local_scored['risk'][rows, local_choice])
        selected_geometry = torch.where(trigger, selected_geometry, local_scored['geometry'][rows, local_choice])
        selected_gate = torch.where(trigger[:, None, None], selected_gate, local_scored['gate'][rows, local_choice])
        selected_descriptor = torch.where(trigger[:, None, None], selected_descriptor, local_scored['descriptors'][rows, local_choice])
        probability = selected_gate.softmax(-1)
        geometry_commit = selected_geometry.sigmoid() >= .5
        predicted_center = center if extra_count else geometry(distribution['reference'])[0] + distribution['means'][rows, distribution['log_weights'].argmax(-1), 0, :2].detach() * geometry(distribution['reference'])[1]
        predicted_size = geometry(distribution['reference'])[1]
        propagated = clip_boxes(torch.cat((predicted_center - predicted_size / 2, predicted_center + predicted_size / 2), -1), images)
        next_box = torch.where(geometry_commit[:, None], chosen_box, propagated).detach()
        state['age'] = state['age'] + 1
        raw = pair['raw'][rows, choice]
        write = (state['age'] % 5 == 0) & (raw > .65) & (selected_risk.sigmoid() < .5)
        verified = probability[..., 0] >= .5
        # RGB/TIR templates are committed independently, preserving the first one.
        if write.any():
            new_template, _ = crop_batch(images, chosen_box, 2., 128, self.device)
            masks = torch.cat((verified[:, 0:1].expand(-1, 3), verified[:, 1:2].expand(-1, 3)), 1) & write[:, None]
            state['templates'][1] = torch.where(masks[:, :, None, None], new_template, state['templates'][1])
        observed = torch.ones(len(images), device=self.device, dtype=torch.bool)
        identity_commit = verified & (selected_quality.sigmoid()[:, None] >= .5) & (selected_risk.sigmoid()[:, None] < .5)
        state['memory'] = modules.memory.update(state['memory'], selected_descriptor, probability, observed, identity_commit)
        state['history'] = torch.cat((state['history'][:, 1:], xyxy(next_box)[:, None]), 1)
        state['frames'] = torch.cat((state['frames'][:, 1:], frames.to(self.device)[:, None]), 1)
        state['valid'] = torch.cat((state['valid'][:, 1:], geometry_commit[:, None]), 1)
        state['quality'] = torch.cat((state['quality'][:, 1:], selected_quality.sigmoid().detach()[:, None]), 1)
        state['box'] = next_box
        return {'box': chosen_box, 'choice': choice, 'local': local, 'local_scored': local_scored,
                'pair': pair, 'scored': scored, 'distribution': distribution, 'search_logits': search_logits,
                'extra_forwards': extra_count, 'geometry_commit': geometry_commit, 'write': write,
                'template_commit': write[:, None] & verified, 'target_memory_commit': identity_commit,
                'geometry_reference': next_box, 'keep_box': clip_boxes(local['boxes'][:, 0], images),
                'selected_quality': selected_quality.sigmoid().detach()}, state

    def training_losses(self, observation, targets, future, future_valid, state):
        current = iou(observation['pair']['boxes'], targets[:, None])
        local_iou = current[:, :5]
        valid = (targets[:, 2:] > targets[:, :2]).all(-1).float()
        weights = valid[:, None]
        scored = observation['scored']
        quality_loss = (F.binary_cross_entropy_with_logits(scored['quality'], current, reduction='none') * weights).mean()
        quality_loss += (F.binary_cross_entropy_with_logits(observation['local_scored']['quality'], local_iou, reduction='none') * weights).mean()
        difference = current - current[:, :1]
        advantage = (F.smooth_l1_loss(scored['advantage'] - scored['advantage'][:, :1], difference, reduction='none') * weights).mean()
        gap = current[:, :, None] - current[:, None, :]
        pairs = (gap > .1) * valid[:, None, None]
        ranking = (F.relu(.03 - (scored['score'][:, :, None] - scored['score'][:, None, :])) * gap.clamp(min=0) * pairs).sum() / pairs.sum().clamp(min=1)
        risk_loss = (F.binary_cross_entropy_with_logits(scored['risk'], (current < .2).float(), reduction='none') * weights).mean()
        geometry_loss = (F.binary_cross_entropy_with_logits(scored['geometry'], (current >= .5).float(), reduction='none') * weights).mean()
        categories = torch.where(current >= .5, 0, torch.where(current == 0, 1, 2)).long()
        gate_ce = F.cross_entropy(scored['gate'].flatten(0, 2), categories[..., None].expand(-1, -1, 2).flatten(), reduction='none').reshape(len(current), -1, 2)
        gate_loss = (gate_ce * valid[:, None, None]).mean()
        support = scored['support'][..., 0].mean(-1)
        identity = (F.relu(.2 - (support[:, :, None] - support[:, None, :])) * pairs).sum() / pairs.sum().clamp(min=1)
        # The teacher retains only causal target-quality-qualified descriptors.
        history_teacher = torch.cat(state['teacher_features'], 1).detach()
        teacher_valid = torch.cat(state['teacher_valid'], 1)
        descriptor = scored['descriptors'].detach()
        teacher = torch.einsum('bsmd,bkmd->bksm', history_teacher, descriptor)
        teacher = teacher.masked_fill(~teacher_valid[:, None, :, None], -torch.inf).max(2).values.mean(-1)
        teacher_gap = teacher[:, :, None] - teacher[:, None, :]
        student_gap = support[:, :, None] - support[:, None, :]
        preservation = ((student_gap - teacher_gap).square() * pairs).sum() / pairs.sum().clamp(min=1)
        motion_target = relative_geometry(future, observation['distribution']['reference'])
        distribution = observation['distribution']
        distance = (motion_target[:, None] - distribution['means']) / distribution['deviations']
        component = -.5 * (distance.square() + 1.837877).sum(-1) - distribution['deviations'].log().sum(-1)
        time = future_valid[:, None].float()
        motion_loss = -torch.logsumexp((component * time).sum(-1) + distribution['log_weights'], -1).mean() / future_valid.sum(-1).float().mean().clamp(min=1)
        absence = (local_iou.max(-1).values < .5).float()
        search_gain = current[:, 5:].max(-1).values - local_iou.max(-1).values
        search_loss = (F.binary_cross_entropy_with_logits(observation['search_logits'][:, 0], absence, reduction='none') * valid).mean()
        search_loss += (F.smooth_l1_loss(observation['search_logits'][:, 1].tanh(), search_gain, reduction='none') * valid).mean()
        total = quality_loss + advantage + ranking + .2 * (risk_loss + geometry_loss + gate_loss + search_loss) + .1 * (identity + preservation + motion_loss)
        return total, {'quality': quality_loss, 'advantage': advantage, 'ranking': ranking, 'risk': risk_loss,
                       'geometry': geometry_loss, 'gate': gate_loss, 'identity': identity, 'preservation': preservation,
                       'motion': motion_loss, 'search': search_loss}

    def forward(self, clips, compute_loss=True):
        device = self.device
        labels = torch.tensor([c['boxes'].tolist() for c in clips], device=device)
        frames = torch.tensor([c['frames'].tolist() for c in clips], device=device)
        steps = len(clips[0]['images']) - 1
        state = self.initialize([c['images'][0] for c in clips], labels[:, 0], frames[:, 0])
        # Teacher labels exist only in this loss branch, never in step().
        if compute_loss:
            state['teacher_features'] = [state['anchor'][:, None].detach()]
            state['teacher_valid'] = [torch.ones(len(clips), 1, device=device, dtype=torch.bool)]
        losses, metrics, predictions, diagnostics = [], {}, [], []
        for time in range(1, steps + 1):
            observation, state = self.step([c['images'][time] for c in clips], frames[:, time], state, training_candidates=compute_loss)
            predictions.append(observation['box'])
            diagnostics.append(observation['extra_forwards'])
            if compute_loss:
                # Distribution originates at time-1: its +1 mode is THIS frame.
                future_xywh = labels[:, time:time + 3]
                loss, parts = self.training_losses(observation, xyxy(labels[:, time]), xyxy(future_xywh), (future_xywh[..., 2:] > 0).all(-1), state)
                losses.append(loss)
                for name, value in parts.items(): metrics[name] = metrics.get(name, 0.) + value.detach() / steps
                rows = torch.arange(len(clips), device=device)
                state['teacher_features'].append(observation['scored']['descriptors'][rows, observation['choice']][:, None].detach())
                chosen_iou = iou(xyxy(observation['box']), xyxy(labels[:, time]))
                state['teacher_valid'].append((chosen_iou >= .5)[:, None])
        prediction = torch.stack(predictions, 1)
        truth = xyxy(labels[:, 1:steps + 1])
        valid = (labels[:, 1:steps + 1, 2:] > 0).all(-1)
        quality = iou(xyxy(prediction), truth)
        sequence_iou = (quality * valid).sum(1) / valid.sum(1).clamp(min=1)
        return {'loss': torch.stack(losses).mean() if compute_loss else prediction.new_zeros(()), 'parts': metrics,
                'predictions': prediction, 'sequence_iou': sequence_iou,
                'extra_forwards': sum(diagnostics)}
