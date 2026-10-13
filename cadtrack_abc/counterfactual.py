"""Matched native state interventions; future GT enters label_rollout only.

This is a training collector, not the current running training/evaluation policy.
Each observed context contains local candidates and at most one motion region.
"""
import torch

from .data import clip_boxes, crop_batch, iou, xyxy
from .model import candidates
from .ported_modules import geometry

STATE_FIELDS = {'templates', 'box', 'anchor', 'memory', 'history', 'frames', 'valid', 'quality', 'age'}
ACTION_MODES = ('policy', 'pause_identity', 'defer_geometry', 'accept_geometry', 'rgb_only', 'tir_only')


def clone_state(state):
    assert set(state) == STATE_FIELDS
    return {key: [value.detach().clone() for value in state[key]] if key == 'templates'
            else state[key].detach().clone() for key in state}


def assert_same_state(first, second):
    assert set(first) == set(second) == STATE_FIELDS
    for key in STATE_FIELDS:
        values_a = first[key] if key == 'templates' else [first[key]]
        values_b = second[key] if key == 'templates' else [second[key]]
        assert len(values_a) == len(values_b)
        assert all(torch.equal(a, b) for a, b in zip(values_a, values_b)), key


@torch.inference_mode()
def observe_query(model, images, state):
    """Cache three independent legal local+one-region contexts, without commits."""
    assert len(images) == len(state['box']) == 1
    modules = model.abc
    distribution = modules.motion(state['history'], state['frames'], state['valid'], state['quality'],
                                  state['anchor'], state['memory'])
    patch, transform = crop_batch(images, state['box'], 4., 256, model.device)
    local = candidates(model.base(state['templates'], patch), transform, state['box'])
    encoded = modules.encode(local, state, distribution, 0.)
    local_scored = modules.score(encoded, local)
    summary = torch.cat((encoded['tokens'].mean(1), encoded['tokens'].max(1).values,
                         distribution['means'].flatten(1), distribution['log_weights']), -1)
    query = {'distribution': distribution, 'search_logits': modules.search(summary),
             'candidates': [local], 'scored': [local_scored], 'search_input': summary}
    # Region1 is the parent's highest-probability mode, including stable ties.
    order = distribution['log_weights'][0].argsort(descending=True, stable=True)
    query['motion_mode_order'] = order.cpu().tolist()
    for mode in order:
        target = distribution['means'][:, mode, 0]
        center, size = geometry(distribution['reference'])
        center = center + target[:, :2] * size
        size = size * target[:, 2:].clamp(-1.3863, 1.3863).exp()
        extra_box = clip_boxes(torch.cat((center - size / 2, center + size / 2), -1), images)
        patch, transform = crop_batch(images, extra_box, 6., 256, model.device)
        extra = candidates(model.base(state['templates'], patch), transform, extra_box)
        extra_encoded = modules.encode(extra, state, distribution, 1.)
        pair = {key: torch.cat((local[key], extra[key]), 1) for key in local}
        pair_encoded = {key: torch.cat((encoded[key], extra_encoded[key]), 1) for key in encoded}
        query['candidates'].append(pair)
        query['scored'].append(modules.score(pair_encoded, pair))
    return query


def parent_action(model, query):
    """Exact native selection: only local or the parent's highest-density region."""
    logits = query['search_logits']
    trigger = ((logits[0, 0].sigmoid() > .5)
               | (query['scored'][0]['quality'][0, 0].sigmoid() < .5))
    trigger &= logits[0, 1].tanh() > model.threshold + .01
    context = int(trigger)
    scores = query['scored'][context]['score'][0].clone()
    if context:
        scores[5:] -= .01
    winner = scores.argmax()
    candidate = int(winner) if scores[winner] > scores[0] + model.threshold else 0
    return context, candidate, 'policy'


@torch.inference_mode()
def advance_candidate(model, images, frames, prior, query, context, candidate, mode='policy'):
    """Output a candidate and independently commit geometry and modal evidence.

    With parent_action this must exactly reconstruct native step()'s state.
    These explicit interventions are used only to measure training consequences.
    Geometry modes keep normal identity writes; rgb_only/tir_only restrict
    commits, not the multimodal observation used to choose the candidate.
    """
    assert mode in ACTION_MODES and len(images) == len(prior['box']) == 1
    pair, scored = query['candidates'][context], query['scored'][context]
    assert 0 <= candidate < pair['boxes'].shape[1]
    state = clone_state(prior)
    selected_quality = scored['quality'][:, candidate]
    selected_risk = scored['risk'][:, candidate]
    probability = scored['gate'][:, candidate].softmax(-1)
    descriptor = scored['descriptors'][:, candidate]
    chosen = clip_boxes(pair['boxes'][:, candidate], images).detach()
    geometry_commit = scored['geometry'][:, candidate].sigmoid() >= .5
    if mode in ('defer_geometry', 'accept_geometry'):
        geometry_commit = torch.full_like(geometry_commit, mode == 'accept_geometry')
    distribution = query['distribution']
    center, size = geometry(distribution['reference'])
    dominant = distribution['log_weights'].argmax(-1)
    center = center + distribution['means'][torch.arange(1, device=model.device), dominant, 0, :2] * size
    propagated = clip_boxes(torch.cat((center - size / 2, center + size / 2), -1), images)
    next_box = torch.where(geometry_commit[:, None], chosen, propagated).detach()
    state['age'] += 1
    ordinary_identity = mode in ('policy', 'defer_geometry', 'accept_geometry')
    permit = torch.tensor([[ordinary_identity or mode == 'rgb_only', ordinary_identity or mode == 'tir_only']],
                          device=model.device, dtype=torch.bool)
    raw = pair['raw'][:, candidate]
    write = (state['age'] % 5 == 0) & (raw > .65) & (selected_risk.sigmoid() < .5)
    verified = probability[..., 0] >= .5
    template_commit = write[:, None] & verified & permit
    if template_commit.any():
        template, _ = crop_batch(images, chosen, 2., 128, model.device)
        masks = template_commit.repeat_interleave(3, 1)
        state['templates'][1] = torch.where(masks[:, :, None, None], template, state['templates'][1])
    identity_commit = (verified & (selected_quality.sigmoid()[:, None] >= .5)
                       & (selected_risk.sigmoid()[:, None] < .5) & permit)
    observed = torch.ones(1, device=model.device, dtype=torch.bool)
    updated_memory = model.abc.memory.update(state['memory'], descriptor, probability, observed, identity_commit)
    state['memory'] = torch.where(permit[:, None, :, None], updated_memory, state['memory'])
    state['history'] = torch.cat((state['history'][:, 1:], xyxy(next_box)[:, None]), 1)
    state['frames'] = torch.cat((state['frames'][:, 1:], frames.to(model.device)[:, None]), 1)
    state['valid'] = torch.cat((state['valid'][:, 1:], geometry_commit[:, None]), 1)
    state['quality'] = torch.cat((state['quality'][:, 1:], selected_quality.sigmoid()[:, None]), 1)
    state['box'] = next_box
    trace = {'box': chosen, 'template_commit': template_commit, 'target_memory_commit': identity_commit,
             'geometry_commit': geometry_commit, 'extra_forwards': int(context > 0)}
    return trace, state


def trace_from_observation(observation):
    return {key: observation[key] for key in
            ('box', 'template_commit', 'target_memory_commit', 'geometry_commit', 'extra_forwards')}


@torch.inference_mode()
def rollout(model, state, images, frames):
    """Same frozen continuation policy for every intervention; no GT input."""
    state = clone_state(state)
    traces = []
    for image, frame in zip(images, frames):
        observation, state = model.step([image], torch.tensor([frame], device=model.device), state)
        traces.append(trace_from_observation(observation))
    return traces


def label_rollout(traces, labels):
    """Training-only utility; invalid GT is masked, not treated as known absence."""
    boxes = torch.cat([trace['box'].cpu() for trace in traces], 0)
    truth = torch.as_tensor(labels, dtype=boxes.dtype)
    valid = (truth[:, 2:] > 0).all(-1)
    assert valid[0] and valid[1:].any()
    overlap = iou(xyxy(boxes), xyxy(truth))
    future_mean = overlap[1:][valid[1:]].mean()
    appearance = torch.tensor([bool(trace['template_commit'].any() | trace['target_memory_commit'].any()) for trace in traces])
    geometric = torch.tensor([bool(trace['geometry_commit'][0]) for trace in traces])
    incorrect = (overlap < .2) & valid
    wrong_appearance = (appearance & incorrect).float().sum() / valid.sum()
    wrong_geometry = (geometric & incorrect).float().sum() / valid.sum()
    wrong_fraction = .5 * (wrong_appearance + wrong_geometry)
    cost = .01 * sum(trace['extra_forwards'] for trace in traces) / len(traces)
    utility = .7 * overlap[0] + .3 * future_mean - .1 * wrong_fraction - cost
    return {'current_iou': overlap[0], 'future_iou': overlap[1:], 'future_valid': valid[1:],
            'future_mean_iou': future_mean, 'future_lost_fraction': (overlap[1:][valid[1:]] < .2).float().mean(),
            'wrong_appearance_fraction': wrong_appearance, 'wrong_geometry_fraction': wrong_geometry,
            'search_cost': cost, 'utility': utility}


def decision_cache(query, prior):
    """Decision tensors only; future images/GT never enter this feature pack."""
    contexts = []
    for candidate, scored in zip(query['candidates'], query['scored']):
        context = {key: candidate[key].cpu() for key in ('boxes', 'raw', 'hann', 'instance', 'fused')}
        context.update({key: scored[key].cpu() for key in
                        ('tokens', 'descriptors', 'support', 'gate', 'evidence', 'quality', 'risk', 'geometry', 'score')})
        contexts.append(context)
    return {'contexts': contexts, 'anchor': prior['anchor'].cpu(), 'memory': prior['memory'].cpu(),
            'history': prior['history'].cpu(), 'frames': prior['frames'].cpu(), 'valid': prior['valid'].cpu(),
            'quality': prior['quality'].cpu(), 'search_input': query['search_input'].cpu(),
            'motion': {key: value.cpu() for key, value in query['distribution'].items()},
            'motion_mode_order': query['motion_mode_order']}
