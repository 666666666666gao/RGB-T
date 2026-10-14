"""Batch independent native-policy interventions for TRAIN/DEV supervision.

Every row owns its templates, memory and geometry. Candidate contexts are
gathered independently; regions never attend to other regions or batch rows.
"""
import torch

from .counterfactual import ACTION_MODES, STATE_FIELDS, label_rollout
from .data import clip_boxes, crop_batch, xyxy
from .ported_modules import geometry


def repeat_state(prior, count):
    assert set(prior) == STATE_FIELDS and len(prior['box']) == 1
    return {key: [value.repeat(count, *([1] * (value.ndim - 1))) for value in prior[key]]
            if key == 'templates' else prior[key].repeat(count, *([1] * (prior[key].ndim - 1)))
            for key in STATE_FIELDS}


@torch.inference_mode()
def advance_actions(model, image, frame, prior, query, actions):
    """Vectorized equivalent of advance_candidate for independent query actions."""
    count = len(actions)
    state = repeat_state(prior, count)
    images = [image] * count
    frames = torch.full((count,), frame, device=model.device, dtype=torch.long)
    contexts = torch.tensor([action[0] for action in actions], device=model.device)
    modes = [action[2] for action in actions]
    assert all(mode in ACTION_MODES for mode in modes)
    selected = {key: torch.cat([query['scored'][context][key][:, candidate]
                               for context, candidate, _ in actions], 0)
                for key in ['quality', 'risk', 'geometry', 'gate', 'descriptors']}
    boxes = torch.cat([query['candidates'][context]['boxes'][:, candidate]
                       for context, candidate, _ in actions], 0)
    raw = torch.cat([query['candidates'][context]['raw'][:, candidate]
                     for context, candidate, _ in actions], 0)
    chosen = clip_boxes(boxes, images).detach()
    geometry_commit = selected['geometry'].sigmoid() >= .5
    defer = torch.tensor([mode == 'defer_geometry' for mode in modes], device=model.device)
    accept = torch.tensor([mode == 'accept_geometry' for mode in modes], device=model.device)
    geometry_commit = (geometry_commit & ~defer) | accept
    distribution = query['distribution']
    center, size = geometry(distribution['reference'])
    dominant = distribution['log_weights'].argmax(-1)
    center = center + distribution['means'][torch.arange(1, device=model.device), dominant, 0, :2] * size
    propagated = clip_boxes(torch.cat((center - size / 2, center + size / 2), -1).repeat(count, 1), images)
    next_box = torch.where(geometry_commit[:, None], chosen, propagated).detach()
    state['age'] += 1
    ordinary = [mode in ('policy', 'defer_geometry', 'accept_geometry') for mode in modes]
    permit = torch.tensor([[normal or mode == 'rgb_only', normal or mode == 'tir_only']
                           for normal, mode in zip(ordinary, modes)], device=model.device, dtype=torch.bool)
    probability = selected['gate'].softmax(-1)
    verified = probability[..., 0] >= .5
    write = (state['age'] % 5 == 0) & (raw > .65) & (selected['risk'].sigmoid() < .5)
    template_commit = write[:, None] & verified & permit
    if template_commit.any():
        template, _ = crop_batch(images, chosen, 2., 128, model.device)
        masks = template_commit.repeat_interleave(3, 1)
        state['templates'][1] = torch.where(masks[:, :, None, None], template, state['templates'][1])
    identity_commit = (verified & (selected['quality'].sigmoid()[:, None] >= .5)
                       & (selected['risk'].sigmoid()[:, None] < .5) & permit)
    observed = torch.ones(count, device=model.device, dtype=torch.bool)
    updated = model.abc.memory.update(state['memory'], selected['descriptors'], probability, observed, identity_commit)
    state['memory'] = torch.where(permit[:, None, :, None], updated, state['memory'])
    state['history'] = torch.cat((state['history'][:, 1:], xyxy(next_box)[:, None]), 1)
    state['frames'] = torch.cat((state['frames'][:, 1:], frames[:, None]), 1)
    state['valid'] = torch.cat((state['valid'][:, 1:], geometry_commit[:, None]), 1)
    state['quality'] = torch.cat((state['quality'][:, 1:], selected['quality'].sigmoid()[:, None]), 1)
    state['box'] = next_box
    trace = {'box': chosen, 'template_commit': template_commit,
             'target_memory_commit': identity_commit, 'geometry_commit': geometry_commit,
             'extra_forwards': (contexts > 0).long()}
    return trace, state


@torch.inference_mode()
def rollout_actions(model, state, future_images, future_frames):
    count = len(state['box'])
    traces = []
    for image, frame in zip(future_images, future_frames):
        observation, state = model.step([image] * count,
                                       torch.full((count,), frame, device=model.device, dtype=torch.long), state)
        # Native step executes a common batch when any row needs an extra crop.
        # Labels use each row's deployed budget, not the physical batch total.
        logits = observation['search_logits']
        extra = (((logits[:, 0].sigmoid() > .5)
                  | (observation['local_scored']['quality'][:, 0].sigmoid() < .5))
                 & (logits[:, 1].tanh() > model.threshold + .01))
        traces.append({key: observation[key] for key in
                       ['box', 'template_commit', 'target_memory_commit', 'geometry_commit']}
                      | {'extra_forwards': extra.long()})
    return traces


def unpack_traces(traces, count):
    cpu = [{key: value.cpu() for key, value in trace.items()} for trace in traces]
    return [[{key: int(value[row]) if key == 'extra_forwards' else value[row:row + 1]
              for key, value in trace.items()} for trace in cpu] for row in range(count)]


@torch.inference_mode()
def collect_batched_labels(model, image, frame, prior, query, future_images, future_frames, labels, modes, batch):
    assert batch > 1 and model.__class__.__name__ == 'CADTrackABC'
    actions = [(context, candidate, mode)
               for context, candidates in enumerate(query['candidates'])
               for candidate in range(candidates['boxes'].shape[1]) for mode in modes]
    output = []
    for begin in range(0, len(actions), batch):
        current = actions[begin:begin + batch]
        trace, state = advance_actions(model, image, frame, prior, query, current)
        traces = [trace] + rollout_actions(model, state, future_images, future_frames)
        for (context, candidate, mode), single in zip(current, unpack_traces(traces, len(current))):
            labeled = label_rollout(single, labels)
            assert torch.isfinite(labeled['utility'])
            output.append(dict(context=context, candidate=candidate, mode=mode, **labeled))
    return output
