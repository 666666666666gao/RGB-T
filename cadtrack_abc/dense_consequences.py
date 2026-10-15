"""R3 matched action consequences; annotations are used only for labels.

Every counterfactual starts from the same native snapshot. Continuations use
the frozen teacher's real online policy, including its search and submissions.
Old R1 action labels do not have the R3 geometry-quarantine semantics.
"""
import numpy as np
import torch

from .counterfactual import ACTION_MODES, label_rollout
from .future_actions import action_features


def assert_same_snapshot(first, second):
    assert first['box'] == second['box'] and first['frame'] == second['frame']
    a, b = first['state'], second['state']
    assert a['age'] == b['age']
    assert all(torch.equal(x, y) and x.stride() == y.stride()
               for x, y in zip(a['templates'], b['templates']))
    assert all(torch.equal(a[name], b[name]) for name in ('box', 'history', 'frames', 'valid', 'quality'))
    assert set(a['memory']) == set(b['memory'])
    assert all(torch.equal(a['memory'][name], b['memory'][name]) for name in a['memory'])


def label_native_traces(traces, labels):
    """Reuse the established utility on actual native boxes/commit receipts."""
    converted = [dict(box=torch.as_tensor(trace['box'], dtype=torch.float64)[None],
        template_commit=torch.tensor([trace['template_commit_rgb_tir']], dtype=torch.bool),
        target_memory_commit=torch.tensor([trace['target_memory_commit_rgb_tir']], dtype=torch.bool),
        geometry_commit=torch.tensor([trace['geometry_commit']], dtype=torch.bool),
        extra_forwards=trace['extra_forwards']) for trace in traces]
    return label_rollout(converted, labels)


def cache_context(query, age):
    candidate, context = query['pair'], query['context']
    baseline = torch.zeros(1, dtype=torch.long, device=candidate['raw'].device)
    frame = torch.full_like(baseline, age)
    features = action_features(dict(candidate, **context), query['distribution'], frame, baseline)
    return dict(features=features.detach().cpu(), baseline=0,
        native_boxes_xywh=np.asarray(query['native_boxes'], dtype=np.float64),
        extra_forwards=query['extra_forwards'])


@torch.inference_mode()
def collect_matched_query(tracker, image, future_images, labels):
    """Four independent contexts: local or local+one probability-ranked region.

All three extra forwards are training-only label collection. The deployed
policy still observes zero or one additional region, never their union.
"""
    prior = tracker.snapshot()
    local = tracker.observe_local(image)
    order = local['distribution']['log_weights'][0].argsort(descending=True, stable=True)
    contexts = [local] + [tracker.observe_extra(image, local, int(mode)) for mode in order]
    assert_same_snapshot(prior, tracker.snapshot())
    baseline = torch.zeros(1, dtype=torch.long, device=local['pair']['raw'].device)
    age = torch.full_like(baseline, prior['state']['age'])

    # Reconstruct the actual deployed decision from the independently cached
    # contexts, and compare it with track(), including every mutable state.
    score, rank = local['search']['scores'].max(1)
    reference_context = 1 + int(rank[0]) if bool(score[0] > tracker.network.decisions.threshold + .01) else 0
    reference_query = contexts[reference_context]
    choice, mode, _ = tracker.network.decisions.choose(reference_query['pair'],
        reference_query['context'], reference_query['distribution'], age)
    reference_action = (reference_context, int(choice[0]), int(mode[0]))
    replay = tracker.advance(image, reference_query, reference_action[1], ACTION_MODES[reference_action[2]])
    replay_state = tracker.snapshot()
    tracker.restore(prior)
    native = tracker.track(image)
    native_state = tracker.snapshot()
    assert np.array_equal(replay['box'], native['box'])
    assert replay['choice'] == native['choice'] and replay['future_mode'] == native['future_mode']
    assert_same_snapshot(replay_state, native_state)

    records = []
    for query in contexts:
        cache = cache_context(query, prior['state']['age'])
        count = len(query['native_boxes'])
        action_labels = []
        for candidate in range(count):
            for mode in ACTION_MODES:
                tracker.restore(prior)
                trace = tracker.advance(image, query, candidate, mode)
                traces = [trace] + [tracker.track(future) for future in future_images]
                result = label_native_traces(traces, labels)
                assert bool(torch.isfinite(result['utility']))
                action_labels.append(result)
        for field in ('utility', 'current_iou', 'future_mean_iou', 'future_lost_fraction',
                      'wrong_appearance_fraction', 'wrong_geometry_fraction', 'search_cost'):
            cache[field] = torch.stack([label[field] if torch.is_tensor(label[field])
                else torch.tensor(label[field]) for label in action_labels]).reshape(count, len(ACTION_MODES)).float()
        cache['future_iou'] = torch.stack([label['future_iou'] for label in action_labels]).reshape(count, len(ACTION_MODES), -1).float()
        cache['future_valid'] = action_labels[0]['future_valid']
        records.append(cache)
    tracker.restore(native_state)
    assert_same_snapshot(native_state, tracker.snapshot())
    return dict(contexts=records, search_input=local['search_input'].detach().cpu(),
        motion_mode_order=order.cpu().tolist(), reference_action=reference_action,
        prefix_state_unchanged_by_counterfactuals=True,
        actual_native_replay_equal=True, actions=sum(record['utility'].numel() for record in records),
        annotation_scope='labels only; candidate/action/search features are causal',
        context_budget='local5 or local5+one executed region5; three extras never interact')
