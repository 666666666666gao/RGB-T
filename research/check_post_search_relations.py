"""Check actual cached causal inputs, independent pairs and matched scorer cost."""
import argparse
import json
import time
from pathlib import Path

import torch

from .recoverability_modules import DECISION_FIELDS, RecoverabilityModules, observed_output, objective
from .train_recoverability import load_data, forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    c1 = torch.load(checkpoint['args']['c1_head'], map_location='cpu', weights_only=False)
    models = [RecoverabilityModules(c1, candidate_relations=True, post_search_bidirectional=bidirectional,
                                   observed_pair_training=True).to(device).eval() for bidirectional in (False, True)]
    for model in models:
        model.load_state_dict(checkpoint['model'], strict=True)
    assert all(torch.equal(a, b) for a, b in zip(models[0].state_dict().values(), models[1].state_dict().values()))
    data, _, _, _ = load_data([args.validation], 'validation', device)
    eligible = (data['valid'][:, 1].any(-1) & data['valid'][:, 2].any(-1)).nonzero(as_tuple=True)[0]
    assert len(eligible)
    batch = {key: value[eligible[:4]] for key, value in data.items()}
    with torch.no_grad():
        old, new = (forward(model, batch) for model in models)
        assert all(torch.equal(old[key], new[key]) for key in old)
        perturbed = {key: value.clone() for key, value in batch.items()}
        for key in ('features', 'instance_features', 'evidence', 'c1_quality', 'raw_score', 'image_boxes'):
            perturbed[key][:, 2] += .37
        changed = forward(models[1], perturbed)
        for key in ('region_advantage', 'region_success_logits', 'absence_logit'):
            assert torch.equal(new[key], changed[key])
        for field in ('scores', 'advantage', 'quality_logits', 'harm_logits'):
            assert torch.equal(new[field][:, 0], changed[field][:, 0])
            assert torch.equal(new['post_search_' + field][:, 0], changed['post_search_' + field][:, 0])
        assert not torch.equal(new['post_search_quality_logits'][:, 1, 0], changed['post_search_quality_logits'][:, 1, 0])
        only_pair = {key: value.clone() for key, value in batch.items()}
        only_pair['valid'][:, 2:] = False
        single = forward(models[1], only_pair)
        torch.testing.assert_close(new['post_search_scores'][:, 0], single['post_search_scores'][:, 0], rtol=1e-6, atol=1e-6)
        local_only = dict(batch, valid=batch['valid'].clone())
        local_only['valid'][:, 1:] = False
        no_extra = forward(models[1], local_only)
        no_extra_decision = observed_output(no_extra, torch.zeros_like(batch['original_choice'].long()))
        assert all(torch.equal(no_extra[key], no_extra_decision[key]) for key in ('scores', 'quality_logits', 'advantage'))
        assert all(torch.equal(old[key][:, 0], no_extra[key][:, 0]) for key in ('scores', 'quality_logits', 'advantage'))
        rows = torch.arange(len(eligible[:4]), device=device)
        keep = batch['original_choice'].long()
        for region in range(1, 7):
            current = observed_output(new, torch.full_like(keep, region))
            assert not current['advantage'][rows, 0, keep, 0].any()
            assert not current['scores'][rows, 0, keep, 0].any()
        assert set(DECISION_FIELDS).isdisjoint({'current_iou', 'future_iou', 'history_iou', 'action_valid', 'wrong_update_fraction'})

    for model in models:
        model.train()
        output = forward(model, batch)
        loss, _ = objective(model, output, batch, 'oracle', .03, 'budgeted', False, 'action', 'gross')
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in model.c1.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.relations.parameters())
        model.eval()

    # Same single observed pair; CUDA-synchronised head cost, no extra GOLA pass.
    one = {key: value[:1] for key, value in only_pair.items()}
    times = [[], []]
    with torch.inference_mode():
        for _ in range(20):
            for model in models: forward(model, one)
        for iteration in range(100):
            for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                forward(models[index], one)
                torch.cuda.synchronize(device)
                times[index].append((time.perf_counter() - started) * 1000)
    result = {'completed': True, 'same_loaded_parameters': True,
              'pre_search_outputs_and_local_only_scores_exact': True,
              'unobserved_region_does_not_change_other_pair': True,
              'observed_region_changes_both_sides': True,
              'single_executed_pair_matches_its_independent_cache_case': True,
              'post_search_keep_reference_recomputed': True,
              'both_matched_pair_objectives_backward_finite_relations_gradient': True,
              'frozen_c1_gradients_absent': True, 'optimizer_updates': 0,
              'causal_input_fields': DECISION_FIELDS,
              'scorer_ms': {name: {'mean': sum(values) / len(values), 'p95': sorted(values)[94]}
                            for name, values in zip(('one_way', 'post_search_bidirectional'), times)},
              'timing_scope': '100 alternating synchronised cached module forwards, one actual region, batch1; excludes GOLA and is not full-system FPS',
              'method_effect_or_formal_accuracy_proven': False}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
