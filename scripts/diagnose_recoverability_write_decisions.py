"""Read-only cached-state audit of write risk, verification, and forced pauses.

Labels score decisions after the causal forward. Future utilities use frozen-C1
continuations; this is neither a deployment policy nor full-video performance.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from research.recoverability_modules import RecoverabilityModules, action_utility, select_actions
from research.train_recoverability import forward, load_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--partition', choices=('train', 'validation'), required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-queries', type=int)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    data, configs, _, jobs = load_data([args.cache], args.partition, torch.device('cpu'))
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    assert checkpoint['args']['c1_head'] == configs[0]['head']
    c1 = torch.load(configs[0]['head'], map_location='cpu', weights_only=False)
    model = RecoverabilityModules(c1).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    count = len(jobs) if args.max_queries is None else min(args.max_queries, len(jobs))
    assert count > 0 and args.batch_size > 0
    groups = {}
    query_rows = []
    with torch.no_grad():
        for start in range(0, count, args.batch_size):
            batch = {k: v[start:min(start + args.batch_size, count)] for k, v in data.items()}
            output = forward(model, batch)
            chosen = select_actions(output, batch, checkpoint['args']['threshold'])
            action_chosen = select_actions(output, batch, checkpoint['args']['threshold'], 'action')
            assert torch.equal(chosen['search_triggered'], action_chosen['search_triggered'])
            assert torch.equal(chosen['searched_region'], action_chosen['searched_region'])
            valid = batch['valid']
            eligible = valid & (batch['raw_score'] > .84)
            quality = batch['current_iou'].float()
            wrong = quality < .2
            risk = output['write_risk_logits'].sigmoid()
            proof = output['target_probability']
            verified = (proof >= .5).all(-1)
            denied = eligible & ~verified
            pause_score_gap = output['scores'][..., 1] - output['scores'][..., 0]
            utility = action_utility(batch)
            pause_utility_gap = utility[..., 1] - utility[..., 0]
            bce = F.binary_cross_entropy_with_logits(output['write_risk_logits'], wrong.float(), reduction='none')
            joint_bce = F.binary_cross_entropy_with_logits(output['write_risk_logits'], (wrong & eligible).float(), reduction='none')
            masks = {'all_valid': valid, 'write_eligible': eligible,
                     'eligible_correct': eligible & (quality >= .5),
                     'eligible_wrong': eligible & wrong,
                     'eligible_uncertain': eligible & (quality >= .2) & (quality < .5)}
            for name, mask in masks.items():
                values = {'candidates': int(mask.sum()), 'wrong': int((mask & wrong).sum()),
                          'predicted_risk_sum': float(risk[mask].sum()),
                          'conditional_bce_sum': float(bce[mask].sum()),
                          'joint_event_bce_sum': float(joint_bce[mask].sum()),
                          'risk_brier_sum': float((risk[mask] - wrong[mask].float()).square().sum()),
                          'regular_denied_by_verification': int((mask & denied).sum()),
                          'denied_but_regular_score_not_worse': int((mask & denied & (pause_score_gap <= 0)).sum()),
                          'denied_but_pause_teacher_utility_not_better': int((mask & denied & (pause_utility_gap <= 0)).sum()),
                          'rgb_target_probability_sum': float(proof[..., 0][mask].sum()),
                          'tir_target_probability_sum': float(proof[..., 1][mask].sum())}
                if name not in groups:
                    groups[name] = {key: 0 for key in values}
                for key, value in values.items():
                    groups[name][key] += value
            for row in range(len(valid)):
                region, candidate = int(chosen['region'][row]), int(chosen['candidate'][row])
                index = (row, region, candidate)
                action_region, action_candidate = int(action_chosen['region'][row]), int(action_chosen['candidate'][row])
                action_index = (row, action_region, action_candidate)
                query_rows.append({'query_index': start + row, 'chosen_region': region,
                                   'chosen_candidate': candidate, 'chosen_pause': bool(chosen['pause'][row]),
                                   'write_eligible': bool(eligible[index]),
                                   'current_iou': float(quality[index]), 'predicted_write_risk': float(risk[index]),
                                   'regular_verified': bool(verified[index]),
                                   'pause_score_gap': float(pause_score_gap[index]),
                                   'pause_teacher_utility_gap': float(pause_utility_gap[index]),
                                   'chosen_teacher_utility': float(utility[row, region, candidate, int(chosen['pause'][row])]),
                                   'action_region': action_region, 'action_candidate': action_candidate,
                                   'action_pause': bool(action_chosen['pause'][row]),
                                   'action_current_iou': float(quality[action_index]),
                                   'action_teacher_utility': float(utility[row, action_region, action_candidate, int(action_chosen['pause'][row])])})
        # An actual writable C1-keep state isolates the existing risk coefficient.
        rows = torch.arange(len(jobs))
        writable_keep = data['raw_score'][rows, 0, data['original_choice'].long()] > .84
        witness_index = int(writable_keep.nonzero()[0, 0])
        witness_data = {key: value[witness_index:witness_index + 1] for key, value in data.items()}
        witness_model = RecoverabilityModules(c1).eval()
        witness_model.action[-1].bias[5] = 10.
        risk_only = forward(witness_model, witness_data)
        selected = select_actions(risk_only, witness_data, .03)
        keep = int(witness_data['original_choice'][0])
        gap = float(risk_only['scores'][0, 0, keep, 1] - risk_only['scores'][0, 0, keep, 0])
        assert 0 < gap < .025 < .03
        assert int(selected['flat_action'][0]) == keep * 2 and not selected['search_triggered'].any()
        blocked = dict(risk_only)
        blocked['target_probability'] = torch.full_like(risk_only['target_probability'], .4)
        identity_choice = select_actions(blocked, witness_data, .03)
        action_choice = select_actions(blocked, witness_data, .03, 'action')
        assert int(identity_choice['flat_action'][0]) == keep * 2 + 1
        assert int(action_choice['flat_action'][0]) == keep * 2
    result = {'completed': True, 'checkpoint_epoch': checkpoint['epoch'], 'model': args.model,
              'cache': args.cache, 'partition': args.partition, 'queries': count,
              'visual_forwards': 0, 'optimization_updates': 0, 'groups': groups,
              'queries_scored_after_forward': query_rows,
              'risk_only_witness': {'query_index': witness_index, 'risk_logit': 10., 'pause_score_gap': gap,
                                    'keep_threshold': .03, 'regular_keep_retained': True,
                                    'witness_uses_fresh_initial_model_not_modified_checkpoint': True,
                                    'low_identity_identity_mode_forces_pause': True,
                                    'low_identity_action_mode_retains_regular': True},
              'action_comparison': 'Same output/state/checkpoint: identity versus action verification. Search decisions/cost unchanged; action mode still uses learned regular/pause scores and .03 keep threshold, unlike unsafe-writes. Target memory gates unchanged. Query utility is frozen-C1 cached counterfactual, not full-video causality.',
              'scope': 'Read-only CPU causal cache forwards; GT only after choices. Candidate groups cover all cached regions, not actual online writes. Training risk label is eligible & IoU<.2, measured by joint_event_bce_sum. conditional_bce_sum/risk_brier_sum instead use IoU<.2; only the eligible groups represent conditional write-risk calibration. all_valid versions are localization-label diagnostics, not the original training risk loss. IoU is a localization proxy, not sensor reliability or semantic identity. Future pause utilities remain frozen-C1 teacher labels; pause_teacher_utility_gap is meaningful only on write_eligible rows. Risk-only witness is one additional actual writable state from this partition, independent of max-queries aggregate. No deployed threshold change or performance claim.',
              'elapsed_seconds': time.perf_counter() - started}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != 'queries_scored_after_forward'}))


if __name__ == '__main__':
    main()
