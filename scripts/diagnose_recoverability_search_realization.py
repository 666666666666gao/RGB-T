"""Read-only cached-state search/selection and template-verification diagnostic.

Forced regions expose the deployed selector to one already collected extra crop.
No visual forward or optimizer runs. Future utility labels use frozen C1, so this
is a conditional cached-state diagnostic, not continuous tracking performance.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from research.recoverability_modules import RecoverabilityModules, action_utility, select_actions
from research.train_recoverability import forward, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--partition', choices=('train', 'validation'), required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-queries', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
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
    threshold = checkpoint['args']['threshold']
    records = []
    with torch.no_grad():
        for start in range(0, count, args.batch_size):
            batch = {k: v[start:min(start + args.batch_size, count)] for k, v in data.items()}
            output = forward(model, batch)  # Labels never enter model.forward.
            chosen = select_actions(output, batch, threshold)
            utility = action_utility(batch)
            current = batch['current_iou'].float()
            keep = batch['original_choice'].long()
            rows = torch.arange(len(keep))
            reference = utility[rows, 0, keep, 0]
            writes = batch['raw_score'] > .84
            verified = (output['target_probability'] >= .5).all(-1)
            gated_valid = batch['action_valid'].clone()
            gated_valid[..., 0] &= ~writes | verified
            forced_results = []
            for region in range(1, 7):
                forced = dict(output)
                forced['region_advantage'] = torch.full_like(output['region_advantage'], -1)
                forced['region_advantage'][:, region - 1] = 1
                forced['region_success_logits'] = torch.zeros_like(output['region_success_logits'])
                forced['absence_logit'] = torch.full_like(output['absence_logit'], 20)
                action = select_actions(forced, batch, threshold)
                assert action['search_triggered'].all() and (action['searched_region'] == region).all()
                assert ((action['region'] == 0) | (action['region'] == region)).all()
                assert gated_valid.flatten(1)[rows, action['flat_action']].all()
                available = torch.zeros_like(batch['action_valid'])
                available[:, 0] = True
                available[:, region] = True
                oracle = utility.masked_fill(~(available & batch['action_valid']), -torch.inf).flatten(1).max(1).values
                gated_oracle = utility.masked_fill(~(available & gated_valid), -torch.inf).flatten(1).max(1).values
                region_exists = batch['valid'][:, region].any(1)
                selected_utility = utility.flatten(1)[rows, action['flat_action']]
                assert (selected_utility <= gated_oracle + 1e-6).all()
                selected_current = current[rows, action['region'], action['candidate']]
                eligible = available & gated_valid
                score_winner = output['scores'].masked_fill(~eligible, -torch.inf).flatten(1).argmax(1)
                score_winner_current = current.flatten(1)[rows, score_winner // 2]
                ungated_winner = output['scores'].masked_fill(~(available & batch['action_valid']), -torch.inf).flatten(1).argmax(1)
                ungated_winner_current = current.flatten(1)[rows, ungated_winner // 2]
                correct_available = (available & batch['action_valid'] & (current[..., None] >= .5)).flatten(1).any(1)
                correct_eligible = (eligible & (current[..., None] >= .5)).flatten(1).any(1)
                # A correct highest-score action can still lose to the keep margin.
                threshold_blocks_correct_winner = (score_winner_current >= .5) & (selected_current < .5)
                ranking_rejects_correct = correct_eligible & (score_winner_current < .5)
                eligibility_rejects_correct = correct_available & ~correct_eligible
                region_correct = (current[:, region].masked_fill(~batch['valid'][:, region], -1).max(1).values >= .5)
                region_best = utility[:, region].masked_fill(~batch['action_valid'][:, region], -torch.inf).flatten(1).max(1).values
                region_best = torch.where(region_exists, region_best, reference)
                beneficial = (region_best - reference) > .05
                selector_net_gain = selected_utility - reference - .01
                for row in range(len(keep)):
                    forced_results.append((row, {
                        'region': region, 'region_has_candidate': bool(region_exists[row]),
                        'extra_has_correct_candidate': bool(region_correct[row]),
                        'region_label_gain': float(region_best[row] - reference[row]),
                        'oracle_beneficial': bool(beneficial[row]),
                        'region_oracle_net_gain': float(region_best[row] - reference[row] - .01),
                        'available_oracle_net_gain': float(oracle[row] - reference[row] - .01),
                        'gated_oracle_net_gain': float(gated_oracle[row] - reference[row] - .01),
                        'selector_net_gain': float(selector_net_gain[row]),
                        'selector_nonpositive': bool(selector_net_gain[row] <= 0),
                        'selector_harm_gt_005': bool(selector_net_gain[row] < -.05),
                        'predicted_region_advantage': float(output['region_advantage'][row, region - 1]),
                        'selected_current_iou': float(selected_current[row]),
                        'selected_flat_action': int(action['flat_action'][row]),
                        'selected_pause': bool(action['pause'][row]),
                        'correct_action_available': bool(correct_available[row]),
                        'correct_action_eligible': bool(correct_eligible[row]),
                        'score_winner_flat_action': int(score_winner[row]),
                        'score_winner_current_iou': float(score_winner_current[row]),
                        'threshold_blocks_correct_score_winner': bool(threshold_blocks_correct_winner[row]),
                        'ranking_rejects_eligible_correct_candidate': bool(ranking_rejects_correct[row]),
                        'eligibility_rejects_all_correct_actions': bool(eligibility_rejects_correct[row]),
                        'verification_changes_correct_score_winner_to_wrong': bool(
                            (ungated_winner_current[row] >= .5) & (score_winner_current[row] < .5)),
                    }))
            for row in range(len(keep)):
                job = jobs[start + row]
                original_correct = (current[row, 0].masked_fill(~batch['valid'][row, 0], -1).max() >= .5)
                record = {
                    'sequence': job[0], 'query_frame': job[1], 'prefix_policy': job[2],
                    'keep_current_iou': float(current[row, 0, keep[row]]),
                    'keep_utility': float(reference[row]), 'original_has_correct_candidate': bool(original_correct),
                    'keep_raw_write': bool(writes[row, 0, keep[row]]),
                    'keep_target_probability': output['target_probability'][row, 0, keep[row]].tolist(),
                    'keep_write_verified': bool(verified[row, 0, keep[row]]),
                    'natural_pause': bool(chosen['pause'][row]),
                    'natural_search': bool(chosen['search_triggered'][row]),
                    'natural_flat_action': int(chosen['flat_action'][row]),
                    'forced_regions': [value for index, value in forced_results if index == row],
                }
                records.append(record)
            print(json.dumps({'queries_complete': len(records), 'queries': count}), flush=True)
    regions = [(row, r) for row in records for r in row['forced_regions'] if r['region_has_candidate']]
    missing = [(row, r) for row, r in regions if not row['original_has_correct_candidate'] and r['extra_has_correct_candidate']]
    beneficial = [(row, r) for row, r in regions if r['oracle_beneficial']]
    summary = {
        'queries': len(records), 'nonempty_query_region_pairs': len(regions),
        'oracle_beneficial_pairs': len(beneficial),
        'oracle_beneficial_but_selector_nonpositive': sum(r['selector_nonpositive'] for _, r in beneficial),
        'oracle_beneficial_but_selector_harm_gt_005': sum(r['selector_harm_gt_005'] for _, r in beneficial),
        'missing_target_recovered_pairs': len(missing),
        'recovered_pairs_selected_correct': sum(r['selected_current_iou'] >= .5 for _, r in missing),
        'missing_recovered_pairs_threshold_blocked': sum(r['threshold_blocks_correct_score_winner'] for _, r in missing),
        'missing_recovered_pairs_ranking_rejected': sum(r['ranking_rejects_eligible_correct_candidate'] for _, r in missing),
        'missing_recovered_pairs_all_correct_actions_ineligible': sum(r['eligibility_rejects_all_correct_actions'] for _, r in missing),
        'missing_recovered_ranking_failures_mediated_by_verification': sum(r['verification_changes_correct_score_winner_to_wrong'] for _, r in missing),
        'missing_target_recovered_unique_queries': len({(row['sequence'], row['query_frame']) for row, _ in missing}),
        'recovered_selected_unique_queries': len({(row['sequence'], row['query_frame']) for row, r in missing if r['selected_current_iou'] >= .5}),
        'natural_search_queries': sum(r['natural_search'] for r in records),
        'natural_pause_queries': sum(r['natural_pause'] for r in records),
        'keep_raw_write_queries': sum(r['keep_raw_write'] for r in records),
        'keep_regular_action_denied_by_verification': sum(r['keep_raw_write'] and not r['keep_write_verified'] for r in records),
        'correct_keep_regular_action_denied': sum(r['keep_raw_write'] and not r['keep_write_verified'] and r['keep_current_iou'] >= .5 for r in records),
        'incorrect_keep_regular_action_denied_iou_lt_02': sum(r['keep_raw_write'] and not r['keep_write_verified'] and r['keep_current_iou'] < .2 for r in records),
    }
    assert (summary['missing_recovered_pairs_threshold_blocked']
            + summary['missing_recovered_pairs_ranking_rejected']
            + summary['missing_recovered_pairs_all_correct_actions_ineligible']
            + summary['recovered_pairs_selected_correct']) == summary['missing_target_recovered_pairs']
    result = {
        'completed': True, 'model': args.model, 'checkpoint_epoch': checkpoint['epoch'],
        'cache': args.cache, 'partition': args.partition, 'threshold': threshold,
        'neural_forward_batches': (len(records) + args.batch_size - 1) // args.batch_size,
        'visual_forwards': 0, 'optimization_updates': 0, 'summary': summary,
        'scope': 'CPU cached causal decision inputs; preloaded labels excluded from model inputs and used only for post-forward scoring. Six forced one-extra-region counterfactuals per query; future labels continue frozen C1. Not full-video or official performance. Keep regular-action denial describes C1-keep eligibility, not actual naturally withheld writes.',
        'search_cost': .01, 'elapsed_seconds': time.perf_counter() - started,
        'empty_regions_excluded_from_aggregate': True, 'records': records,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
