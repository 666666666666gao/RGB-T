"""CPU witness for selector-realizable search labels and default oracle labels."""
import argparse
import json
from pathlib import Path

import torch

from research.recoverability_modules import search_supervision_targets, select_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    batch, regions, candidates = 3, 7, 5
    valid = torch.zeros(batch, regions, candidates, dtype=torch.bool)
    valid[:, :, 0] = True
    valid[:, 6] = False  # A genuinely empty cached region.
    valid[2, 0, 1] = True
    current = torch.full((batch, regions, candidates), .2)
    current[0, 1, 0] = .9  # Correct candidate exists, selector rejects it.
    current[1, 0, 0] = .8
    current[1, 2, 0] = .1  # Selector chooses a harmful new candidate.
    current[2, 0, 1] = .7  # Local improvement must not count as search gain.
    current[2, 3, 0] = .9
    data = {
        'valid': valid, 'original_choice': torch.zeros(batch, dtype=torch.long),
        'raw_score': torch.full((batch, regions, candidates), .7),
        'current_iou': current, 'future_iou': current[..., None, None].expand(-1, -1, -1, 2, 3),
        'wrong_update_fraction': torch.zeros(batch, regions, candidates, 2),
    }
    data['action_valid'] = valid[..., None].expand(-1, -1, -1, 2).clone()
    data['action_valid'][..., 1] = False
    scores = torch.zeros(batch, regions, candidates, 2)
    scores[:, 1:] -= .01
    scores[1, 2, 0, 0] = .5
    scores[2, 0, 1, 0] = .3
    scores[2, 3, 0, 0] = .6
    scores.requires_grad_(True)
    output = {
        'scores': scores, 'region_advantage': torch.zeros(batch, 6, requires_grad=True),
        'region_success_logits': torch.zeros(batch, 6), 'absence_logit': torch.zeros(batch),
        'quality_logits': torch.zeros(batch, regions, candidates),
        'target_probability': torch.full((batch, regions, candidates, 2), .8),
    }
    oracle_gain, oracle_success = search_supervision_targets(output, data)
    expected = current.masked_fill(~valid, -torch.inf).max(-1).values[:, 1:]
    expected = torch.where(valid[:, 1:].any(-1), expected, current[:, 0, 0, None]) - current[:, 0, 0, None]
    assert torch.allclose(oracle_gain, expected, atol=1e-7, rtol=0)
    gain, success = search_supervision_targets(output, data, 'selector')
    assert not gain.requires_grad and not success.requires_grad
    assert torch.allclose(gain[0, 0], torch.tensor(0.))
    assert torch.allclose(gain[1, 1], torch.tensor(-.7), atol=1e-7)
    assert torch.allclose(gain[2, 0], torch.tensor(0.))
    assert torch.allclose(gain[2, 2], torch.tensor(.2), atol=1e-7)
    assert torch.equal(gain[:, 5], torch.zeros(batch)) and not success[:, 5].any()
    assert success[0, 0] == 0 and success[1, 1] == 0 and success[2, 2] == 1
    # Cost appears once in deployment; a .2 gross gain becomes .19 net.
    assert torch.allclose(gain[2, 2] - .01, torch.tensor(.19), atol=1e-7)
    prior = {key: value.clone() for key, value in output.items()}
    search_supervision_targets(output, data, 'selector')
    assert all(torch.equal(value, prior[key]) for key, value in output.items())
    # Changing GT cannot change forced selector decisions.
    forced = dict(output)
    forced['region_advantage'] = torch.full_like(output['region_advantage'], -1)
    forced['region_advantage'][:, 1] = 1
    forced['region_success_logits'] = torch.zeros_like(output['region_success_logits'])
    forced['absence_logit'] = torch.full_like(output['absence_logit'], 20)
    first = select_actions(forced, data)['flat_action']
    changed = dict(data)
    changed['current_iou'] = 1 - data['current_iou']
    changed['future_iou'] = 1 - data['future_iou']
    changed['wrong_update_fraction'] = torch.ones_like(data['wrong_update_fraction'])
    assert torch.equal(first, select_actions(forced, changed)['flat_action'])
    result = {
        'completed': True, 'status': 'PASS', 'device': 'cpu', 'optimization_updates': 0,
        'visual_forwards': 0, 'oracle_default_equal': True, 'selector_targets_detached': True,
        'oracle_correct_but_rejected_gain': float(gain[0, 0]),
        'harmful_selected_gain': float(gain[1, 1]),
        'local_improvement_not_search_gain': float(gain[2, 0]),
        'realized_search_gross_gain': float(gain[2, 2]),
        'realized_search_net_gain': float(gain[2, 2] - .01),
        'empty_region_targets_zero': True, 'output_unmodified': True,
        'gt_changes_do_not_select_actions': True,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
