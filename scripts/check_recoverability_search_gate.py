"""CPU regression for unconditional regional utility gain and search cost."""
import argparse
import json

import torch

from research.recoverability_modules import search_supervision_targets, select_actions


def fixture():
    current = torch.full((1, 7, 5), .17)
    current[:, 1] = .25
    valid = torch.ones_like(current, dtype=torch.bool)
    data = {'valid': valid, 'original_choice': torch.zeros(1, dtype=torch.long),
            'raw_score': torch.full_like(current, .6), 'current_iou': current,
            'future_iou': current[..., None, None].expand(-1, -1, -1, 2, 3),
            'wrong_update_fraction': torch.zeros((1, 7, 5, 2)),
            'action_valid': valid[..., None].expand(-1, -1, -1, 2)}
    output = {'region_advantage': torch.tensor([[.08, 0., 0., 0., 0., 0.]]),
              'region_success_logits': torch.full((1, 6), torch.logit(torch.tensor(.25))),
              'absence_logit': torch.zeros(1), 'quality_logits': torch.logit(current),
              'target_probability': torch.ones((1, 7, 5, 2)),
              'scores': torch.zeros((1, 7, 5, 2))}
    return output, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-value', choices=('weighted', 'gross'), default='gross')
    args = parser.parse_args()
    output, data = fixture()
    default = select_actions(output, data, .03, 'action')
    weighted = select_actions(output, data, .03, 'action', search_value='weighted')
    assert all(torch.equal(default[key], weighted[key]) for key in default)
    assert not weighted['search_triggered'].item()
    gain, success = search_supervision_targets(output, data, 'oracle')
    assert torch.isclose(gain[0, 0], torch.tensor(.08)) and success[0, 0] == 0
    kwargs = {'search_value': args.search_value}
    actual = select_actions(output, data, .03, 'action', **kwargs)
    assert actual['search_triggered'].item(), (
        'Unconditional utility gain .08 minus .01 search cost exceeds .03; '
        'multiplying by success probability .25 suppresses this supported search.')
    assert actual['searched_region'].item() == 1
    assert actual['region'].item() == 0, 'Executing search must not force an unsupported candidate switch'
    output['region_advantage'].zero_()
    assert not select_actions(output, data, .03, 'action', **kwargs)['search_triggered'].item()
    output, data = fixture()
    output['absence_logit'].fill_(-20)
    output['quality_logits'].fill_(20)
    assert not select_actions(output, data, .03, 'action', **kwargs)['search_triggered'].item()
    print(json.dumps({'status': 'PASS', 'search_value': args.search_value,
                      'real_target_gain': gain[0, 0].item(), 'success_label': success[0, 0].item(),
                      'cases': ['partial-recovery gain', 'cost rejection', 'normal-frame rejection'],
                      'default_weighted_behavior_unchanged': True,
                      'neural_forward_calls': 0, 'optimizer_steps': 0}))


if __name__ == '__main__':
    main()
