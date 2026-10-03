"""CPU witness: relative-to-keep success can leave the wrong competitor ahead."""
import argparse
import json
from pathlib import Path

import torch

from research.recoverability_modules import ranking_supervision_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    scores = torch.zeros(1, 7, 5, 2)
    utility = torch.zeros_like(scores)
    legal = torch.zeros_like(scores, dtype=torch.bool)
    legal[0, 0, :3, 0] = True
    utility[0, 0, 1, 0], utility[0, 0, 2, 0] = .7, .3
    scores[0, 0, 1, 0], scores[0, 0, 2, 0] = .04, .10
    reference = torch.zeros(1)
    scores.requires_grad_(True)
    old = ranking_supervision_loss(scores, utility, legal, reference)
    new = ranking_supervision_loss(scores, utility, legal, reference, 'pairwise')
    assert old == 0 and new > 0
    new.backward()
    assert scores.grad[0, 0, 1, 0] < 0 and scores.grad[0, 0, 2, 0] > 0
    assert not scores.grad[~legal].any()
    improved = scores.detach().clone()
    improved[0, 0, 1, 0] = .15
    assert ranking_supervision_loss(improved, utility, legal, reference, 'pairwise') == 0
    # Alternative regions2 and5 cannot coexist in the single-extra budget.
    separated = torch.zeros_like(scores)
    values = torch.zeros_like(utility)
    valid = torch.zeros_like(legal)
    valid[0, 0, 0, 0] = valid[0, 2, 0, 0] = valid[0, 5, 0, 0] = True
    values[0, 2, 0, 0], values[0, 5, 0, 0] = .7, .3
    separated[0, 2, 0, 0], separated[0, 5, 0, 0] = .04, .10
    assert ranking_supervision_loss(separated, values, valid, reference, 'pairwise') == 0
    # Same-region candidates compete, including original versus extra.
    valid[0, 5, 0, 0] = False
    valid[0, 2, 1, 0] = True
    values[0, 2, 1, 0], separated[0, 2, 1, 0] = .3, .10
    assert ranking_supervision_loss(separated, values, valid, reference, 'pairwise') > 0
    result = {'completed': True, 'status': 'PASS', 'device': 'cpu', 'visual_forwards': 0,
              'optimization_updates': 0, 'old_loss_wrong_competitor_ahead': float(old),
              'new_loss_wrong_competitor_ahead': float(new.detach()),
              'gradient_raises_better_and_lowers_worse': True, 'invalid_actions_have_zero_gradient': True,
              'correct_margin_satisfies_loss': True, 'different_extra_regions_not_compared': True,
              'same_region_competitors_compared': True}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
