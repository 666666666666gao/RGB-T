"""CPU witness for the observed sub-margin pause calibration error."""
import argparse
import json
from pathlib import Path

import torch

from research.recoverability_modules import ranking_supervision_loss, write_pair_supervision_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    scores = torch.zeros(1, 7, 5, 2, requires_grad=True)
    utility = torch.zeros_like(scores)
    valid = torch.zeros_like(scores, dtype=torch.bool)
    valid[0, 0, 0] = True
    utility[0, 0, 0, 1] = .001
    with torch.no_grad():
        scores[0, 0, 0, 1] = .04
    old = ranking_supervision_loss(scores, utility, valid, torch.zeros(1), 'pairwise')
    loss = write_pair_supervision_loss(scores, utility, valid)
    assert old == 0 and loss > 0
    loss.backward()
    assert scores.grad[0, 0, 0, 1] > 0 and scores.grad[0, 0, 0, 0] < 0
    assert not scores.grad[~valid].any()
    calibrated = utility.detach().clone().requires_grad_(True)
    assert write_pair_supervision_loss(calibrated, utility, valid) == 0
    no_write = valid.clone()
    no_write[..., 1] = False
    empty = write_pair_supervision_loss(scores, utility, no_write)
    assert empty == 0 and torch.isfinite(empty)
    beneficial = utility.clone()
    beneficial[0, 0, 0, 1] = .1
    good_loss = write_pair_supervision_loss(scores, beneficial, valid)
    good_gradient = torch.autograd.grad(good_loss, scores)[0]
    assert good_gradient[0, 0, 0, 1] < 0 and good_gradient[0, 0, 0, 0] > 0
    result = {'status': 'PASS', 'completed': True, 'device': 'cpu', 'visual_forwards': 0,
              'optimizer_updates': 0, 'old_pairwise_loss_on_false_pause': float(old),
              'write_pair_loss_on_false_pause': float(loss.detach()),
              'gradient_lowers_false_pause_and_raises_regular': True,
              'gradient_preserves_learning_of_beneficial_pause': True,
              'invalid_actions_zero_gradient': True, 'no_writable_candidate_finite_zero': True,
              'correctly_calibrated_gap_zero_loss': True}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
