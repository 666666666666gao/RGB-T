"""Check deployment threshold, actual singleton semantics, and one-extra budget."""
import ast
import json
from pathlib import Path

import torch
from torch.nn import functional as F
import research.recoverability_modules as modules

torch.set_num_threads(4)
private = Path('/data/gb/experiments/recoverability_budgeted_action_20261004/research')
assert Path(modules.__file__).parent == private
old_tree = ast.parse(Path('/data/gb/GOLA/research/recoverability_modules.py').read_text())
new_tree = ast.parse(Path(modules.__file__).read_text())
old_defs = {n.name: n for n in old_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
new_defs = {n.name: n for n in new_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
unchanged = [name for name in old_defs if name != 'objective']
assert all(ast.dump(old_defs[name]) == ast.dump(new_defs[name]) for name in unchanged)


def example():
    scores = torch.zeros(1, 7, 5, 2, requires_grad=True)
    utility = torch.zeros_like(scores)
    legal = torch.zeros_like(scores, dtype=torch.bool)
    utility[0, 0, 2, 0] = .2
    legal[0, 0, 2, 0] = True
    return scores, utility, legal


def loss_and_gradient(scores, utility, legal):
    keep = torch.tensor([2])
    scores = scores.clone().detach().requires_grad_()
    # The real action head's relative keep score is identically zero.
    nonkeep = torch.ones_like(scores, dtype=torch.bool)
    nonkeep[0, 0, 2, 0] = False
    relative = scores.masked_fill(~nonkeep, 0)
    loss = modules.budgeted_winner_loss(relative, utility, legal, utility[:, 0, 2, 0], keep, .03)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(scores.grad).all()
    return float(loss), scores.grad


results = {}
for name, gain, score, direction in [('beneficial_local', .04, 0., -1),
                                      ('unnecessary_switch', .02, .06, 1)]:
    s, u, valid = example()
    u[0, 0, 0, 0] = .2 + gain
    valid[0, 0, 0, 0] = True
    with torch.no_grad():
        s[0, 0, 0, 0] = score
    loss, gradient = loss_and_gradient(s, u, valid)
    assert gradient[0, 0, 0, 0] * direction > 0
    updated = score - .1 * float(gradient[0, 0, 0, 0])
    assert (updated > .03) == (direction == -1)
    results[name] = {'loss': loss, 'candidate_gradient': float(gradient[0, 0, 0, 0]),
                     'score_after_one_gradient_step': updated}

s, u, valid = example()
u[0, 1, 0, 0] = .235  # Gross +.035, but net +.025 after the real search cost.
valid[0, 1, 0, 0] = True
with torch.no_grad():
    s[0, 1, 0, 0] = .06
loss, gradient = loss_and_gradient(s, u, valid)
assert gradient[0, 1, 0, 0] > 0
results['extra_cost_preserves_keep'] = {'loss': loss, 'gradient': float(gradient[0, 1, 0, 0])}

for gain, direction in [(.03, 1), (.030001, -1)]:
    s, u, valid = example()
    u[0, 0, 2, 0] = 0
    u[0, 0, 0, 0] = gain
    valid[0, 0, 0, 0] = True
    with torch.no_grad():
        s[0, 0, 0, 0] = .06 if direction == 1 else 0
    loss, gradient = loss_and_gradient(s, u, valid)
    assert gradient[0, 0, 0, 0] * direction > 0
    results[f'strict_threshold_{gain}'] = {'loss': loss, 'gradient': float(gradient[0, 0, 0, 0])}

s, u, valid = example()
valid[:, 1:, :, 0] = True
with torch.no_grad():
    s[:, 1:, :, 0] = -1
loss, gradient = loss_and_gradient(s, u, valid)
assert loss == 0 and (gradient == 0).all()
results['actual_original_singleton_has_no_fake_rival'] = {'loss': loss, 'finite_zero_gradient': True}

s, u, valid = example()
u[0, 1, 0, 0], u[0, 2, 0, 0] = .8, .4
valid[0, 1:3, 0, 0] = True
with torch.no_grad():
    s[0, 1, 0, 0] = .9
loss1, gradient1 = loss_and_gradient(s, u, valid)
with torch.no_grad():
    s[0, 1, 0, 0] = 1.9
loss2, gradient2 = loss_and_gradient(s, u, valid)
assert loss1 == loss2 and torch.equal(gradient1, gradient2)
assert gradient2[0, 2, 0, 0] < 0 and gradient2[0, 1, 0, 0] == 0
results['incompatible_extra_regions_do_not_compete'] = {'loss': loss2, 'gradient_exactly_unchanged': True}

record = {'status': 'PASS', 'scope': 'CPU loss and unchanged forward checks; no NN, GPU, or performance claim',
          'module_path': modules.__file__, 'old_forward_and_helpers_ast_exact': unchanged, 'checks': results}
Path('/data/gb/setup/budgeted_action_cpu_loss_checks_20261004.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
