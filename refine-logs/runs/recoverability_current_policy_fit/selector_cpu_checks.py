"""CPU source checks on two existing collection queries per teacher; no fit."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PRIVATE = Path('/data/gb/experiments/recoverability_current_policy_fit_20261004')
REVIEW = Path('/data/gb/setup/current_policy_fit_review')
OUTPUTS = Path('/data/gb/outputs')
ACCEPTANCE = Path('/data/gb/setup/current_policy_selector_fit_m0_acceptance_20261004.json')
START = time.perf_counter()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
os.chdir(PRIVATE)
sys.path.insert(0, str(PRIVATE))
sys.path.append('/data/gb/GOLA')
import torch
import research.recoverability_modules as modules
import research.train_recoverability as trainer

torch.set_num_threads(4)
assert not torch.cuda.is_initialized()
assert Path(trainer.__file__).parent == Path(modules.__file__).parent == PRIVATE / 'research'
parent_path = OUTPUTS / 'recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
assert parent_sha == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
parent = torch.load(parent_path, map_location='cpu', weights_only=False)
c1 = torch.load(OUTPUTS / 'c1_initial_seed42/best.pth', map_location='cpu', weights_only=False)
assert parent['epoch'] == 4
model = modules.RecoverabilityModules(c1).cpu().eval()
model.load_state_dict(parent['model'], strict=True)
runner = PRIVATE / 'scripts/run_recoverability_current_policy_selector_fit.sh'
subprocess.run(['bash', '-n', str(runner)], check=True)
guard_line = next(line.strip() for line in runner.read_text().splitlines() if str(ACCEPTANCE) in line)
assert not ACCEPTANCE.exists()
closed = subprocess.run([sys.executable, '-c', shlex.split(guard_line)[2]], capture_output=True, text=True)
assert closed.returncode != 0 and 'FileNotFoundError' in closed.stderr


def manual_choice(output, data, extra_region):
    legal = data['valid'][..., None].expand(-1, -1, -1, 2).clone()
    legal[..., 1] &= data['raw_score'] > .84
    available = torch.zeros_like(legal)
    available[:, 0] = True
    if extra_region:
        available[:, extra_region] = True
    scores = output['scores'].masked_fill(~(legal & available), -torch.inf).flatten(1)
    keep = data['original_choice'].long() * 2
    rows = torch.arange(len(keep))
    best_score, best = scores.max(1)
    return torch.where(best_score > scores[rows, keep] + .03, best, keep)


checks = {}
for policy in ('c1', 'own'):
    root = OUTPUTS / 'recoverability_current_policy_m0_20261004' / policy
    data, _, _, jobs = trainer.load_data([root], 'train', torch.device('cpu'))
    assert len(jobs) == 2
    output = trainer.forward(model, data)
    assert output['scores'].requires_grad
    gain, success = modules.search_supervision_targets(output, data, 'selector', .03, 'action')
    assert not gain.requires_grad and not success.requires_grad
    utility = modules.action_utility(data).flatten(1)
    rows = torch.arange(2)
    local = manual_choice(output, data, 0)
    reference = utility[rows, local]
    expected_gains, expected_success = [], []
    for region in range(1, 7):
        selected = manual_choice(output, data, region)
        has_candidate = data['valid'][:, region].any(1)
        expected_gains.append(torch.where(has_candidate, utility[rows, selected] - reference, 0))
        selected_iou = data['current_iou'][rows, selected // 10, selected % 10 // 2]
        expected_success.append(((selected_iou >= .5) & has_candidate).float())
    expected_gains = torch.stack(expected_gains, 1)
    expected_success = torch.stack(expected_success, 1)
    assert torch.equal(gain, expected_gains) and torch.equal(success, expected_success)
    altered = dict(data)
    altered['current_iou'] = torch.full_like(data['current_iou'], .123)
    altered['future_iou'] = torch.full_like(data['future_iou'], .987)
    altered['wrong_update_fraction'] = torch.full_like(data['wrong_update_fraction'], .456)
    before = modules.select_actions(output, data, .03, 'action')
    after = modules.select_actions(output, altered, .03, 'action')
    assert all(torch.equal(value, after[key]) for key, value in before.items())
    _, oracle_parts = modules.objective(model, output, data, 'oracle', .03, 'budgeted', False, 'action')
    _, selector_parts = modules.objective(model, output, data, 'selector', .03, 'budgeted', False, 'action')
    target_dependent = {'search_value', 'search_success', 'search_ranking'}
    assert oracle_parts.keys() == selector_parts.keys()
    assert all(oracle_parts[key] == selector_parts[key] for key in oracle_parts if key not in target_dependent)
    checks[policy] = {
        'existing_collection_M0_rows': 2, 'six_extra_regions_checked': True,
        'actual_targets_equal_manual_legal_local_vs_one_extra_selection': True,
        'targets_detached_despite_gradient_enabled_model_output': True,
        'GT_label_perturbation_does_not_change_selector_actions': True,
        'other_objective_parts_exact_between_oracle_and_selector': True,
        'changed_target_components_only': sorted(target_dependent),
        'sample_nonzero_gross_gain_elements': int(torch.count_nonzero(gain)),
        'sample_max_abs_gross_gain': float(gain.abs().max()),
    }
assert all(torch.equal(value, parent['model'][key]) for key, value in model.state_dict().items())
assert all(not p.requires_grad and p.grad is None for p in model.c1.parameters())
assert not torch.cuda.is_initialized()
record = {
    'status': 'PASS_CPU_SELECTOR_SOURCE_CHECKS', 'sample_checks': checks,
    'actual_missing_selector_M0_acceptance_rejected': True,
    'source_sha256': {
        'trainer_sha256': hashlib.sha256(Path(trainer.__file__).read_bytes()).hexdigest(),
        'modules_sha256': hashlib.sha256(Path(modules.__file__).read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
        'm0_audit_sha256': hashlib.sha256((REVIEW / 'selector_actual_m0_cpu_audit.py').read_bytes()).hexdigest()},
    'execution': {'CUDA_initialized': False, 'GPU_queries': 0, 'optimizer_steps': 0,
                  'neural_jobs_started': 0, 'weights_modified': False,
                  'runtime_seconds': round(time.perf_counter() - START, 3)},
    'limits': 'Actual CPU source behavior on the existing two-query collection M0 per teacher. No complete new cache, new training M0, full fit, learning gain or tracking accuracy acceptance.'}
(REVIEW / 'selector_cpu_checks.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
