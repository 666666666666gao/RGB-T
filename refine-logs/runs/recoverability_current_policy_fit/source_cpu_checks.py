"""Bounded source preparation checks; no new collection or training acceptance."""
import ast
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PRIVATE = Path('/data/gb/experiments/recoverability_current_policy_fit_20261004')
WARM = Path('/data/gb/experiments/recoverability_warm_budgeted_action_20261004')
MAIN = Path('/data/gb/GOLA')
REVIEW = Path('/data/gb/setup/current_policy_fit_review')
OUTPUTS = Path('/data/gb/outputs')
PARENT = OUTPUTS / 'recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
PREFIX = OUTPUTS / 'recoverability_warm_budgeted_action_own_b384_full_20261004/best.pth'
C1 = OUTPUTS / 'c1_initial_seed42/best.pth'
ACCEPTANCE = Path('/data/gb/setup/current_policy_fit_m0_acceptance_20261004.json')
START = time.perf_counter()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
os.chdir(PRIVATE)
sys.path.insert(0, str(PRIVATE))
sys.path.append(str(MAIN))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def objects(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(Path(path).read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


for path in [PRIVATE / 'research/train_recoverability.py',
             PRIVATE / 'research/recoverability_modules.py',
             REVIEW / 'merge_current_policy_caches.py', REVIEW / 'actual_m0_cpu_audit.py']:
    ast.parse(path.read_text())
runner = PRIVATE / 'scripts/run_recoverability_current_policy_fit.sh'
subprocess.run(['bash', '-n', str(runner)], check=True)
source_hashes = {
    'trainer_sha256': sha(PRIVATE / 'research/train_recoverability.py'),
    'modules_sha256': sha(PRIVATE / 'research/recoverability_modules.py'),
    'runner_sha256': sha(runner),
    'merge_sha256': sha(REVIEW / 'merge_current_policy_caches.py'),
    'm0_audit_sha256': sha(REVIEW / 'actual_m0_cpu_audit.py'),
}
new_names = {p.name for p in (PRIVATE / 'research').glob('*.py')}
warm_names = {p.name for p in (WARM / 'research').glob('*.py')}
assert new_names == warm_names and len(new_names) == 39
assert all((PRIVATE / 'research' / name).read_bytes() ==
           (WARM / 'research' / name).read_bytes() for name in new_names)
module_objects = objects(PRIVATE / 'research/recoverability_modules.py')
main_objects = objects(MAIN / 'research/recoverability_modules.py')
for name in ('InstanceMemory', 'RecoverabilityModules', 'select_actions'):
    assert module_objects[name] == main_objects[name]
assert sha(MAIN / 'research/train_recoverability.py') == '4b1327a541d426aeade308e4fc3d7c8e3110e35dae4612c11b93029783fd4e93'
assert sha(MAIN / 'research/recoverability_modules.py') == '371269db053cdbd38b510557e213f1acfb9c6427e6eabd2df5685c59bdfbb158'
old_runner = (WARM / 'scripts/run_recoverability_warm_budgeted_action_fit.sh').read_text()
new_runner = runner.read_text()
command_start = '/data/gb/envs/gola/bin/python -u -m research.train_recoverability'
command_end = 'date -Iseconds > "$run/training_completed.txt"'
assert old_runner.split(command_start)[1].split(command_end)[0] == new_runner.split(command_start)[1].split(command_end)[0]
guard_line = next(line.strip() for line in new_runner.splitlines() if str(ACCEPTANCE) in line)
guard_code = shlex.split(guard_line)[2]
assert not ACCEPTANCE.exists()
closed = subprocess.run([sys.executable, '-c', guard_code], capture_output=True, text=True)
assert closed.returncode != 0 and 'FileNotFoundError' in closed.stderr
assert not (OUTPUTS / 'recoverability_current_policy_merged_20261004/paired_cache_gate.json').exists()
assert all(not (OUTPUTS / f'recoverability_current_policy_fit_{policy}_b384_{stage}_20261004').exists()
           for policy in ('c1', 'own') for stage in ('m0', 'full'))

import numpy as np
import torch
import research.train_recoverability as trainer
import research.recoverability_modules as modules

torch.set_num_threads(4)
torch.set_grad_enabled(False)
assert not torch.cuda.is_initialized()
assert Path(trainer.__file__).parent == Path(modules.__file__).parent == PRIVATE / 'research'
parent_sha, prefix_sha = sha(PARENT), sha(PREFIX)
assert parent_sha == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
assert prefix_sha == 'a1ce56f4c7fa9137483de6c70c430165f1a265e726ac1bed56d02aa32a659e62'
parent = torch.load(PARENT, map_location='cpu', weights_only=False)
prefix = torch.load(PREFIX, map_location='cpu', weights_only=False)
c1 = torch.load(C1, map_location='cpu', weights_only=False)
assert parent['module'] == prefix['module'] == 'ABC_recoverability'
assert parent['epoch'] == 4 and prefix['epoch'] == 25
assert len(parent['model']) == len(prefix['model']) == 25 and len(c1['head']) == 8
for checkpoint in (parent, prefix):
    assert all(torch.equal(checkpoint['model']['c1.' + key], value) for key, value in c1['head'].items())
model = modules.RecoverabilityModules(c1).cpu().eval()
model.load_state_dict(parent['model'], strict=True)
assert all(torch.equal(value, parent['model'][key]) for key, value in model.state_dict().items())
model.train()
assert not model.c1.training and all(not p.requires_grad and p.grad is None for p in model.c1.parameters())
model.eval()
assert sum(p.requires_grad for p in model.parameters()) == 17
spec = importlib.util.spec_from_file_location('research._review_main_modules', MAIN / 'research/recoverability_modules.py')
main_modules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_modules)
main_model = main_modules.RecoverabilityModules(c1).cpu().eval()
main_model.load_state_dict(parent['model'], strict=True)
sample_checks = {}
for policy in ('c1', 'own'):
    root = OUTPUTS / 'recoverability_current_policy_m0_20261004' / policy
    cfg = read(root / 'config.json')
    assert cfg['prefix_checkpoint_epoch'] == 25 and cfg['prefix_model'] == str(PREFIX)
    data, _, _, jobs = trainer.load_data([root], 'train', torch.device('cpu'))
    assert len(jobs) == 2
    output = trainer.forward(model, data)
    main_output = main_model({key: data[key] for key in modules.DECISION_FIELDS})
    assert output.keys() == main_output.keys()
    assert all(torch.equal(output[key], main_output[key]) for key in output)
    chosen = modules.select_actions(output, data, .03, 'action')
    main_chosen = main_modules.select_actions(main_output, data, .03, 'action')
    assert all(torch.equal(value, main_chosen[key]) for key, value in chosen.items())
    sample_checks[policy] = {'existing_collection_M0_rows': 2,
                             'all_forward_outputs_exact_to_frozen_main_class': True,
                             'all_selector_fields_exact_to_frozen_main_selector': True,
                             'new_training_or_accuracy_evidence': False}
references = {
    'baseline': OUTPUTS / 'abc_internal_validation_v1/baseline/predictions',
    'c1': OUTPUTS / 'abc_internal_validation_v1/c1/predictions',
    'global_old4': OUTPUTS / 'recoverability_write_pair_reference_own_b384_full_20261004/predictions',
    'matched_original_warm_c1': OUTPUTS / 'recoverability_warm_budgeted_action_c1_b384_full_20261004/predictions',
    'matched_original_warm_own': OUTPUTS / 'recoverability_warm_budgeted_action_own_b384_full_20261004/predictions',
}
reference_checks = {}
for name, root in references.items():
    done = read(root / 'inference_completion.json')
    cfg = read(root / 'inference_config.json')
    assert done['completed'] and done['sequences'] == 98 and done['frames'] == 49418
    assert cfg['validation_split'] == '/data/gb/outputs/c1_initial_seed42/split.json'
    assert cfg['root'] == '/data/wangwj/dataset/LasHeR/traingset'
    assert cfg['limit_sequences'] == 0 and cfg['max_frames'] == 0
    reference_checks[name] = {'root': str(root), 'complete_sequences': 98, 'complete_frames': 49418}
assert parent_sha == sha(PARENT) and prefix_sha == sha(PREFIX)
assert not torch.cuda.is_initialized()
record = {
    'status': 'PASS_CPU_SOURCE_PREPARATION',
    'source_sha256': source_hashes,
    'isolated_research_files': 39, 'all39_exact_to_accepted_warm_private_sources': True,
    'main_model_source_hashes_unchanged': True,
    'all_forward_classes_and_select_actions_AST_exact_to_main': True,
    'trainer_loss_hparams_and_init_unchanged_from_warm': True,
    'actual_import_paths': [trainer.__file__, modules.__file__],
    'protected_checkpoints': {
        'training_parent': {'path': str(PARENT), 'epoch': 4, 'sha256_before_after': parent_sha},
        'data_prefix': {'path': str(PREFIX), 'epoch': 25, 'sha256_before_after': prefix_sha},
        'strict25_tensor_load': True, 'frozen8_C1_tensors_exact': True, 'ABC_trainable_tensors': 17},
    'sample_source_checks': sample_checks,
    'full98_existing_reference_checks': reference_checks,
    'actual_new_M0_acceptance_missing_and_guard_refused': True,
    'new_cache_gate_missing': True, 'new_M0_and_full_roots_absent': True,
    'execution': {'CUDA_initialized': False, 'GPU_queries': 0, 'optimizer_steps': 0,
                  'jobs_started': 0, 'production_sources_modified': False,
                  'deleted_weights_read': False, 'weights_modified': False,
                  'elapsed_seconds': round(time.perf_counter() - START, 3)},
    'limits': 'Existing two-query collection M0 inputs check source behavior only. Complete new cache merge, actual 3-epoch/9-update fits, M0 acceptance and full98 accuracy remain pending.'}
(REVIEW / 'source_cpu_checks.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
