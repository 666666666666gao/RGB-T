"""CPU audit of real M0/full fits; selected-best replay is separate from terminal updates."""
import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--stage', choices=('m0', 'full'), required=True)
args = p.parse_args()
PRIVATE = Path('/data/gb/experiments/recoverability_module_freeze_20261004')
MAIN = Path('/data/gb/GOLA')
OUTPUTS = Path('/data/gb/outputs')
CACHE = OUTPUTS / 'recoverability_current_policy_merged_20261004'
PARENT = OUTPUTS / 'recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
C1 = OUTPUTS / 'c1_initial_seed42/best.pth'
REVIEW = Path('/data/gb/setup/module_freeze_runner_audit_source_review_20261004.json')
STAGING = Path('/data/gb/setup/module_freeze_source_staging_20261004.json')
ACCEPTANCE = Path('/data/gb/setup/module_freeze_m0_acceptance_20261004.json')
AUDIT = ACCEPTANCE if args.stage == 'm0' else Path('/data/gb/setup/module_freeze_actual_full_fit_cpu_audit_20261004.json')
FROZEN = {'bc': ['A'], 'ac': ['B'], 'ab': ['C'], 'c': ['A', 'B']}
EPOCHS = 3 if args.stage == 'm0' else 60
BATCH = 416
assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and Path.cwd() == PRIVATE
sys.path.insert(0, str(PRIVATE))
import numpy as np
import torch
import research.recoverability_modules as modules
import research.train_recoverability as trainer

torch.set_num_threads(4)
torch.set_grad_enabled(False)
assert not torch.cuda.is_initialized()
assert Path(modules.__file__).parent == Path(trainer.__file__).parent == PRIVATE / 'research'
started = time.perf_counter()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


review = read(REVIEW)
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family'
assert review['acceptance_status'] == 'provisional'
staging = read(STAGING)
assert staging['status'] == 'PASS'
source_hashes = staging['audited_source_sha256']
assert all(sha(path) == digest for path, digest in source_hashes.items())
gate = read(CACHE / 'paired_cache_gate.json')
assert gate['status'] == 'PASS' and gate['all_non_future_arrays_exact']
assert gate['matched_partitions'] == {'train': 902, 'validation': 128}
if args.stage == 'full':
    m0 = read(ACCEPTANCE)
    assert m0['status'] == 'PASS' and m0['all_four_arms_passed']
    assert m0['batch_size'] == BATCH and m0['epochs_per_arm'] == 3 and m0['optimizer_steps_per_arm'] == 9
    runner = PRIVATE / 'scripts/run_recoverability_module_freeze.sh'
    guard_line = next(line.strip() for line in runner.read_text().splitlines()
                      if line.strip().startswith('/data/gb/envs/gola/bin/python -c') and str(ACCEPTANCE) in line)
    opened = subprocess.run([sys.executable, '-c', shlex.split(guard_line)[2]], capture_output=True, text=True)
    assert opened.returncode == 0, opened.stderr
parent_sha = sha(PARENT)
assert parent_sha == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
parent = torch.load(PARENT, map_location='cpu', weights_only=False)
c1 = torch.load(C1, map_location='cpu', weights_only=False)
assert parent['module'] == 'ABC_recoverability' and parent['epoch'] == 4
assert len(parent['model']) == 25 and len(c1['head']) == 8
assert all(torch.equal(parent['model']['c1.' + key], value) for key, value in c1['head'].items())


def replay(model, data):
    return trainer.evaluate(model, data, BATCH, .03, details=True, search_supervision='oracle',
                            action_ranking='budgeted', write_pair_calibration=False, write_verification='action')


def metric_errors(actual, expected):
    assert set(actual) == set(expected)
    errors = {key: abs(actual[key] - expected[key]) for key in actual}
    assert all(value <= 2e-6 for value in errors.values()), errors
    return errors


def state_changes(state):
    result = {}
    for group, prefix in (('A', 'memory.'), ('B', 'search.'), ('C', 'action.')):
        keys = [key for key in parent['model'] if key.startswith(prefix)]
        result[group] = {
            'changed_tensors': sum(not torch.equal(state[key], parent['model'][key]) for key in keys),
            'total_tensors': len(keys),
            'changed_elements': sum(int((state[key] != parent['model'][key]).sum()) for key in keys),
            'max_abs_change': max(float((state[key] - parent['model'][key]).abs().max()) for key in keys),
        }
    return result


arms = {}
for variant, frozen in FROZEN.items():
    active = {name: name not in frozen for name in ('A', 'B', 'C')}
    root = OUTPUTS / ('recoverability_module_freeze_' + variant + '_b416_' + args.stage + '_20261004')
    required = ('config.json', 'completion.json', 'metrics.json', 'train.jsonl', 'best.pth',
                'best_validation.npz', 'training_completed.txt', 'job_completed.txt')
    assert all((root / name).is_file() for name in required)
    cfg, receipt, metrics = [read(root / name) for name in ('config.json', 'completion.json', 'metrics.json')]
    assert cfg['module'] == 'ABC_recoverability' and receipt['completed']
    assert cfg['epochs'] == receipt['epochs'] == EPOCHS and cfg['batch_size'] == BATCH
    assert cfg['seed'] == 42 and cfg['train_clips'] == 902 and cfg['validation_clips'] == 128
    assert cfg['lr'] == cfg['weight_decay'] == 1e-4 and cfg['threshold'] == .03
    assert cfg['search_supervision'] == 'oracle' and cfg['action_ranking'] == 'budgeted'
    assert cfg['write_verification'] == 'action' and cfg['write_pair_calibration'] is False
    assert not cfg['prefer_last_prefix'] and cfg['frozen_c1']
    assert cfg['frozen_modules'] == receipt['frozen_modules'] == frozen
    assert cfg['c1_head'] == str(C1) and cfg['init_checkpoint'] == str(PARENT)
    assert cfg['initial_checkpoint_epoch'] == 4 and type(cfg['initial_all_choices_match_c1']) is bool
    assert cfg['output'] == str(root)
    assert cfg['train'] == [str(CACHE / 'own/train')] and cfg['validation'] == str(CACHE / 'own/validation')
    source_configs = [read(CACHE / 'own' / part / 'config.json') for part in ('train', 'validation')]
    assert cfg['source_configs'] == source_configs
    split = read(source_configs[0]['split'])
    assert len(split['train']) == 881 and len(split['validation']) == 98
    assert not set(split['train']) & set(split['validation'])
    for part, count, source in zip(('train', 'validation'), (902, 128), source_configs):
        cache_done = read(CACHE / 'own' / part / 'completion.json')
        assert source['partition'] == cache_done['partition'] == part
        assert source['clips'] == cache_done['clips'] == count and cache_done['completed']
        assert not cache_done['decision_input_contains_future'] and source['head'] == str(C1)
        assert source['future_policy_mode'] == 'own' and source['prefix_checkpoint_epoch'] == 25
        assert source['prefix_write_verification'] == 'action' and source['max_prefix'] == 1024
        jobs = cfg[part + '_jobs']
        assert len(jobs) == len({tuple(job) for job in jobs}) == count
        assert {job[0] for job in jobs} <= set(split[part])
    logs = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    triplets = [[row['epoch'], row['step'], row['optimizer_steps']] for row in logs]
    assert triplets == [[epoch, step, (epoch - 1) * 3 + step]
                        for epoch in range(1, EPOCHS + 1) for step in (1, 3)]
    assert receipt['optimizer_steps'] == EPOCHS * math.ceil(902 / BATCH) == EPOCHS * 3
    assert all(np.isfinite(row['loss']) and all(np.isfinite(value) for value in row['parts'].values()) for row in logs)
    assert all(set(row['module_gradient_norms']) == {'A', 'B', 'C'} for row in logs)
    for row in logs:
        assert all(np.isfinite(value) and ((value > 0) if active[name] else value == 0)
                   for name, value in row['module_gradient_norms'].items())
    assert receipt['modules_changed'] == active
    assert receipt['frozen_module_weights_unchanged'] == {name: True for name in frozen}
    assert all(np.isfinite(value) and ((value > 0) if active[name] else value == 0)
               and value >= max(row['module_gradient_norms'][name] for row in logs)
               for name, value in receipt['max_module_gradient_norms'].items())
    assert receipt['frozen_c1_gradients_absent'] and receipt['strict_reload_metrics_equal']
    assert receipt['retained_weights'] == ['best.pth'] and not receipt['official_tracking_accuracy']
    assert sorted(path.name for path in root.glob('*.pth')) == ['best.pth']
    assert [row['epoch'] for row in metrics] == list(range(EPOCHS + 1))
    assert all(np.isfinite(value) for row in metrics for value in row.values())
    best_epoch = max(range(EPOCHS + 1), key=lambda epoch: metrics[epoch]['utility'])
    ck = torch.load(root / 'best.pth', map_location='cpu', weights_only=False)
    assert ck['module'] == 'ABC_recoverability' and ck['epoch'] == receipt['best_epoch'] == best_epoch
    assert ck['validation'] == receipt['best_validation'] == {key: value for key, value in metrics[best_epoch].items() if key != 'epoch'}
    assert receipt['last_validation'] == {key: value for key, value in metrics[EPOCHS].items() if key != 'epoch'}
    assert all(cfg[key] == value for key, value in ck['args'].items())
    assert ck['selection_policy'] == cfg['checkpoint_selection']
    assert len(ck['model']) == 25 and set(ck['model']) == set(parent['model'])
    assert all(torch.equal(ck['model']['c1.' + key], value) for key, value in c1['head'].items())
    changes = state_changes(ck['model'])
    for name, change in changes.items():
        assert change['changed_tensors'] == 0 if name in frozen or best_epoch == 0 else change['changed_tensors'] > 0
    model = modules.RecoverabilityModules(c1).cpu().eval()
    model.load_state_dict(parent['model'], strict=True)
    for name, block in (('A', model.memory), ('B', model.search), ('C', model.action)):
        if name in frozen:
            block.requires_grad_(False)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.c1.parameters())
    model.train()
    assert not model.c1.training
    model.eval()
    counts = {name: sum(parameter.numel() for parameter in block.parameters() if parameter.requires_grad)
              for name, block in (('A', model.memory), ('B', model.search), ('C', model.action))}
    totals = {'A': 109187, 'B': 117133, 'C': 86791}
    assert counts == cfg['trainable_parameters'] == {name: total if active[name] else 0 for name, total in totals.items()}
    data, _, _, jobs = trainer.load_data([cfg['validation']], 'validation', torch.device('cpu'))
    assert [list(job) for job in jobs] == cfg['validation_jobs']
    initial_metrics, initial_values = replay(model, data)
    initial_error = metric_errors(initial_metrics, {key: value for key, value in metrics[0].items() if key != 'epoch'})
    val_parity = np.array_equal(initial_values['flat_action'], data['original_choice'].numpy() * 2) and not initial_values['search_triggered'].any()
    assert not cfg['initial_all_choices_match_c1'] or val_parity
    model.load_state_dict(ck['model'], strict=True)
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    measured, values = (initial_metrics, initial_values) if best_epoch == 0 else replay(model, data)
    best_error = metric_errors(measured, ck['validation'])
    with np.load(root / 'best_validation.npz', allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    assert set(saved) == set(values) and len(saved) == 21
    field_errors = {}
    for key, value in values.items():
        assert value.shape == saved[key].shape == (128,) and np.isfinite(saved[key]).all()
        field_errors[key] = float(np.abs(value.astype(float) - saved[key].astype(float)).max())
        if value.dtype.kind in 'biu':
            assert np.array_equal(value, saved[key]), (variant, key)
        else:
            assert field_errors[key] <= np.finfo(np.float32).eps, (variant, key, field_errors[key])
    arms[variant] = {
        'status': 'PASS', 'root': str(root), 'frozen_modules': frozen, 'active_modules': [name for name in active if active[name]],
        'epochs': EPOCHS, 'optimizer_steps': EPOCHS * 3, 'logged_step_triplets': triplets,
        'terminal_modules_changed': receipt['modules_changed'], 'terminal_frozen_weights_unchanged': receipt['frozen_module_weights_unchanged'],
        'terminal_update_evidence': 'Source-linked real receipt and optimizer logs before best reload; no last weights retained.',
        'max_module_gradient_norms': receipt['max_module_gradient_norms'], 'trainable_parameters': counts,
        'best_epoch': best_epoch, 'best0_is_parent_not_new_learning': best_epoch == 0,
        'selected_best_changes_vs_parent': changes, 'C1_all8_retained_tensors_exact': True,
        'strict25_tensor_CPU_load': True, 'epoch0_parent_VAL128_replay': True,
        'initial_metric_max_abs_errors': initial_error, 'best_metric_max_abs_errors': best_error,
        'all21_VAL128_fields_CPU_reproduced': True, 'per_field_max_abs_errors': field_errors,
        'best_validation': ck['validation'], 'peak_cuda_mib': receipt['peak_cuda_mib'],
        'artifact_sha256': {name: sha(root / name) for name in required}, 'learning_gain_claimed': False,
    }
    print('ARM_PASS ' + variant + ' best_epoch=' + str(best_epoch), file=sys.stderr, flush=True)
    del model, data, ck, initial_values, values, saved

assert set(arms) == set(FROZEN) and all(row['status'] == 'PASS' for row in arms.values())
assert all(sha(path) == digest for path, digest in source_hashes.items())
assert sha(PARENT) == parent_sha and not torch.cuda.is_initialized()
record = {
    'status': 'PASS', 'all_four_arms_passed': True, 'stage': args.stage, 'batch_size': BATCH,
    'epochs_per_arm': EPOCHS, 'optimizer_steps_per_arm': EPOCHS * 3,
    'review_independence': 'same-family', 'acceptance_status': 'provisional',
    'accepted_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'scope': 'Real training and retained-tensor CPU replay; complete ABC still executes. No continuous or native improvement claim.',
    'source_review': str(REVIEW), 'source_staging': str(STAGING), 'source_sha256': source_hashes,
    'parent_checkpoint': str(PARENT), 'parent_sha256_before_and_after': parent_sha,
    'actual_import_paths': [trainer.__file__, modules.__file__], 'arms': arms,
    'limitations': ['Best selection uses the same128 developer VAL queries.',
                   'Terminal update evidence is receipt/log based; retained-best differences are independently measured.',
                   'Batch416 differs from the prior joint-ABC batch384 reference; this is not an isolated module ablation.'],
    'execution': {'CUDA_initialized': False, 'GPU_queries': 0, 'optimizer_steps': 0,
                  'TRAIN_embeddings_read': False, 'jobs_started': 0, 'weights_modified': False,
                  'runtime_seconds': round(time.perf_counter() - started, 3)},
}
AUDIT.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': 'PASS', 'stage': args.stage, 'arms': {key: value['best_epoch'] for key, value in arms.items()},
                  'receipt': str(AUDIT)}, allow_nan=False))
