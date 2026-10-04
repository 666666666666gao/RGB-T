"""Source-only checks of joint-B416 routing; imports no production modules."""
import argparse
import ast
import copy
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from types import SimpleNamespace

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
AUDITOR = RUN / 'actual_joint_b416_fit_cpu_audit.py'
RUNNER = ROOT / 'scripts/run_recoverability_joint_b416.sh'
PRIOR = ROOT / 'refine-logs/runs/recoverability_module_freeze'


def assignment(body, name):
    return next(node for node in body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def execute(nodes, namespace):
    exec(compile(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]), '<source fixture>', 'exec'), namespace)


def evaluate(node, namespace):
    return eval(compile(ast.Expression(body=copy.deepcopy(node)), '<source fixture>', 'eval'), namespace)


def rejected(code, namespace):
    try:
        exec(code, namespace)
    except (AssertionError, FileNotFoundError):
        return
    raise AssertionError('Invalid fixture was accepted')


source, runner = AUDITOR.read_text(), RUNNER.read_text()
tree = ast.parse(source)
compile(tree, str(AUDITOR), 'exec')
trainer = ast.parse((ROOT / 'research/train_recoverability.py').read_text())
modules = ast.parse((ROOT / 'research/recoverability_modules.py').read_text())
plan = json.loads((RUN / 'plan.json').read_text())
staging = json.loads((PRIOR / 'source_staging.json').read_text())
for name in ('train_recoverability.py', 'recoverability_modules.py'):
    accepted = staging['audited_source_sha256'][staging['private_source'] + '/research/' + name]
    assert hashlib.sha256((ROOT / 'research' / name).read_bytes()).hexdigest() == accepted

expected = (PRIOR / 'actual_module_freeze_fit_cpu_audit.py').read_text()
substitutions = [
    ('recoverability_module_freeze_20261004', 'recoverability_joint_b416_20261004'),
    ('module_freeze_runner_audit_source_review_20261004', 'joint_b416_source_review_20261004'),
    ('module_freeze_source_staging_20261004', 'joint_b416_source_staging_20261004'),
    ('module_freeze_m0_acceptance_20261004', 'joint_b416_m0_acceptance_20261004'),
    ('module_freeze_actual_full_fit_cpu_audit_20261004', 'joint_b416_actual_full_fit_cpu_audit_20261004'),
    ("FROZEN = {'bc': ['A'], 'ac': ['B'], 'ab': ['C'], 'c': ['A', 'B']}", "FROZEN = {'budgeted': [], 'pairwise': []}"),
    ('all_four_arms_passed', 'both_ranking_arms_passed'),
    ('run_recoverability_module_freeze.sh', 'run_recoverability_joint_b416.sh'),
    ('def replay(model, data):', 'def replay(model, data, ranking):'),
    ("action_ranking='budgeted'", 'action_ranking=ranking'),
    ("'recoverability_module_freeze_' + variant + '_b416_'", "'recoverability_joint_b416_' + variant + '_'"),
    ("cfg['action_ranking'] == 'budgeted'", "cfg['action_ranking'] == variant"),
    ('replay(model, data)', 'replay(model, data, variant)'),
    ('Batch416 differs from the prior joint-ABC batch384 reference; this is not an isolated module ablation.',
     'Both completeABC ranking arms match batch416; comparisons against the priorABC batch384 reference retain the batch limitation.'),
]
for before, after in substitutions:
    assert before in expected, before
    expected = expected.replace(before, after)
assert source == expected
bash = r'E:\dcda6-main\Git\bin\bash.exe'
syntax = subprocess.run([bash, '-n', str(RUNNER)], capture_output=True, text=True)
assert syntax.returncode == 0, syntax.stderr
assert 'set -euo pipefail' in runner and 'test ! -e "$run"' in runner
assert '[[ "$ranking" == budgeted || "$ranking" == pairwise ]]' in runner
assert '[[ "$stage" == m0 || "$stage" == full ]]' in runner
assert runner.index('cd "$private_source"') < runner.index('print("TRAINING_SOURCE"') < runner.index('python -u -m research.train_recoverability')
assert runner.index('python -u -m research.train_recoverability') < runner.index('date -Iseconds > "$run/training_completed.txt"') < runner.index('date -Iseconds > "$run/job_completed.txt"')
assert 'evaluate_recoverability' not in runner

gates = [shlex.split(line.strip())[2] for line in runner.splitlines()
         if line.strip().startswith('/data/gb/envs/gola/bin/python -c') and 'import json;' in line]
payloads = [
    {'status': 'PASS', 'all_non_future_arrays_exact': True, 'matched_partitions': {'train': 902, 'validation': 128}},
    {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional'},
    {'status': 'PASS', 'both_ranking_arms_passed': True, 'batch_size': 416, 'optimizer_steps_per_arm': 9, 'epochs_per_arm': 3},
]
assert len(gates) == len(payloads)
guard_cases = []
for code, payload in zip(gates, payloads):
    exec(code, {'open': lambda path: io.StringIO(json.dumps(payload))})
    def missing(path):
        raise FileNotFoundError(path)
    rejected(code, {'open': missing})
    for key, value in payload.items():
        invalid = dict(payload)
        invalid[key] = False if isinstance(value, bool) else (value + 1 if isinstance(value, int) else 'INVALID')
        rejected(code, {'open': lambda path: io.StringIO(json.dumps(invalid))})
    guard_cases.append({'valid_passed': True, 'missing_rejected': True, 'invalid_fields_rejected': list(payload)})
assert runner.index('if [[ "$stage" == full ]]') < runner.index('joint_b416_m0_acceptance_20261004.json') < runner.index('epochs=60')

frozen = ast.literal_eval(assignment(tree.body, 'FROZEN').value)
assert frozen == {'budgeted': [], 'pairwise': []}
arm = next(node for node in tree.body if isinstance(node, ast.For) and ast.unparse(node.iter) == 'FROZEN.items()')
parser = {'argparse': argparse, '__doc__': 'Source parser fixture'}
execute([function(trainer, 'arguments')], parser)
command = next(line for line in runner.replace('\\\n', ' ').splitlines() if line.startswith('/data/gb/envs/gola/bin/python -u -m '))
tokens = shlex.split(command)[4:]
cases, configs = [], {}
for variant, groups in frozen.items():
    for stage, epochs in (('m0', 3), ('full', 60)):
        namespace = {'OUTPUTS': PurePosixPath('/data/gb/outputs'), 'variant': variant, 'args': SimpleNamespace(stage=stage)}
        root = str(evaluate(assignment(arm.body, 'root').value, namespace))
        assert root == f'/data/gb/outputs/recoverability_joint_b416_{variant}_{stage}_20261004'
        expanded = [token.replace('$cache', '/data/gb/outputs/recoverability_current_policy_merged_20261004')
                    .replace('$epochs', str(epochs)).replace('$run', root).replace('$ranking', variant) for token in tokens]
        sys.argv = ['fixture', *expanded]
        cfg = vars(parser['arguments']())
        assert cfg['frozen_modules'] == groups == []
        assert cfg['batch_size'] == plan['batch_size'] == 416 and cfg['epochs'] == plan[stage + '_epochs'] == epochs
        assert math.ceil(plan['train_clips'] / cfg['batch_size']) * epochs == plan[stage + '_steps']
        assert cfg['seed'] == plan['seed'] == 42 and cfg['lr'] == cfg['weight_decay'] == 1e-4 and cfg['threshold'] == .03
        assert cfg['search_supervision'] == 'oracle' and cfg['action_ranking'] == variant and cfg['write_verification'] == 'action'
        assert cfg['train'] == ['/data/gb/outputs/recoverability_current_policy_merged_20261004/own/train']
        assert cfg['validation'] == '/data/gb/outputs/recoverability_current_policy_merged_20261004/own/validation'
        assert cfg['init_checkpoint'] == plan['parent'] and cfg['output'] == root
        assert cfg['c1_head'] == '/data/gb/outputs/c1_initial_seed42/best.pth'
        assert not cfg['write_pair_calibration'] and not cfg['prefer_last_prefix']
        configs[variant, stage] = cfg
        cases.append({'ranking': variant, 'stage': stage, 'root': root, 'epochs': epochs, 'optimizer_steps': epochs * 3})
for stage in ('m0', 'full'):
    assert {key for key in configs['budgeted', stage] if configs['budgeted', stage][key] != configs['pairwise', stage][key]} == {'action_ranking', 'output'}
assert plan['all_trainable_modules'] == ['A', 'B', 'C'] and plan['gpu_assignments'] == {'budgeted': 0, 'pairwise': 3}
assert plan['source'] == '/data/gb/experiments/recoverability_joint_b416_20261004' and plan['validation_clips'] == 128

def capture(*args, **kwargs):
    return args, kwargs
replay_ns = {'trainer': SimpleNamespace(evaluate=capture), 'BATCH': 416}
execute([function(tree, 'replay')], replay_ns)
for variant in frozen:
    positional, keywords = replay_ns['replay']('model', 'data', variant)
    assert positional == ('model', 'data', 416, .03)
    assert keywords == {'details': True, 'search_supervision': 'oracle', 'action_ranking': variant, 'write_pair_calibration': False, 'write_verification': 'action'}
calls = [node for node in ast.walk(arm) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'replay']
assert len(calls) == 2 and all(ast.unparse(call.args[-1]) == 'variant' for call in calls)
best_expr = assignment(arm.body, 'best_epoch').value
assert evaluate(best_expr, {'EPOCHS': 3, 'metrics': [{'utility': value} for value in (.5, .5, .4, .5)]}) == 0
assert evaluate(best_expr, {'EPOCHS': 3, 'metrics': [{'utility': value} for value in (.4, .6, .6, .5)]}) == 1
state_assert = next(node for node in ast.walk(arm) if isinstance(node, ast.Assert) and isinstance(node.test, ast.IfExp))
for best_epoch, changed in ((0, 0), (1, 1)):
    for name in ('A', 'B', 'C'):
        namespace = {'name': name, 'change': {'changed_tensors': changed}, 'frozen': [], 'best_epoch': best_epoch}
        execute([state_assert], namespace)
        namespace['change'] = {'changed_tensors': 1 - changed}
        rejected(compile(ast.Module(body=[state_assert], type_ignores=[]), '<source fixture>', 'exec'), namespace)
decision = ast.literal_eval(assignment(modules.body, 'DECISION_FIELDS').value)
labels = ast.literal_eval(assignment(trainer.body, 'LABEL_FIELDS').value)
assert not set(decision) & set(labels)
selector_fields = {node.slice.value for node in ast.walk(function(modules, 'select_actions'))
                   if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'data' and isinstance(node.slice, ast.Constant)}
assert selector_fields <= set(decision)
details = next(node for node in ast.walk(function(trainer, 'evaluate')) if isinstance(node, ast.Dict)
               and any(isinstance(key, ast.Constant) and key.value == 'flat_action' for key in node.keys))
assert len(details.keys) == 21
assert "torch" not in sys.modules and "numpy" not in sys.modules
report = {
    'status': 'PASS', 'scope': 'Source syntax, accepted-source comparison, literal gates, parser/routing and scalar fixtures only',
    'bash_syntax': 'PASS', 'python_compile': 'PASS', 'production_sources_equal_accepted_freeze_staging': True,
    'auditor_exact_intended_substitutions': True, 'guard_cases': guard_cases, 'command_cases': cases,
    'both_replays_use_actual_arm_ranking': True, 'only_ranking_and_output_differ_between_arms': True,
    'strict_first_max_and_best0_vs_trained_state_cases': 'PASS', 'decision_label_fields_disjoint': True,
    'selector_reads_only_decision_fields': True, 'evaluator_detail_fields': 21,
    'python_version': sys.version, 'torch_imported': False, 'numpy_imported': False, 'neural_execution': False,
    'ssh_used': False, 'gpu_used': False, 'runtime_certification': False,
}
(RUN / 'source_checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PASS', 'guard_cases': len(guard_cases), 'command_cases': len(cases), 'replay_modes': list(frozen), 'runtime_certification': False}))
