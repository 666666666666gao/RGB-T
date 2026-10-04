"""Source-only review fixtures. No production imports, datasets or checkpoints."""
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
PRIOR = ROOT / 'refine-logs/runs/recoverability_joint_b416'
RUNNER = ROOT / 'scripts/run_recoverability_best_policy_fit.sh'
AUDITOR = RUN / 'actual_best_policy_fit_cpu_audit.py'
STAGER = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/stage_best_policy_fit_20261004.py')
LAUNCHER = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/launch_best_policy_fit_20261004.py')
BASH = r'E:\dcda6-main\Git\bin\bash.exe'


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
    except (AssertionError, FileNotFoundError, KeyError):
        return
    raise AssertionError('Invalid fixture was accepted')


source, runner = AUDITOR.read_text(), RUNNER.read_text()
tree = ast.parse(source)
trainer = ast.parse((ROOT / 'research/train_recoverability.py').read_text())
modules = ast.parse((ROOT / 'research/recoverability_modules.py').read_text())
launch = ast.parse(LAUNCHER.read_text())
plan = json.loads((RUN / 'plan.json').read_text())
python_sources = [AUDITOR, ROOT / 'research/train_recoverability.py', ROOT / 'research/recoverability_modules.py',
                  ROOT / 'research/candidate_learning.py', ROOT / 'research/temporal_modules.py',
                  RUN / 'memory_monitor.py', STAGER, LAUNCHER,
                  ROOT / 'refine-logs/runs/recoverability_best_policy/actual_completed_cache_gate_cpu.py',
                  ROOT / 'refine-logs/runs/recoverability_best_policy/actual_partition_merge_cpu.py']
for path in python_sources:
    compile(path.read_text(), str(path), 'exec')
accepted = {'train_recoverability.py': 'f1089cf0d8e8ebbc2d1f83ac60bff0267edd4aa6c1c3c717829393e4d7622474',
            'recoverability_modules.py': '082945bc2742c046dc75439ec41e2d37e209974d4afc72a8f1cbb5b096824aaa'}
for name, digest in accepted.items():
    assert hashlib.sha256((ROOT / 'research' / name).read_bytes()).hexdigest() == digest
assert len(list((ROOT / 'research').rglob('*.py'))) == 39

expected = (PRIOR / 'actual_joint_b416_fit_cpu_audit.py').read_text()
substitutions = [
    ('recoverability_joint_b416_20261004', 'recoverability_best_policy_fit_20261004'),
    ('recoverability_current_policy_merged_20261004', 'recoverability_best_policy_merged_20261004'),
    ('joint_b416_source_review_20261004', 'best_policy_fit_source_review_20261004'),
    ('joint_b416_source_staging_20261004', 'best_policy_fit_source_staging_20261004'),
    ('joint_b416_m0_acceptance_20261004', 'best_policy_fit_m0_acceptance_20261004'),
    ('joint_b416_actual_full_fit_cpu_audit_20261004', 'best_policy_fit_actual_full_fit_cpu_audit_20261004'),
    ("FROZEN = {'budgeted': [], 'pairwise': []}", "ARMS = {'pairwise_lr4': ('pairwise', 1e-4), 'pairwise_lr5': ('pairwise', 1e-5), 'budgeted_lr4': ('budgeted', 1e-4), 'budgeted_lr5': ('budgeted', 1e-5)}"),
    ("gate = read(CACHE / 'paired_cache_gate.json')", "gate = read(CACHE / 'best_policy_cache_cpu_gate.json')"),
    ("gate['status'] == 'PASS' and gate['all_non_future_arrays_exact']", "gate['status'] == 'PASS' and gate['all_ordered_jobs_exact'] and gate['all_GT_current_and_history_replayed'] and gate['all_schema_and_sources_verified']"),
    ('both_ranking_arms_passed', 'all_four_lr_ranking_arms_passed'),
    ('run_recoverability_joint_b416.sh', 'run_recoverability_best_policy_fit.sh'),
    ('for variant, frozen in FROZEN.items():', 'for variant, (ranking, lr) in ARMS.items():\n    frozen = []'),
    ("'recoverability_joint_b416_' + variant", "'recoverability_best_policy_' + variant"),
    ("cfg['lr'] == cfg['weight_decay'] == 1e-4", "cfg['lr'] == lr and cfg['weight_decay'] == 1e-4"),
    ("cfg['action_ranking'] == variant", "cfg['action_ranking'] == ranking"),
    ("source['prefix_checkpoint_epoch'] == 25", "source['prefix_checkpoint_epoch'] == source['future_checkpoint_epoch'] == 4"),
    ('replay(model, data, variant)', 'replay(model, data, ranking)'),
    ('set(arms) == set(FROZEN)', 'set(arms) == set(ARMS)'),
]
for before, after in substitutions:
    assert before in expected, before
    expected = expected.replace(before, after)
assert source == expected
assert (RUN / 'memory_monitor.py').read_text() == (PRIOR / 'memory_monitor.py').read_text().replace('joint_b416_', 'best_policy_fit_')
syntax = subprocess.run([BASH, '-n', str(RUNNER)], capture_output=True, text=True)
assert syntax.returncode == 0, syntax.stderr

gate_lines = [line for line in runner.splitlines()
              if line.strip().startswith('/data/gb/envs/gola/bin/python -c') and 'import json;' in line]
gates = [shlex.split(line.strip())[2] for line in gate_lines]
payloads = [
    {'status': 'PASS', 'matched_partitions': {'train': 902, 'validation': 128}, 'all_ordered_jobs_exact': True,
     'all_GT_current_and_history_replayed': True, 'all_schema_and_sources_verified': True, 'prefix_epoch': 4, 'future_epoch': 4},
    {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional'},
    {'status': 'PASS', 'all_four_lr_ranking_arms_passed': True, 'batch_size': 416, 'optimizer_steps_per_arm': 9, 'epochs_per_arm': 3},
]
assert len(gates) == len(payloads) == 3
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
    guard_cases.append({'valid_passed': 1, 'missing_rejected': 1, 'invalid_fields_rejected': list(payload)})
old_m0 = payloads[-1].copy()
old_m0['both_ranking_arms_passed'] = old_m0.pop('all_four_lr_ranking_arms_passed')
rejected(gates[-1], {'open': lambda path: io.StringIO(json.dumps(old_m0))})
assert runner.index('best_policy_cache_cpu_gate.json') < runner.index('if [[ "$stage" == full ]]')
assert runner.index('if [[ "$stage" == full ]]') < runner.index('best_policy_fit_m0_acceptance_20261004.json') < runner.index('epochs=60')
assert runner.index('cd "$private"') < runner.index('print("TRAINING_SOURCE"') < runner.index('python -u -m research.train_recoverability')
assert runner.index('python -u -m research.train_recoverability') < runner.index('date -Iseconds > "$run/training_completed.txt"') < runner.index('date -Iseconds > "$run/job_completed.txt"')
assert 'test ! -e "$run"' in runner and 'set -euo pipefail' in runner
assert 'evaluate_recoverability' not in runner and 'full98' not in runner

arms = ast.literal_eval(assignment(tree.body, 'ARMS').value)
assert arms == {name: (row['ranking'], row['lr']) for name, row in plan['arms'].items()}
assignments = ast.literal_eval(assignment(launch.body, 'assignments').value)
assert assignments == [('pairwise_lr4', 0), ('pairwise_lr5', 1), ('budgeted_lr4', 2), ('budgeted_lr5', 3)]
assert dict(assignments) == {name: row['gpu'] for name, row in plan['arms'].items()}
launch_loop = next(node for node in launch.body if isinstance(node, ast.For))
popen = next(node for node in ast.walk(launch_loop) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'subprocess.Popen')
for stage in ('m0', 'full'):
    roots = evaluate(assignment(launch.body, 'roots').value,
                     {'Path': PurePosixPath, 'assignments': assignments, 'stage': stage})
    for (variant, gpu), root in zip(assignments, roots):
        assert str(root) == f'/data/gb/outputs/recoverability_best_policy_{variant}_{stage}_20261004'
        command_args = evaluate(popen.args[0], {'private': PurePosixPath(plan['private_source']),
                                               'gpu': gpu, 'arm': variant, 'stage': stage})
        assert command_args == ['bash', plan['private_source'] + '/scripts/run_recoverability_best_policy_fit.sh',
                                str(gpu), variant, stage]
arm_loop = next(node for node in tree.body if isinstance(node, ast.For) and ast.unparse(node.iter) == 'ARMS.items()')
assert ast.literal_eval(assignment(arm_loop.body, 'frozen').value) == []
parser = {'argparse': argparse, '__doc__': 'Source parser fixture'}
execute([function(trainer, 'arguments')], parser)
command = next(line for line in runner.replace('\\\n', ' ').splitlines() if line.startswith('/data/gb/envs/gola/bin/python -u -m '))
tokens = shlex.split(command)[4:]
# Execute only the source's scalar routing fragment. All three Python gate
# commands are removed here and independently exercised above using JSON fixtures.
routing = runner[:runner.index('test ! -e "$run"')]
for line in gate_lines:
    routing = routing.replace(line, ':')
assert '/data/gb/envs/gola/bin/python' not in routing
routing += '\nexport CUDA_VISIBLE_DEVICES="$gpu"\nprintf "%s\\n" "$arm" "$ranking" "$lr" "$epochs" "$run" "$CUDA_VISIBLE_DEVICES"\n'
command_cases, configs = [], {}
for variant, gpu in assignments:
    ranking, lr = arms[variant]
    for stage, epochs in (('m0', 3), ('full', 60)):
        routed = subprocess.run([BASH, '-s', '--', str(gpu), variant, stage], input=routing, capture_output=True, text=True)
        assert routed.returncode == 0, routed.stderr
        arm_name, actual_ranking, actual_lr, actual_epochs, root, actual_gpu = routed.stdout.splitlines()
        namespace = {'OUTPUTS': PurePosixPath('/data/gb/outputs'), 'variant': variant, 'args': SimpleNamespace(stage=stage)}
        assert root == str(evaluate(assignment(arm_loop.body, 'root').value, namespace))
        assert (arm_name, actual_ranking, float(actual_lr), int(actual_epochs), int(actual_gpu)) == (variant, ranking, lr, epochs, gpu)
        expanded = [token.replace('$cache', plan['cache']).replace('$epochs', str(epochs)).replace('$run', root)
                    .replace('$ranking', actual_ranking).replace('$lr', actual_lr) for token in tokens]
        sys.argv = ['fixture', *expanded]
        cfg = vars(parser['arguments']())
        assert cfg['frozen_modules'] == [] and not cfg['write_pair_calibration'] and not cfg['prefer_last_prefix']
        assert cfg['batch_size'] == plan['batch_size'] == 416 and cfg['epochs'] == plan[stage + '_epochs'] == epochs
        assert math.ceil(plan['train_clips'] / cfg['batch_size']) * epochs == plan[stage + '_optimizer_steps']
        assert cfg['seed'] == plan['seed'] == 42 and cfg['lr'] == lr
        assert cfg['weight_decay'] == plan['weight_decay'] == 1e-4 and cfg['threshold'] == plan['threshold'] == .03
        assert cfg['search_supervision'] == 'oracle' and cfg['action_ranking'] == ranking and cfg['write_verification'] == 'action'
        assert cfg['train'] == [plan['cache'] + '/own/train'] and cfg['validation'] == plan['cache'] + '/own/validation'
        assert cfg['init_checkpoint'] == plan['parent'] and cfg['output'] == root
        assert cfg['c1_head'] == '/data/gb/outputs/c1_initial_seed42/best.pth'
        configs[variant, stage] = cfg
        command_cases.append({'arm': variant, 'gpu': gpu, 'ranking': ranking, 'lr': lr, 'stage': stage,
                              'root': root, 'epochs': epochs, 'optimizer_steps': epochs * 3})
for variant, stage in (('pairwise', 'm0'), ('invalid', 'm0'), ('pairwise_lr4', 'invalid')):
    invalid = subprocess.run([BASH, '-s', '--', '0', variant, stage], input=routing, capture_output=True, text=True)
    assert invalid.returncode != 0
for stage in ('m0', 'full'):
    for left, right, differing in (
        ('pairwise_lr4', 'pairwise_lr5', {'lr', 'output'}),
        ('budgeted_lr4', 'budgeted_lr5', {'lr', 'output'}),
        ('pairwise_lr4', 'budgeted_lr4', {'action_ranking', 'output'}),
        ('pairwise_lr5', 'budgeted_lr5', {'action_ranking', 'output'}),
    ):
        assert {key for key in configs[left, stage] if configs[left, stage][key] != configs[right, stage][key]} == differing

def capture(*args, **kwargs):
    return args, kwargs
replay_ns = {'trainer': SimpleNamespace(evaluate=capture), 'BATCH': 416}
execute([function(tree, 'replay')], replay_ns)
for variant, (ranking, lr) in arms.items():
    positional, keywords = replay_ns['replay']('model', 'data', ranking)
    assert positional == ('model', 'data', 416, .03)
    assert keywords == {'details': True, 'search_supervision': 'oracle', 'action_ranking': ranking,
                        'write_pair_calibration': False, 'write_verification': 'action'}
calls = [node for node in ast.walk(arm_loop) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'replay']
assert len(calls) == 2 and all(ast.unparse(call.args[-1]) == 'ranking' for call in calls)
best_expr = assignment(arm_loop.body, 'best_epoch').value
assert evaluate(best_expr, {'EPOCHS': 3, 'metrics': [{'utility': value} for value in (.5, .5, .4, .5)]}) == 0
assert evaluate(best_expr, {'EPOCHS': 3, 'metrics': [{'utility': value} for value in (.4, .6, .6, .5)]}) == 1
state_assert = next(node for node in ast.walk(arm_loop) if isinstance(node, ast.Assert) and isinstance(node.test, ast.IfExp))
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
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
record = {
    'status': 'PASS', 'scope': 'Source syntax, scalar shell routing, argparse, literal gates and AST fixtures only',
    'compiled_python_sources': len(python_sources), 'bash_syntax': 'PASS',
    'accepted_trainer_model_hashes_exact': True, 'local_research_source_count': 39,
    'auditor_equals_accepted_joint_with_intended_substitutions': True,
    'memory_monitor_equals_accepted_joint_with_label_replacement': True,
    'literal_guard_cases': guard_cases, 'literal_guard_valid_count': 3,
    'literal_guard_rejection_count': sum(1 + len(payload) for payload in payloads) + 1,
    'old_two_arm_acceptance_rejected': True, 'command_cases': command_cases,
    'command_case_count': len(command_cases), 'invalid_arm_stage_rejections': 3,
    'external_launcher_source_argv_cases': 8,
    'ranking_lr_factorial_only_expected_config_differences': True, 'ranking_cpu_replay_routes': 4,
    'first_max_cases': 2, 'selected_best_state_cases': 12,
    'decision_fields': len(decision), 'label_fields': len(labels), 'decision_label_fields_disjoint': True,
    'selector_reads_only_decision_fields': True, 'evaluator_detail_fields': len(details.keys),
    'python_version': sys.version, 'torch_imported': False, 'numpy_imported': False,
    'NN_execution': False, 'SSH_calls': 0, 'GPU_calls': 0, 'actual_data_or_checkpoint_reads': 0,
    'staging_or_launcher_executed': False, 'runtime_certification': False,
}
(RUN / 'source_checks.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PASS', 'commands': len(command_cases), 'gate_rejections': record['literal_guard_rejection_count'],
                  'python_sources_compiled': len(python_sources), 'runtime_certification': False}))
