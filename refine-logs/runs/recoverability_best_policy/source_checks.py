"""Source-only checks; no production import, actual-probe receipt, NN or dataset read."""
import argparse
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
RUNNER = ROOT / 'scripts/run_recoverability_best_policy_collect.sh'
COLLECTOR = ROOT / 'research/collect_recoverability.py'
AUDITOR = RUN / 'actual_probe_cpu_audit.py'

runner = RUNNER.read_text(encoding='utf-8')
collector_tree = ast.parse(COLLECTOR.read_text(encoding='utf-8'))
audit_tree = ast.parse(AUDITOR.read_text(encoding='utf-8'))
compile(collector_tree, str(COLLECTOR), 'exec')
compile(audit_tree, str(AUDITOR), 'exec')
subprocess.run([sys.argv[1], '-n', str(RUNNER)], check=True)

main = next(node for node in collector_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
parser_nodes = []
for node in main.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets):
        break
    parser_nodes.append(node)
parser_scope = {'argparse': argparse, '__doc__': ''}
exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), str(COLLECTOR), 'exec'), parser_scope)
parser = parser_scope['p']
jobs_block = next(node for node in main.body if isinstance(node, ast.If) and ast.unparse(node.test) == 'args.jobs_file')
jobs_code = compile(ast.Module(body=[jobs_block], type_ignores=[]), str(COLLECTOR), 'exec')
command = runner[runner.index('/data/gb/envs/gola/bin/python -u -m'):runner.index('\ndate -Iseconds')]
tokens = shlex.split(command.replace('\\\n', ' '))
tokens = tokens[tokens.index('--root'):]
plan = json.loads((RUN / 'plan.json').read_text())
cases = []
partitions = {}
for part in ('train', 'validation'):
    previous = json.loads((ROOT / 'refine-logs/runs/recoverability_write_pair/future_policy_jobs' / (part + '.json')).read_text())
    probe = json.loads((RUN / ('jobs_' + part + '_probe.json')).read_text())['jobs']
    rest = json.loads((RUN / ('jobs_' + part + '_rest.json')).read_text())['jobs']
    assert probe + rest == previous
    assert len(probe) == 16 and len(rest) == plan['remaining_jobs'][part]
    assert len({(job['sequence'], job['query_frame']) for job in previous}) == len(previous)
    partitions[part] = {job['sequence'] for job in previous}
    names = list(dict.fromkeys(job['sequence'] for job in previous))
    dataset = [SimpleNamespace(get_name=lambda name=name: name) for name in names]
    eligible = [(index, [job['query_frame'] for job in previous if job['sequence'] == name])
                for index, name in enumerate(names)]
    for stage, expected in (('probe', probe), ('rest', rest)):
        run = '/data/gb/outputs/recoverability_best_policy_collect_' + part + '_' + stage + '_20261004'
        cli = [token.replace('${partition}', part).replace('$partition', part)
               .replace('${stage}', stage).replace('$stage', stage).replace('$run', run) for token in tokens]
        args = parser.parse_args(cli)
        assert args.output == run and args.partition == part
        assert args.clips == args.batch_clips == 16 and args.forward_batch == 64 and args.seed == 42
        assert args.prefix_model == plan['prefix_model'] and args.future_policy == 'own'
        assert args.prefix_write_verification == 'action'
        assert args.jobs_file == '/data/gb/setup/best_policy_jobs_' + part + '_' + stage + '_20261004.json'
        for key, value in plan['source_options'].items():
            assert getattr(args, key) == value
        args.jobs_file = str(RUN / ('jobs_' + part + '_' + stage + '.json'))
        scope = {'args': args, 'json': json, 'Path': Path, 'dataset': dataset, 'eligible': eligible}
        exec(jobs_code, scope)
        actual = [{'sequence': dataset[index].get_name(), 'query_frame': frame} for index, frame in scope['jobs']]
        assert actual == expected and args.clips == len(expected)
        cases.append({'partition': part, 'stage': stage, 'clips_after_exact_jobs_override': args.clips,
                      'ordered_jobs_exact': True, 'dataset_eligibility': 'synthetic source interface fixture only'})
assert not partitions['train'] & partitions['validation']

gates = [shlex.split(line.strip())[-1] for line in runner.splitlines() if 'python -c ' in line]
assert len(gates) == 2
gate_inputs = [
    {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional'},
    {'status': 'PASS', 'both_partitions_passed': True, 'prefix_epoch': 4,
     'future_teacher_is_global_best': True, 'clips_per_partition': 16},
]
gate_results = []
for code, valid in zip(gates, gate_inputs):
    def execute(value):
        exec(code, {'open': lambda path: io.StringIO(json.dumps(value))})
    execute(valid)
    rejected = []
    for key in valid:
        invalid = copy.deepcopy(valid)
        invalid[key] = False
        try:
            execute(invalid)
        except AssertionError:
            rejected.append(key)
        else:
            raise AssertionError(('gate accepted invalid field', key))
    def missing(path):
        raise FileNotFoundError(path)
    try:
        exec(code, {'open': missing})
    except FileNotFoundError:
        missing_rejected = True
    else:
        raise AssertionError('gate accepted a missing receipt')
    gate_results.append({'valid_accepted': True, 'missing_rejected': missing_rejected, 'invalid_fields_rejected': rejected})
assert runner.index(gates[1]) < runner.index('export CUDA_DEVICE_ORDER') < runner.index('python -u -m')
assert 'if [[ "$stage" == rest ]]; then' in runner
assert 'test ! -e "$run"' in runner

modules = ast.parse((ROOT / 'research/recoverability_modules.py').read_text(encoding='utf-8'))
fields = ast.literal_eval(next(node.value for node in modules.body if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'DECISION_FIELDS' for t in node.targets)))
assert len(fields) == 19
assert not set(fields) & {'current_iou', 'future_iou', 'history_iou', 'wrong_update_fraction', 'action_valid', 'motion_targets'}
audit_assertions = [ast.unparse(node.test) for node in ast.walk(audit_tree) if isinstance(node, ast.Assert)]
assert any("cfg['prefix_checkpoint_epoch'] == cfg['future_checkpoint_epoch'] == 4" in node for node in audit_assertions)
assert any("cfg['jobs'] == jobs == previous['jobs'][:16]" in node for node in audit_assertions)
assert any("len(jobs) == 16" in node for node in audit_assertions)
local_yaml = (ROOT / 'config/_dataset/train-lasher.yaml').read_bytes()
remote_yaml = (RUN / 'remote_train_lasher.yaml').read_bytes()
assert local_yaml.replace(b'\r\n', b'\n') == remote_yaml
assert local_yaml.count(b'\r\n') == 23 and remote_yaml.count(b'\r') == 0
local_dataset = (ROOT / 'trackit/datasets/MMOT/specialization/memory_mapped/dataset.py').read_bytes()
remote_dataset = (RUN / 'remote_MMOT_dataset.py').read_bytes()
assert local_dataset.replace(b'\r\n', b'\n') == remote_dataset
assert local_dataset.count(b'\r\n') == 188 and remote_dataset.count(b'\r') == 0
assert ast.dump(ast.parse(local_dataset), include_attributes=False) == ast.dump(ast.parse(remote_dataset), include_attributes=False)
launcher = ROOT.parent / 'launch_best_policy_probe_20261004.py'
launcher_text = launcher.read_text(encoding='utf-8')
launcher_tree = ast.parse(launcher_text)
compile(launcher_tree, str(launcher), 'exec')
symbols = ast.literal_eval(next(node.value for node in launcher_tree.body if isinstance(node, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == 'symbols' for t in node.targets)))
assert symbols == ('InstanceMemory', 'RecoverabilityModules', 'select_actions', 'DECISION_FIELDS')
selector = next(node for node in launcher_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'node')
selector_scope = {'ast': ast}
exec(compile(ast.Module(body=[selector], type_ignores=[]), str(launcher), 'exec'), selector_scope)
assert all(selector_scope['node'](modules, name) is not None for name in symbols)
yaml_guard = next(node for node in launcher_tree.body if isinstance(node, ast.Assert)
                  and 'remote_train_lasher.yaml' in ast.unparse(node))
assert ast.unparse(yaml_guard.test) == "sha(main / 'config/_dataset/train-lasher.yaml') == review['source_sha256']['refine-logs/runs/recoverability_best_policy/remote_train_lasher.yaml']"
dataset_guard = next(node for node in launcher_tree.body if isinstance(node, ast.Assert)
                     and 'remote_MMOT_dataset.py' in ast.unparse(node))
assert ast.unparse(dataset_guard.test) == "sha(main / 'trackit/datasets/MMOT/specialization/memory_mapped/dataset.py') == review['source_sha256']['refine-logs/runs/recoverability_best_policy/remote_MMOT_dataset.py']"
assert launcher_text.index('for name in symbols:') < launcher_text.index("memory = subprocess.run(")
assert "(('train', 1), ('validation', 2))" in launcher_text
assert "torch" not in sys.modules and "numpy" not in sys.modules
result = {'status': 'PASS', 'scope': 'Source-only syntax, exact job interface and in-memory gate fixtures',
          'actual_probe_acceptance': False, 'python_compile': 'PASS', 'bash_syntax': 'PASS',
          'job_cases': cases, 'partition_sequence_sets_disjoint': True, 'guard_cases': gate_results,
          'both_actual_checkpoint_epochs_required_4': True, 'decision_fields_count': len(fields),
          'decision_label_fields_disjoint': True, 'neural_forward_calls': 0, 'optimizer_steps': 0,
          'deployment_preflight': {'YAML_CRLF_to_LF_bytes_exact': True,
              'local_YAML_bytes': len(local_yaml), 'remote_YAML_bytes': len(remote_yaml),
              'remote_YAML_sha256': hashlib.sha256(remote_yaml).hexdigest(),
              'MMOT_dataset_CRLF_to_LF_bytes_exact': True, 'MMOT_dataset_full_AST_equal': True,
              'local_MMOT_dataset_bytes': len(local_dataset), 'remote_MMOT_dataset_bytes': len(remote_dataset),
              'remote_MMOT_dataset_sha256': hashlib.sha256(remote_dataset).hexdigest(),
              'launcher_compile': 'PASS', 'launcher_sha256': hashlib.sha256(launcher.read_bytes()).hexdigest(),
              'launcher_binds_remote_YAML_exact_bytes': True,
              'launcher_binds_remote_MMOT_dataset_exact_bytes': True,
              'inference_symbols_exist_in_local_reviewed_source': list(symbols),
              'source_guards_precede_GPU_query_and_launch': True,
              'launcher_executed': False, 'remote_inference_AST_independently_replayed_by_reviewer': False},
          'torch_imported': False, 'numpy_imported': False, 'dataset_read': False,
          'ssh_used': False, 'GPU_queries': 0, 'runtime_certification': False, 'python_version': sys.version}
(RUN / 'source_checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2, allow_nan=False))
