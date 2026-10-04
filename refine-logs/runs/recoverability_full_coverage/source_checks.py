"""Source/stdlib fixtures only: no project imports, data, checkpoints or launch."""
import ast
import copy
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FILES = (
    'refine-logs/runs/recoverability_full_coverage/prepare_jobs_cpu.py',
    'refine-logs/runs/recoverability_full_coverage/launch_collection.py',
    'refine-logs/runs/recoverability_full_coverage/plan.json',
    'scripts/run_full_coverage_collect.sh',
)
read = lambda name: (ROOT / name).read_text(encoding='utf-8')
load = lambda name: json.loads(read(name))
sha = lambda name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
prepare = ast.parse(read(FILES[0]))
launch = ast.parse(read(FILES[1]))
collector_name = 'research/collect_recoverability.py'
collector = ast.parse(read(collector_name))
main = next(node for node in collector.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
for name in FILES[:2]:
    compile(read(name), name, 'exec')
plan = load(FILES[2])


def assigned(tree, name):
    return next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def evaluate(expression, values):
    return eval(compile(ast.Expression(copy.deepcopy(expression)), '<source fixture>', 'eval'), values)


main_eligibility = next(node for node in main.body if isinstance(node, ast.For)
                        and ast.unparse(node.target) == 'index')
assert ast.dump(assigned(prepare, 'valid')) == ast.dump(assigned(main_eligibility, 'valid'))
prepare_queries = assigned(prepare, 'eligible')
collector_queries = assigned(main_eligibility, 'queries')


class Valid(list):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Valid(value) if isinstance(key, slice) else value

    def all(self):
        return all(self)


cases = [(length, []) for length in (4, 5, 6, 10, 1027, 1028, 1500)]
cases += [(10, [bad]) for bad in (0, 1, 2, 3, 4, 8, 9)]
cases += [(10, list(range(1, 10))), (1500, [512, 1026, 1027])]
fixture_rows = []
guard = next(node.test for node in ast.walk(prepare) if isinstance(node, ast.Assert)
             and ast.unparse(node.test) == 'valid[0] and eligible')
for length, invalid in cases:
    valid = Valid(index not in invalid for index in range(length))
    values = {'boxes': range(length), 'valid': valid, 'args': SimpleNamespace(max_prefix=1024)}
    actual = evaluate(prepare_queries, values)
    original = evaluate(collector_queries, values)
    expected = [frame for frame in range(1, length) if frame <= 1024 and frame + 3 < length
                and all(valid[frame + offset] for offset in range(4))]
    assert actual == original == expected
    accepted = bool(evaluate(guard, values | {'eligible': actual}))
    assert accepted == (0 not in invalid and bool(expected))
    chosen = actual[len(actual) // 2] if accepted else None
    assert chosen is None or 1 <= chosen <= 1024
    fixture_rows.append({'frames': length, 'invalid': invalid, 'eligible_queries': len(actual),
                         'accepted': accepted, 'selected_query': chosen})
assert any(row['frames'] == 1028 and row['eligible_queries'] == 1024 for row in fixture_rows)
assert sum(not row['accepted'] for row in fixture_rows) == 3

split_assert = next(node.test for node in prepare.body if isinstance(node, ast.Assert)
                    and "len(split['train'])" in ast.unparse(node.test))
fixture_split = {'train': ['train_' + str(i) for i in range(881)],
                 'validation': ['val_' + str(i) for i in range(98)]}
assert evaluate(split_assert, {'split': fixture_split})
leaked = copy.deepcopy(fixture_split)
leaked['validation'][0] = leaked['train'][0]
assert not evaluate(split_assert, {'split': leaked})
missing = copy.deepcopy(fixture_split)
missing['train'].pop()
assert not evaluate(split_assert, {'split': missing})
partition_loop = next(node for node in prepare.body if isinstance(node, ast.For))
shard_counts = ast.literal_eval(partition_loop.iter)
assert shard_counts == (('train', (221, 220, 220, 220)), ('validation', (25, 25, 24, 24)))
shard_rows = []
for part, counts in shard_counts:
    names = sorted(fixture_split[part])
    offset, joined = 0, []
    for gpu, count in enumerate(counts):
        selected = names[offset:offset + count]
        assert len(selected) == count
        joined.extend(selected)
        offset += count
        shard_rows.append({'partition': part, 'gpu': gpu, 'clips': count})
    assert joined == names and len(set(joined)) == len(names) == offset

full = load('refine-logs/runs/recoverability_best_policy_fit/actual_full_cpu_acceptance.json')
m0 = load('refine-logs/runs/recoverability_best_policy_fit/actual_m0_cpu_acceptance.json')
assert full['status'] == m0['status'] == 'PASS'
assert full['all_four_lr_ranking_arms_passed'] and m0['all_four_lr_ranking_arms_passed']
assert full['batch_size'] == m0['batch_size'] == 416
assert full['epochs_per_arm'] == 60 and full['optimizer_steps_per_arm'] == 180
assert m0['epochs_per_arm'] == 3 and m0['optimizer_steps_per_arm'] == 9
arms = ast.literal_eval(assigned(prepare, 'arms'))
teacher_arm = evaluate(assigned(prepare, 'chosen'), {'arms': arms, 'accepted': full})
teacher = full['arms'][teacher_arm]
assert teacher_arm == 'pairwise_lr4' and teacher['best_epoch'] == 46
assert teacher['strict25_tensor_CPU_load'] and teacher['C1_all8_retained_tensors_exact']
assert teacher['active_modules'] == ['A', 'B', 'C']
assert teacher['artifact_sha256']['best.pth'] == 'a840219e1d362ac353ea13fae853a11d93505799f5bf7d6fbc2112e7b578a529'
assert sha(collector_name) == full['source_sha256'][
    '/data/gb/experiments/recoverability_best_policy_fit_20261004/' + collector_name]
cfg = load('refine-logs/runs/recoverability_best_policy_fit/full_pairwise_lr4/config.json')
reference = cfg['source_configs'][0]
defaults = {}
for node in ast.walk(main):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
        defaults[node.args[0].value] = {keyword.arg: keyword.value for keyword in node.keywords}
for key in ('root', 'cache', 'split', 'pretrained', 'head', 'motion_run'):
    assert reference[key] == ast.literal_eval(defaults['--' + key.replace('_', '-')]['default'])
assert [len({job[0] for job in cfg[part + '_jobs']}) for part in ('train', 'validation')] == [549, 76]
assert not {job[0] for job in cfg['train_jobs']} & {job[0] for job in cfg['validation_jobs']}

runner = read(FILES[3])
command = runner[runner.index('bash scripts/run_temporal.sh'):runner.index('\n  date -Iseconds')]
argv = shlex.split(command.replace('\\\n', ' '))
assert argv[:4] == ['bash', 'scripts/run_temporal.sh', '$gpu', 'collect_recoverability']
flags = dict(zip(argv[4::2], argv[5::2]))
expected_flags = {'--root': reference['root'], '--partition': '$partition', '--clips': '16',
                  '--batch-clips': '16', '--forward-batch': '64', '--max-prefix': '1024', '--seed': '42',
                  '--prefix-model': '$parent', '--future-policy': 'own', '--prefix-write-verification': 'action',
                  '--jobs-file': '/data/gb/setup/full_coverage_jobs_${partition}_gpu${gpu}_20261004.json',
                  '--output': '$run'}
assert flags == expected_flags
assert all(flag in defaults for flag in flags)
assert 'for partition in train validation; do' in runner
assert runner.count('bash scripts/run_temporal.sh') == 1
assert 'test ! -e "$run"' in runner and 'set -euo pipefail' in runner
assert 'export CUDA_VISIBLE_DEVICES="$gpu"' in read('scripts/run_temporal.sh')
assert 'exec /data/gb/envs/gola/bin/python -u -m "research.$entry" "$@"' in read('scripts/run_temporal.sh')
assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
           and node.func.attr == 'Popen' for node in ast.walk(launch)) == 1
launch_loop = next(node for node in launch.body if isinstance(node, ast.For)
                   and ast.unparse(node.target) == 'gpu')
assert ast.unparse(launch_loop.iter) == 'range(4)'
launch_text = read(FILES[1])
assert "assert not out.exists()" in launch_text
assert "assert not Path('/data/gb/outputs/recoverability_full_coverage_collect_'" in launch_text
assert "assert jobs == prepared['jobs'][part]" in launch_text
assert "'status': 'LAUNCHED_NOT_COMPLETED'" in launch_text
assert "'training_started': False, 'native_started': False" in launch_text
assert "sha(prepared['teacher']) == prepared['teacher_sha256']" in launch_text

guard_cases = 0
for name, positive, negative in (
    ('review', {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional'},
     {'status': 'FAIL', 'review_independence': 'self', 'acceptance_status': 'final'}),
    ('prepared', {'status': 'PASS', 'eligible_train_videos': 881, 'eligible_validation_videos': 98,
                  'all_train_video_names_exact': True, 'all_validation_video_names_exact': True},
     {'status': 'FAIL', 'eligible_train_videos': 880, 'eligible_validation_videos': 97,
      'all_train_video_names_exact': False, 'all_validation_video_names_exact': False}),
):
    guards = [node.test for node in launch.body if isinstance(node, ast.Assert)
              and ast.unparse(node.test).startswith(name + '[')]
    assert guards and all(evaluate(test, {name: positive}) for test in guards)
    for key, value in negative.items():
        altered = positive | {key: value}
        assert not all(evaluate(test, {name: altered}) for test in guards)
        guard_cases += 1

bash = Path('E:/dcda6-main/Git/bin/bash.exe')
for shell_source in (FILES[3], 'scripts/run_temporal.sh'):
    syntax = subprocess.run([str(bash), '-n', shell_source], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert not syntax.stdout and not syntax.stderr
assert plan['status'] == 'PREPARED_NOT_LAUNCHED' and plan['expected_clips'] == {'train': 881, 'validation': 98}
assert plan['training']['all_ABC_active'] and plan['training']['C1_frozen']
assert plan['training']['epochs'] * math.ceil(881 / plan['training']['batch']) == plan['training']['optimizer_steps'] == 180
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
record = {
    'status': 'PASS', 'scope': 'Source, JSON metadata and stdlib synthetic fixtures only.',
    'source_sha256': {name: sha(name) for name in FILES},
    'python_sources_compiled': 2, 'bash_syntax': 'PASS',
    'eligibility_predicate_AST_matches_collector': True, 'eligibility_fixtures': fixture_rows,
    'invalid_anchor_or_no_query_rejections': 3, 'partition_positive_and_negative_cases': 3,
    'ordered_shards': shard_rows, 'teacher_arm': teacher_arm, 'teacher_epoch': teacher['best_epoch'],
    'teacher_schema_from_existing_actual_receipt': 'strict25 tensors, C1 all8 retained exact; no new checkpoint load',
    'existing_M0_and_full_all_four_arms_PASS': True,
    'unchanged_collector_matches_existing_actual_receipt': True,
    'reference_paths_equal_collector_defaults': True, 'previous_unique_train_val_videos': [549, 76],
    'collector_flags': flags, 'literal_gate_negative_fixtures_rejected': guard_cases,
    'launch_process_count': 4, 'per_gpu_partition_order': ['train', 'validation'],
    'command_evidence': [
        sys.executable + ' -I -B refine-logs/runs/recoverability_full_coverage/source_checks.py',
        str(bash) + ' -n scripts/run_full_coverage_collect.sh',
        str(bash) + ' -n scripts/run_temporal.sh'],
    'python_version': sys.version, 'torch_imported': False, 'numpy_imported': False,
    'SSH_calls': 0, 'GPU_calls': 0, 'neural_forward_calls': 0, 'actual_data_or_checkpoint_reads': 0,
    'preparer_or_launcher_executed': False, 'runtime_attested': False,
    'limitation': 'PASS does not establish all979 actual videos eligible, completed collection, actual-GT replay, training or native metrics.',
}
(OUT / 'source_checks.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': record['status'], 'eligibility_fixtures': len(fixture_rows),
                  'gate_negative_fixtures': guard_cases, 'runtime_attested': False}))
