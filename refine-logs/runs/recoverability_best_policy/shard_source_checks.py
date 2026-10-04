"""Review sharding with source/stdlib fixtures only; never run the collector."""
import argparse
import ast
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
COLLECTOR = ROOT / 'research/collect_recoverability.py'
RUNNER = ROOT / 'scripts/run_recoverability_best_policy_shard.sh'
PRIOR_RUNNER = ROOT / 'scripts/run_recoverability_best_policy_collect.sh'
BASH = sys.argv[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


review = read_json(RUN / 'source_review.json')
plan = read_json(RUN / 'plan.json')
shard_plan = read_json(RUN / 'shard_plan.json')
assert (review['status'], review['review_independence'], review['acceptance_status']) == ('PASS', 'same-family', 'provisional')
unchanged = [
    'scripts/run_recoverability_best_policy_collect.sh',
    'research/collect_recoverability.py', 'research/collect_rollouts.py',
    'research/candidate_learning.py', 'research/recoverability_tracker.py',
    'research/recoverability_modules.py', 'research/bounded_recovery.py',
    'config/_dataset/train-lasher.yaml',
    'trackit/datasets/MMOT/specialization/memory_mapped/dataset.py',
    *['refine-logs/runs/recoverability_best_policy/' + name for name in (
        'actual_probe_cpu_audit.py', 'actual_policy_alignment_metadata.json', 'plan.json',
        'jobs_train_probe.json', 'jobs_train_rest.json',
        'jobs_validation_probe.json', 'jobs_validation_rest.json')],
]
for relative in unchanged:
    assert sha(ROOT / relative) == review['source_sha256'][relative], relative

runner = RUNNER.read_text(encoding='utf-8')
prior_runner = PRIOR_RUNNER.read_text(encoding='utf-8')
collector_tree = ast.parse(COLLECTOR.read_text(encoding='utf-8'))
compile(collector_tree, str(COLLECTOR), 'exec')
subprocess.run([BASH, '-n', str(RUNNER)], check=True, capture_output=True, text=True)
main = next(node for node in collector_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
parser_nodes = []
for node in main.body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'args' for target in node.targets):
        break
    parser_nodes.append(node)
scope = {'argparse': argparse, '__doc__': ''}
exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), str(COLLECTOR), 'exec'), scope)
parser = scope['p']
jobs_block = next(node for node in main.body if isinstance(node, ast.If) and ast.unparse(node.test) == 'args.jobs_file')
jobs_code = compile(ast.Module(body=[jobs_block], type_ignores=[]), str(COLLECTOR), 'exec')


def cli_tokens(source, replacements):
    command = source[source.index('/data/gb/envs/gola/bin/python -u -m'):source.index('\ndate -Iseconds')]
    for key, value in replacements.items():
        command = command.replace('${' + key + '}', value).replace('$' + key, value)
    tokens = shlex.split(command.replace('\\\n', ' '))
    return tokens[tokens.index('--root'):]


counts = {'train0': 296, 'train1': 295, 'train2': 295, 'validation': 112}
gpu_assignments = {'train0': 0, 'train1': 1, 'train2': 3, 'validation': 2}
assert shard_plan['counts'] == counts and shard_plan['gpu_assignments'] == gpu_assignments
assert shard_plan['no_query_overlap'] and not shard_plan['temperature_or_power_queries']
full = {part: read_json(ROOT / 'refine-logs/runs/recoverability_write_pair/future_policy_jobs' / (part + '.json'))
        for part in ('train', 'validation')}
probe = {part: read_json(RUN / ('jobs_' + part + '_probe.json'))['jobs'] for part in full}
rest = {part: read_json(RUN / ('jobs_' + part + '_rest.json'))['jobs'] for part in full}
shards = {shard: read_json(RUN / ('jobs_' + shard + '_rest.json'))['jobs'] for shard in counts}
assert len(full['train']) == 902 and len(full['validation']) == 128
assert len(rest['train']) == 886 and len(rest['validation']) == 112
assert shards['train0'] == rest['train'][:296]
assert shards['train1'] == rest['train'][296:591]
assert shards['train2'] == rest['train'][591:]
assert shards['validation'] == rest['validation']
assert probe['train'] + shards['train0'] + shards['train1'] + shards['train2'] == full['train']
assert probe['validation'] + shards['validation'] == full['validation']
for part in full:
    assert len(probe[part]) == 16 and probe[part] + rest[part] == full[part]
    assert len({(job['sequence'], job['query_frame']) for job in full[part]}) == len(full[part])
assert not {job['sequence'] for job in full['train']} & {job['sequence'] for job in full['validation']}
all_groups = list(probe.values()) + list(shards.values())
all_pairs = [{(job['sequence'], job['query_frame']) for job in group} for group in all_groups]
assert all(not left & right for index, left in enumerate(all_pairs) for right in all_pairs[index + 1:])

cases = []
output_roots, jobs_paths = set(), set()
for shard, expected_count in counts.items():
    part = 'validation' if shard == 'validation' else 'train'
    output = '/data/gb/outputs/recoverability_best_policy_collect_' + shard + '_rest_20261004'
    args = parser.parse_args(cli_tokens(runner, {'shard': shard, 'partition': part, 'run': output}))
    old_output = '/data/gb/outputs/recoverability_best_policy_collect_' + part + '_rest_20261004'
    old_args = parser.parse_args(cli_tokens(prior_runner, {'partition': part, 'stage': 'rest', 'run': old_output}))
    assert {key: value for key, value in vars(args).items() if key not in ('jobs_file', 'output')} == {
        key: value for key, value in vars(old_args).items() if key not in ('jobs_file', 'output')}
    assert args.clips == args.batch_clips == 16 and args.forward_batch == 64 and args.seed == 42
    assert args.prefix_model == plan['prefix_model'] and args.future_policy == 'own'
    assert args.prefix_write_verification == 'action' and args.max_prefix == 1024
    for key, value in plan['source_options'].items():
        assert getattr(args, key) == value
    assert args.output == output
    assert args.jobs_file == '/data/gb/setup/best_policy_jobs_' + shard + '_rest_20261004.json'
    output_roots.add(args.output)
    jobs_paths.add(args.jobs_file)
    names = list(dict.fromkeys(job['sequence'] for job in full[part]))
    dataset = [SimpleNamespace(get_name=lambda name=name: name) for name in names]
    eligible = [(index, [job['query_frame'] for job in full[part] if job['sequence'] == name]) for index, name in enumerate(names)]
    args.jobs_file = str(RUN / ('jobs_' + shard + '_rest.json'))
    scope = {'args': args, 'json': json, 'Path': Path, 'dataset': dataset, 'eligible': eligible}
    exec(jobs_code, scope)
    actual = [{'sequence': dataset[index].get_name(), 'query_frame': frame} for index, frame in scope['jobs']]
    assert actual == shards[shard] and args.clips == len(actual) == expected_count
    cases.append({'shard': shard, 'GPU_argument': gpu_assignments[shard], 'partition': part,
                  'clips_before_jobs_override': 16, 'clips_after_exact_jobs_override': args.clips,
                  'ordered_jobs_exact': True, 'output': args.output,
                  'eligibility_scope': 'Synthetic source-interface fixture; no actual dataset read'})
assert len(output_roots) == len(jobs_paths) == 4
assert not output_roots & {'/data/gb/outputs/recoverability_best_policy_collect_' + part + '_probe_20261004' for part in full}

gates = [shlex.split(line.strip())[-1] for line in runner.splitlines() if 'python -c ' in line]
prior_gates = [shlex.split(line.strip())[-1] for line in prior_runner.splitlines() if 'python -c ' in line]
assert len(gates) == 2 and gates[1] == prior_gates[1]
assert gates[0] == prior_gates[0].replace('best_policy_collect_source_review_', 'best_policy_shard_source_review_')
valid_receipts = [
    {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional'},
    {'status': 'PASS', 'both_partitions_passed': True, 'prefix_epoch': 4,
     'future_teacher_is_global_best': True, 'clips_per_partition': 16},
]
gate_results = []
for code, valid in zip(gates, valid_receipts):
    exec(code, {'open': lambda path: io.StringIO(json.dumps(valid))})
    rejected = []
    for key in valid:
        invalid = valid | {key: False}
        try:
            exec(code, {'open': lambda path: io.StringIO(json.dumps(invalid))})
        except AssertionError:
            rejected.append(key)
        else:
            raise AssertionError(('gate accepted invalid field', key))
    def missing(path):
        raise FileNotFoundError(path)
    try:
        exec(code, {'open': missing})
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('gate accepted missing receipt')
    gate_results.append({'valid_accepted': True, 'invalid_fields_rejected': rejected, 'missing_receipt_rejected': True})
assert runner.index(gates[0]) < runner.index(gates[1]) < runner.index('export CUDA_DEVICE_ORDER') < runner.index('python -u -m')
assert runner.splitlines()[1] == 'set -euo pipefail'
assert runner.rstrip().endswith('date -Iseconds > "$run/job_completed.txt"')
assert 'test ! -e "$run"' in runner
assert [line for line in runner.splitlines() if line.startswith('export ')] == [line for line in prior_runner.splitlines() if line.startswith('export ')]

# Only the shell control flow runs. Commands that would read/write real paths or
# execute Python are replaced with in-memory fixtures; no actual receipts exist.
fixture_prefix = r'''
gate_count=0
cd() { [[ "$1" == /data/gb/GOLA ]]; }
test() { [[ "$1" == '!' && "$2" == -e && "$3" == /data/gb/outputs/recoverability_best_policy_collect_* ]]; [[ "$FIXTURE_EXISTS" == 0 ]]; }
fixture_python() {
  if [[ "$1" == -c ]]; then
    gate_count=$((gate_count + 1))
    printf 'GATE %s\n' "$gate_count"
    [[ "$gate_count" != "$FIXTURE_FAIL_GATE" ]]
  else
    printf 'COLLECTOR %s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'ARG %s\n' "$@"
    return "$FIXTURE_COLLECTOR_EXIT"
  fi
}
'''
fixture = fixture_prefix + runner.replace('/data/gb/envs/gola/bin/python', 'fixture_python').replace(
    'date -Iseconds > "$run/job_completed.txt"', 'printf \'MARKER %s\\n\' "$run/job_completed.txt"')
shell_cases = []
for shard, gpu in gpu_assignments.items():
    settings = 'FIXTURE_EXISTS=0\nFIXTURE_FAIL_GATE=0\nFIXTURE_COLLECTOR_EXIT=0\n'
    result = subprocess.run([BASH, '-s', '--', str(gpu), shard], input=settings + fixture, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('COLLECTOR ') == result.stdout.count('MARKER ') == 1
    assert 'COLLECTOR ' + str(gpu) + '\n' in result.stdout
    assert 'ARG /data/gb/setup/best_policy_jobs_' + shard + '_rest_20261004.json\n' in result.stdout
    assert 'MARKER /data/gb/outputs/recoverability_best_policy_collect_' + shard + '_rest_20261004/job_completed.txt\n' in result.stdout
    shell_cases.append({'case': shard, 'returncode': 0, 'collector_calls': 1, 'marker_calls': 1})
for label, shard, fail_gate, exists, collector_exit, expected_calls in (
    ('source_gate_failure', 'train0', 1, 0, 0, 0),
    ('actual_acceptance_gate_failure', 'train0', 2, 0, 0, 0),
    ('existing_output', 'train0', 0, 1, 0, 0),
    ('invalid_shard', 'train9', 0, 0, 0, 0),
    ('collector_nonzero', 'train0', 0, 0, 7, 1),
):
    settings = f'FIXTURE_EXISTS={exists}\nFIXTURE_FAIL_GATE={fail_gate}\nFIXTURE_COLLECTOR_EXIT={collector_exit}\n'
    result = subprocess.run([BASH, '-s', '--', '0', shard], input=settings + fixture, capture_output=True, text=True)
    assert result.returncode != 0 and 'MARKER ' not in result.stdout
    assert result.stdout.count('COLLECTOR ') == expected_calls
    shell_cases.append({'case': label, 'returncode': result.returncode, 'collector_calls': expected_calls, 'marker_calls': 0})

sources = unchanged + [
    'scripts/run_recoverability_best_policy_shard.sh',
    'refine-logs/runs/recoverability_write_pair/future_policy_jobs/train.json',
    'refine-logs/runs/recoverability_write_pair/future_policy_jobs/validation.json',
    *['refine-logs/runs/recoverability_best_policy/' + name for name in (
        'source_review.json', 'shard_plan.json', 'jobs_train0_rest.json',
        'jobs_train1_rest.json', 'jobs_train2_rest.json', 'shard_source_checks.py')],
]
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
record = {
    'status': 'PASS', 'scope': 'Source-only exact AST interfaces, manifests and mocked Bash control flow',
    'actual_probe_acceptance': False, 'runtime_independently_attested': False,
    'source_sha256': {relative: sha(ROOT / relative) for relative in sources},
    'prior_review_bound_sources_unchanged': unchanged,
    'python_compile': 'PASS', 'bash_syntax': 'PASS', 'job_cases': cases,
    'ordered_full_train_902_recomposition_exact': True,
    'ordered_full_validation_128_recomposition_exact': True,
    'probe_and_rest_pairs_all_unique_disjoint': True,
    'train_validation_sequence_sets_disjoint': True,
    'distinct_jobs_paths_and_output_roots': True,
    'guard_cases': gate_results, 'shell_cases': shell_cases,
    'completed_marker_requires_collector_exit_zero': True,
    'execution': {'torch_imported': False, 'numpy_imported': False, 'dataset_reads': 0,
                  'checkpoint_reads': 0, 'native_test_reads': 0, 'actual_probe_outputs_read': False,
                  'neural_forward_calls': 0, 'optimizer_steps': 0, 'new_weights': 0,
                  'SSH_calls': 0, 'GPU_queries': 0, 'production_edits': False},
    'python_version': sys.version,
}
(RUN / 'shard_source_checks.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'status': record['status'], 'job_counts': counts, 'shell_cases': len(shell_cases),
                  'actual_probe_acceptance': False, 'receipt': str(RUN / 'shard_source_checks.json')}))
