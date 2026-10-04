"""Stdlib source fixtures only; no real models, arrays, datasets, GPU or SSH."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASH = r'E:\dcda6-main\Git\bin\bash.exe'
RUNNER = ROOT / 'scripts/run_best_policy_native_complete.sh'
REPORT = ROOT / 'scripts/report_best_policy_native_complete.sh'
LAUNCH = HERE / 'launch_native_complete.py'
MERGE = HERE / 'merge_native_complete.py'
AUDIT = HERE / 'audit_native_complete_cpu.py'
ORIGINAL_AUDIT = ROOT.parent / 'write_pair_native_rgbt234_cpu_audit_original_20261004.py'
NEW = [RUNNER, REPORT, LAUNCH, MERGE, AUDIT, HERE / 'native_complete_plan.json']
UPSTREAM = [ROOT / p for p in (
    'research/evaluate_recoverability.py', 'scripts/run_temporal.sh',
    'research/collect_core_metrics.py', 'research/collect_recoverability_metrics.py',
    'research/paired_sequence_bootstrap.py', 'research/plot_core_metrics.py', 'evaluation.py')]
FILES = NEW + UPSTREAM + [HERE / 'actual_best_policy_fit_cpu_audit.py']
COUNTS = {}


def check(name):
    COUNTS[name] = COUNTS.get(name, 0) + 1


def execute(nodes, namespace, filename):
    nodes = [node for node in nodes if not isinstance(node, (ast.Import, ast.ImportFrom))]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]), str(filename), 'exec'), namespace)


def rejects(action, category):
    try:
        action()
    except (AssertionError, KeyError, FileNotFoundError):
        check(category)
        return
    raise AssertionError('Invalid fixture accepted: ' + category)


for path in FILES:
    if path.suffix == '.py':
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        check('python_compile')
for path in (RUNNER, REPORT, ROOT / 'scripts/run_temporal.sh'):
    subprocess.run([BASH, '-n', str(path)], check=True, capture_output=True, text=True)
    check('bash_syntax')
assert hashlib.sha256(UPSTREAM[0].read_bytes()).hexdigest() == '08a1d2c548cf68410a769cf7eaf72f5727b16df6a409f14d55c8b7e91a75b7a8'
plan = json.loads((HERE / 'native_complete_plan.json').read_text(encoding='utf-8'))
for dataset, count, frames in [('lasher', 245, 220703), ('rgbt234', 234, 116649)]:
    row = plan['datasets'][dataset]
    indices = [i for offset, size in zip(row['offsets'], row['counts']) for i in range(offset, offset + size)]
    assert indices == list(range(count)) and row['frames'] == frames
    check('exact_disjoint_full_shard_partition')

# Run exact Bash arrays and loop command fragments with a no-op command sink.
runner = RUNNER.read_text(encoding='utf-8')
route = runner[runner.index('lasher_offsets='):]
route = '\n'.join(line for line in route.splitlines() if not line.strip().startswith('date -Iseconds'))
prefix = 'set -euo pipefail\ngpu="$1"\nmodel=/fixture/fixed.pth\nbash() { printf "CALL\\n"; printf "%s\\n" "$@"; }\ntest() { :; }\n'
for gpu in range(4):
    result = subprocess.run([BASH, '-c', prefix + route, 'fixture', str(gpu)], check=True, capture_output=True, text=True)
    calls = [part.strip().splitlines() for part in result.stdout.split('CALL\n')[1:]]
    assert len(calls) == 2
    for dataset, argv in zip(('lasher', 'rgbt234'), calls):
        assert argv[:3] == ['/data/gb/GOLA/scripts/run_temporal.sh', str(gpu), 'evaluate_recoverability']
        flags = dict(zip(argv[3::2], argv[4::2]))
        row = plan['datasets'][dataset]
        assert flags == {'--dataset': dataset, '--root': row['root'], '--model': '/fixture/fixed.pth',
                         '--write-verification': 'action', '--seed': '42', '--sequence-offset': str(row['offsets'][gpu]),
                         '--limit-sequences': str(row['counts'][gpu]), '--max-frames': '0',
                         '--output': f'/data/gb/outputs/recoverability_best_policy_native_full_20261004/{dataset}/shards/gpu{gpu}/predictions'}
        check('exact_native_cli_routing')


class MemoryPath:
    """Only in-memory mock filenames; never delegates to the host filesystem."""
    fs = {}

    def __init__(self, path):
        self.path = PurePosixPath(str(path))

    def __truediv__(self, other):
        return MemoryPath(self.path / other)

    def __str__(self):
        return str(self.path)

    @property
    def name(self):
        return self.path.name

    @property
    def stem(self):
        return self.path.stem

    def exists(self):
        return str(self) in self.fs

    def is_file(self):
        return self.exists()

    def read_text(self):
        return self.fs[str(self)]

    def read_bytes(self):
        return self.read_text().encode()

    def write_text(self, value):
        self.fs[str(self)] = value

    def open(self, mode):
        assert mode == 'wb'
        return io.BytesIO()

    def glob(self, pattern):
        assert pattern == '*.txt'
        return [MemoryPath(key) for key in self.fs if PurePosixPath(key).parent == self.path and key.endswith('.txt')]


setup = '/data/gb/setup/'
review_key = setup + 'best_policy_native_source_review_20261004.json'
audit_key = setup + 'best_policy_fit_actual_full_fit_cpu_audit_20261004.json'
selection_key = setup + 'best_policy_native_selection_20261004.json'
runner_key = setup + 'run_best_policy_native_complete_20261004.sh'
arm_names = ('pairwise_lr4', 'pairwise_lr5', 'budgeted_lr4', 'budgeted_lr5')
launch_tree = ast.parse(LAUNCH.read_text(encoding='utf-8'))
review = {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional',
          'source_sha256': {'scripts/run_best_policy_native_complete.sh': hashlib.sha256(RUNNER.read_bytes()).hexdigest()}}
audit = {'status': 'PASS', 'all_four_lr_ranking_arms_passed': True,
         'epochs_per_arm': 60, 'optimizer_steps_per_arm': 180, 'batch_size': 416, 'arms': {}}
for arm in arm_names:
    audit['arms'][arm] = {'status': 'PASS', 'root': '/fixture/' + arm,
                         'best_validation': {'utility': .4}, 'best_epoch': 0,
                         'best0_is_parent_not_new_learning': True,
                         'artifact_sha256': {'best.pth': hashlib.sha256(('mock bytes ' + arm).encode()).hexdigest()}}


def launch_case(audit_change=None, review_change=None, busy=False, reset=True):
    observed, spawned = [], []
    if reset:
        a, r = copy.deepcopy(audit), copy.deepcopy(review)
        if audit_change:
            audit_change(a)
        if review_change:
            review_change(r)
        MemoryPath.fs = {audit_key: json.dumps(a), review_key: json.dumps(r), runner_key: RUNNER.read_bytes().decode()}
        MemoryPath.fs.update({'/fixture/' + arm + '/best.pth': 'mock bytes ' + arm for arm in arm_names})

    def run(argv, **kwargs):
        assert argv == ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']
        observed.append(argv)
        return SimpleNamespace(stdout='\n'.join(f'{i}, {501 if busy else 0}' for i in range(4)))

    def popen(argv, **kwargs):
        assert argv == ['bash', runner_key, str(len(spawned))]
        assert kwargs['cwd'] == '/data/gb/GOLA' and kwargs['start_new_session']
        spawned.append(argv)
        return SimpleNamespace(pid=1000 + len(spawned))

    namespace = {'Path': MemoryPath, 'json': json, 'hashlib': hashlib, 'datetime': datetime,
                 'timedelta': timedelta, 'timezone': timezone,
                 'subprocess': SimpleNamespace(run=run, Popen=popen, STDOUT=-2)}
    execute(launch_tree.body, namespace, LAUNCH)
    assert len(spawned) == 4 and len(observed) == 1
    return json.loads(MemoryPath.fs[selection_key])


for winner in arm_names:
    value = launch_case(lambda a: a['arms'][winner]['best_validation'].update(utility=.5))
    assert value['selected_arm'] == winner and value['best0_is_parent_not_new_learning']
    assert value['both_datasets_same_fixed_checkpoint'] and value['new_complete_native_inference_requested_by_user']
    check('once_only_four_arm_selection_and_four_queue_launch_mock')
assert launch_case()['selected_arm'] == arm_names[0]
check('deterministic_first_arm_tie_selection')
rejects(lambda: launch_case(reset=False), 'duplicate_launch_rejected')
for change in (
    lambda a: a.update(status='PENDING'), lambda a: a.update(all_four_lr_ranking_arms_passed=False),
    lambda a: a.update(epochs_per_arm=59), lambda a: a.update(optimizer_steps_per_arm=179),
    lambda a: a.update(batch_size=384), lambda a: a['arms'].pop('budgeted_lr5'),
    lambda a: a['arms']['budgeted_lr5'].update(status='PENDING'),
    lambda a: a['arms']['pairwise_lr4']['artifact_sha256'].update({'best.pth': 'wrong'})):
    rejects(lambda change=change: launch_case(audit_change=change), 'full_fit_selection_gate_rejection')
rejects(lambda: launch_case(review_change=lambda r: r.update(status='PENDING')), 'source_review_gate_rejection')
rejects(lambda: launch_case(busy=True), 'nonfree_gpu_mock_rejection')

# Execute literal Python guards carried by both reviewed shell scripts.
for path in (RUNNER, REPORT):
    for line in path.read_text(encoding='utf-8').splitlines():
        if "python -c '" not in line:
            continue
        literal = line.split("python -c '", 1)[1].split("'", 1)[0]
        guard_tree = ast.parse(literal)
        good = {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional',
                'all_four_full_fits_CPU_passed': True, 'completed_epochs': 60, 'optimizer_steps': 180,
                'checkpoint': '/fixture/fixed.pth', 'all_four_actual_shards_completed': True, 'all_sequences_full_frames': True}
        def guard(value):
            execute(guard_tree.body, {'json': json, 'sys': SimpleNamespace(argv=['-c', '/fixture/receipt']),
                    'open': lambda *args: io.StringIO(json.dumps(value))}, path)
        guard(good)
        check('literal_shell_gate_positive')
        rejects(lambda: guard(dict(good, status='PENDING')), 'literal_shell_gate_rejection')
        for key in ('all_four_full_fits_CPU_passed', 'all_four_actual_shards_completed', 'all_sequences_full_frames'):
            if "d[\"" + key + "\"]" in literal:
                rejects(lambda key=key: guard(dict(good, **{key: False})), 'literal_shell_gate_rejection')

merge_tree = ast.parse(MERGE.read_text(encoding='utf-8'))
outer = next(node for node in merge_tree.body if isinstance(node, ast.For) and ast.unparse(node.target).startswith('(gpu,'))
inner = next(node for node in outer.body if isinstance(node, ast.For))
metadata_loop = copy.deepcopy(outer)
metadata_loop.body = [node for node in metadata_loop.body if not isinstance(node, ast.For)]
common_node = next(node for node in merge_tree.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'common')
merge_asserts = []
for node in merge_tree.body[merge_tree.body.index(outer) + 1:]:
    if not isinstance(node, ast.Assert):
        break
    merge_asserts.append(node)
old_cfg = json.loads((ROOT / 'refine-logs/runs/recoverability_write_pair/own4_native_rgbt234/predictions/inference_config.json').read_text(encoding='utf-8'))


def merge_case(dataset, mutation=None):
    row = plan['datasets'][dataset]
    total, frames = row['sequences'], row['frames']
    root = MemoryPath('/fixture/' + dataset)
    sequences = [MemoryPath('/fixture/data/' + f's{i:03}') for i in range(total)]
    MemoryPath.fs = {}
    configs, dones = [], []
    for gpu, (offset, size) in enumerate(zip(row['offsets'], row['counts'])):
        pred = root / 'shards' / ('gpu' + str(gpu)) / 'predictions'
        cfg = dict(old_cfg, dataset=dataset, root=row['root'], model='/fixture/fixed.pth', head_epoch=0,
                   sequence_offset=offset, limit_sequences=size, max_frames=0, smoke_only=True)
        records = [{'sequence': sequences[i].name, 'frames': frames // total + (i < frames % total)} for i in range(offset, offset + size)]
        done = {'completed': True, 'smoke_only': True, 'sequences': size, 'frames': sum(r['frames'] for r in records), 'records': records}
        configs.append(cfg)
        dones.append(done)
        MemoryPath.fs[str(root / 'shards' / ('gpu' + str(gpu)) / 'inference_completed.txt')] = 'mock marker'
        for r in records:
            MemoryPath.fs[str(pred / (r['sequence'] + '.txt'))] = 'mock prediction path only'
    if mutation:
        mutation(configs, dones, MemoryPath.fs)
    for gpu, (cfg, done) in enumerate(zip(configs, dones)):
        pred = root / 'shards' / ('gpu' + str(gpu)) / 'predictions'
        MemoryPath.fs[str(pred / 'inference_config.json')] = json.dumps(cfg)
        MemoryPath.fs[str(pred / 'inference_completion.json')] = json.dumps(done)
    namespace = {'root': root, 'sequences': sequences, 'dataset': dataset, 'data': MemoryPath(row['root']),
                 'offsets': row['offsets'], 'counts': row['counts'], 'json': json,
                 'selection': {'checkpoint': '/fixture/fixed.pth', 'selected_best_epoch': 0},
                 'configs': [], 'completions': [], 'records': [], 'expected_count': total, 'expected_frames': frames,
                 'files': [MemoryPath(f'/fixture/{seq.name}{suffix}') for seq in sequences for suffix in ('.txt', '_latency.npy', '_recoverability_decisions.npz')]}
    execute([common_node, metadata_loop, *merge_asserts], namespace, MERGE)
    assert len(namespace['records']) == total


for dataset in ('lasher', 'rgbt234'):
    merge_case(dataset)
    check('complete_merge_metadata_positive')
    for mutation in (
        lambda c, d, fs: fs.pop(next(k for k in fs if k.endswith('gpu3/inference_completed.txt'))),
        lambda c, d, fs: d[3].update(completed=False), lambda c, d, fs: d[0].update(smoke_only=False),
        lambda c, d, fs: c[1].update(sequence_offset=0), lambda c, d, fs: c[0].update(limit_sequences=0),
        lambda c, d, fs: c[0].update(model='/fixture/wrong.pth'), lambda c, d, fs: c[0].update(head_epoch=1),
        lambda c, d, fs: c[0].update(write_verification='identity'), lambda c, d, fs: c[0].update(max_frames=128),
        lambda c, d, fs: c[0].update(validation_split='internal98'), lambda c, d, fs: c[0].update(zero_init=True),
        lambda c, d, fs: c[0].update(policy='c1'), lambda c, d, fs: d[0]['records'].reverse(),
        lambda c, d, fs: d[0].update(frames=d[0]['frames']-1), lambda c, d, fs: c[2].update(seed=43)):
        rejects(lambda mutation=mutation: merge_case(dataset, mutation), 'invalid_merge_metadata_rejection')

# Execute the exact per-sequence assertions with shape/finite metadata doubles.
# This does not load NumPy, real arrays, .npy, .npz, model bytes or dataset paths.
class ArrayMetadata:
    def __init__(self, shape, finite=True, positive=True):
        self.shape, self.finite, self.positive = shape, finite, positive

    def __len__(self):
        return self.shape[0]

    def __gt__(self, other):
        return SimpleNamespace(all=lambda: self.positive)


def frame_case(change=None):
    namespace = {'visible': range(10), 'infrared': range(10), 'record': {'frames': 10},
                 'prediction': ArrayMetadata((10, 4)), 'times': ArrayMetadata((9,)),
                 'np': SimpleNamespace(isfinite=lambda arr: SimpleNamespace(all=lambda: arr.finite))}
    if change:
        change(namespace)
    execute([node for node in inner.body if isinstance(node, ast.Assert)], namespace, MERGE)


frame_case()
check('full_frame_finite_latency_metadata_positive')
for change in (
    lambda n: n.update(infrared=range(9)), lambda n: n['record'].update(frames=9),
    lambda n: n.update(prediction=ArrayMetadata((10, 3))), lambda n: n.update(prediction=ArrayMetadata((10, 4), finite=False)),
    lambda n: n.update(times=ArrayMetadata((10,))), lambda n: n.update(times=ArrayMetadata((9,), finite=False)),
    lambda n: n.update(times=ArrayMetadata((9,), positive=False))):
    rejects(lambda change=change: frame_case(change), 'invalid_full_frame_finite_latency_rejection')
suffix_loop = next(node for node in inner.body if isinstance(node, ast.For))
MemoryPath.fs = {'/fixture/pred/s000' + suffix: 'mock path only' for suffix in ('.txt', '_latency.npy', '_recoverability_decisions.npz')}
namespace = {'pred': MemoryPath('/fixture/pred'), 'sequence': MemoryPath('/fixture/data/s000'), 'files': []}
execute([suffix_loop], namespace, MERGE)
assert len(namespace['files']) == 3
check('prediction_latency_timeline_presence_positive')
MemoryPath.fs.pop('/fixture/pred/s000_recoverability_decisions.npz')
rejects(lambda: execute([suffix_loop], namespace, MERGE), 'missing_timeline_rejection')

# Prove the independent metric arithmetic is the already accepted audit's
# arithmetic, through AST identity after only the documented variable renames.
new_audit = ast.parse(AUDIT.read_text(encoding='utf-8'))
original_audit = ast.parse(ORIGINAL_AUDIT.read_text(encoding='utf-8'))
geometry = next(node for node in new_audit.body if isinstance(node, ast.FunctionDef) and node.name == 'geometry')
old_native = next(node for node in original_audit.body if isinstance(node, ast.FunctionDef) and node.name == 'native_audit')
assignments = {node.targets[0].id: node.value for node in ast.walk(old_native)
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
centers = next(node for node in ast.walk(old_native) if isinstance(node, ast.Assign)
               and isinstance(node.targets[0], ast.Tuple) and ast.unparse(node.targets[0]) == '(pc, gc)')
old_expressions = {'pc': centers.value.elts[0], 'gc': centers.value.elts[1],
                   'error': assignments['pr'], 'normalized': assignments['npr'],
                   'right': assignments['right'], 'left': assignments['left'],
                   'intersection': assignments['intersection'], 'overlap': assignments['sr']}


class Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = {'p': 'prediction', 'g': 'target'}.get(node.id, node.id)
        return node


for node in geometry.body:
    if isinstance(node, ast.Assign):
        previous = Rename().visit(copy.deepcopy(old_expressions[node.targets[0].id]))
        assert ast.dump(previous, include_attributes=False) == ast.dump(node.value, include_attributes=False)
        check('native_geometry_expression_matches_prior_actual_audit')
draws_new = next(node.value for node in ast.walk(new_audit) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'draws')
draws_old = assignments['draws']
assert ast.dump(draws_new, include_attributes=False) == ast.dump(draws_old, include_attributes=False)
check('paired_bootstrap_draws_match_prior_actual_audit')
result_new = next(node.value for node in ast.walk(new_audit) if isinstance(node, ast.Assign)
                  and ast.unparse(node.targets[0]) == 'result[metric]')
result_old = next(node.value for node in ast.walk(old_native) if isinstance(node, ast.Assign)
                  and ast.unparse(node.targets[0]) == 'metric_results[metric]')
assert ast.dump(result_new, include_attributes=False) == ast.dump(result_old, include_attributes=False)
check('paired_bootstrap_result_math_matches_prior_actual_audit')
assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr in ('PR', 'NPR', 'SR', 'MPR', 'MSR') for node in ast.walk(new_audit))
check('independent_audit_calls_no_native_metric_functions')
config_assignment = next(node for node in merge_tree.body if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'config')
assert any(keyword.arg == 'output' and ast.unparse(keyword.value) == 'str(out)'
           for keyword in config_assignment.value.keywords)
check('aggregate_output_metadata_corrected')

assert 'torch' not in sys.modules and 'numpy' not in sys.modules
record = {'status': 'PASS', 'counts': COUNTS,
          'scope': 'Exact source compilation, Bash routing and source-extracted control-flow assertions with stdlib in-memory doubles. No operational launcher or evaluator execution.',
          'limits': 'Array checks use shape/finite metadata doubles; no real array arithmetic, complete merge execution, hard-link IO, ground-truth scoring or new native result is certified.',
          'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES},
          'source_fixture_reference_sha256': {str(ORIGINAL_AUDIT): hashlib.sha256(ORIGINAL_AUDIT.read_bytes()).hexdigest()},
          'execution': {'torch_imported': False, 'numpy_imported': False, 'SSH_calls': 0, 'GPU_queries': 0,
                        'checkpoint_reads': 0, 'dataset_reads': 0, 'NPZ_reads': 0, 'NN_calls': 0,
                        'training_or_evaluation_jobs_started': 0, 'reviewed_source_files_modified': False},
          'python_version': sys.version, 'checked_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
(HERE / 'native_source_checks.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': record['status'], 'counts': COUNTS}))
