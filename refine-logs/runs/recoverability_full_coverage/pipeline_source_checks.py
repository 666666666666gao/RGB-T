"""Full-coverage pipeline source fixtures only; no real models, arrays, datasets, GPU or SSH."""
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
RUNNER = ROOT / 'scripts/run_full_coverage_native_complete.sh'
REPORT = ROOT / 'scripts/report_full_coverage_native_complete.sh'
LAUNCH = HERE / 'launch_native_complete.py'
MERGE = HERE / 'merge_native_complete.py'
AUDIT = HERE / 'audit_native_complete_cpu.py'
ORIGINAL_AUDIT = ROOT.parent / 'write_pair_native_rgbt234_cpu_audit_original_20261004.py'
NEW = [RUNNER, REPORT, LAUNCH, MERGE, AUDIT, ROOT / 'scripts/run_full_coverage_fit.sh'] + [HERE / name for name in ('actual_partition_merge_cpu.py', 'actual_completed_cache_gate_cpu.py', 'actual_full_fit_cpu_audit.py', 'launch_fit.py', 'memory_monitor.py', 'complete_pipeline.py')]
UPSTREAM = [ROOT / p for p in (
    'research/evaluate_recoverability.py', 'scripts/run_temporal.sh',
    'research/collect_core_metrics.py', 'research/collect_recoverability_metrics.py',
    'research/paired_sequence_bootstrap.py', 'research/plot_core_metrics.py', 'evaluation.py')]
FILES = NEW + UPSTREAM
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
for path in (RUNNER, REPORT, ROOT / 'scripts/run_full_coverage_fit.sh', ROOT / 'scripts/run_temporal.sh'):
    subprocess.run([BASH, '-n', str(path)], check=True, capture_output=True, text=True)
    check('bash_syntax')
assert hashlib.sha256(UPSTREAM[0].read_bytes()).hexdigest() == '08a1d2c548cf68410a769cf7eaf72f5727b16df6a409f14d55c8b7e91a75b7a8'
plan = json.loads((ROOT / 'refine-logs/runs/recoverability_best_policy_fit/native_complete_plan.json').read_text(encoding='utf-8'))
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
                         '--output': f'/data/gb/outputs/recoverability_full_coverage_native_full_20261004/{dataset}/shards/gpu{gpu}/predictions'}
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
review_key = setup + 'full_coverage_pipeline_source_review_20261004.json'
audit_key = setup + 'full_coverage_actual_full_fit_cpu_acceptance_20261004.json'
selection_key = setup + 'full_coverage_native_selection_20261004.json'
runner_key = setup + 'run_full_coverage_native_complete_20261004.sh'
arm_names = ('pairwise_lr4', 'pairwise_lr5', 'budgeted_lr4', 'budgeted_lr5')
launch_tree = ast.parse(LAUNCH.read_text(encoding='utf-8'))
review = {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional',
          'source_sha256': {'scripts/run_full_coverage_native_complete.sh': hashlib.sha256(RUNNER.read_bytes()).hexdigest()}}
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
        MemoryPath.fs = {audit_key: json.dumps(a), review_key: json.dumps(r), runner_key: RUNNER.read_bytes().decode(), '/data/gb/outputs/recoverability_full_coverage_merged_20261004/full_coverage_cache_cpu_gate.json': json.dumps({'status':'PASS', 'matched_partitions':{'train':881,'validation':98}, 'all881_98_video_names_exact':True})}
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

# Confirm the accepted training source tree and native arithmetic are reused.
PRIOR = ROOT / 'refine-logs/runs/recoverability_best_policy_fit'
COLLECTION_PRIOR = ROOT / 'refine-logs/runs/recoverability_best_policy'
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
full = read(PRIOR / 'actual_full_cpu_acceptance.json')
m0 = read(PRIOR / 'actual_m0_cpu_acceptance.json')
for receipt, epochs, steps in ((m0, 3, 9), (full, 60, 180)):
    assert receipt['status'] == 'PASS' and receipt['all_four_lr_ranking_arms_passed']
    assert receipt['batch_size'] == 416 and receipt['epochs_per_arm'] == epochs and receipt['optimizer_steps_per_arm'] == steps
    check('existing_actual_all_four_receipt_metadata')
research_paths = list((ROOT / 'research').glob('*.py'))
assert len(research_paths) == 39
for path in research_paths:
    key = '/data/gb/experiments/recoverability_best_policy_fit_20261004/research/' + path.name
    assert sha(path) == full['source_sha256'][key], path
    check('accepted_private_research_source_unchanged')
teacher_arm = max(arm_names, key=lambda arm: full['arms'][arm]['best_validation']['utility'])
assert teacher_arm == 'pairwise_lr4' and full['arms'][teacher_arm]['best_epoch'] == 46
check('existing_receipt_current_best_teacher46')
native_review = read(PRIOR / 'native_source_review.json')
for path in UPSTREAM:
    assert sha(path) == native_review['source_sha256'][path.relative_to(ROOT).as_posix()]
    check('accepted_native_source_unchanged')
for name in ('merge_native_complete.py', 'audit_native_complete_cpu.py'):
    old = (PRIOR / name).read_text(encoding='utf-8')
    for a, b in (('recoverability_best_policy_native_full_20261004', 'recoverability_full_coverage_native_full_20261004'),
                 ('best_policy_native_selection_20261004', 'full_coverage_native_selection_20261004'),
                 ('best_policy_native_source_review_20261004', 'full_coverage_pipeline_source_review_20261004'),
                 ('best_policy_complete', 'full_coverage_complete')):
        old = old.replace(a, b)
    assert ast.dump(ast.parse(old)) == ast.dump(ast.parse((HERE / name).read_text(encoding='utf-8')))
    check('native_whole_AST_only_declared_path_label_changes')


def assigned(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def evaluate(expression, namespace):
    return eval(compile(ast.Expression(copy.deepcopy(expression)), '<scalar source fixture>', 'eval'), namespace)


fit_path = HERE / 'actual_full_fit_cpu_audit.py'
fit = ast.parse(fit_path.read_text(encoding='utf-8'))
old_fit = ast.parse((PRIOR / 'actual_best_policy_fit_cpu_audit.py').read_text(encoding='utf-8'))
for name in ('replay', 'metric_errors', 'state_changes'):
    assert ast.dump(function(fit, name)) == ast.dump(function(old_fit, name))
    check('fit_replay_metric_state_helpers_unchanged')
assert ast.literal_eval(assigned(fit, 'ARMS').value) == {
    'pairwise_lr4': ('pairwise', 1e-4), 'pairwise_lr5': ('pairwise', 1e-5),
    'budgeted_lr4': ('budgeted', 1e-4), 'budgeted_lr5': ('budgeted', 1e-5)}
assert evaluate(assigned(fit, 'EPOCHS').value, {'args': SimpleNamespace(stage='full')}) == 60
assert ast.literal_eval(assigned(fit, 'BATCH').value) == 416
assert ast.literal_eval(assigned(fit, 'totals').value) == {'A': 109187, 'B': 117133, 'C': 86791}
assert sum(ast.literal_eval(assigned(fit, 'totals').value).values()) == 313111
assert ast.literal_eval(assigned(fit, 'frozen').value) == []
check('full_training_counts_and_all_ABC_parameters')
trainer = ast.parse((ROOT / 'research/train_recoverability.py').read_text(encoding='utf-8'))
modules = ast.parse((ROOT / 'research/recoverability_modules.py').read_text(encoding='utf-8'))
decision_fields = ast.literal_eval(assigned(modules, 'DECISION_FIELDS').value)
label_fields = ast.literal_eval(assigned(trainer, 'LABEL_FIELDS').value)
assert len(decision_fields) == 19 and len(label_fields) == 5 and not set(decision_fields) & set(label_fields)
assert 'for key in DECISION_FIELDS' in ast.unparse(function(trainer, 'forward'))
check('training_labels_excluded_from_model_input')
main = function(trainer, 'main')
changed = assigned(main, 'changed')
reload_nodes = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'load_state_dict']
assert any(node.lineno > changed.lineno for node in reload_nodes)
assert not any(isinstance(node, (ast.Break, ast.Continue)) for node in ast.walk(main))
check('real_full_epoch_loop_terminal_update_before_best_reload')
best_expr = assigned(fit, 'best_epoch').value
for utilities, expected in (([.5, .5, .4], 0), ([.4, .6, .6], 1)):
    metrics = [{'utility': value} for value in utilities]
    assert evaluate(best_expr, {'EPOCHS': 2, 'metrics': metrics}) == expected
    check('retained_first_max_epoch_scalar')
retained_guard = next(node for node in ast.walk(fit) if isinstance(node, ast.Assert)
                      and isinstance(node.test, ast.IfExp))
for name in ('A', 'B', 'C'):
    for epoch, changed_count, accepted in ((0, 0, True), (0, 1, False), (1, 1, True), (1, 0, False)):
        actual = evaluate(retained_guard.test, {'name': name, 'frozen': [], 'best_epoch': epoch,
                                               'change': {'changed_tensors': changed_count}})
        assert actual is accepted
        check('retained_best_vs_parent_state_branch')
triplet_guard = next(node.test for node in ast.walk(fit) if isinstance(node, ast.Assert)
                     and ast.unparse(node.test).startswith('triplets =='))
triplets = [[epoch, step, (epoch - 1) * 3 + step] for epoch in range(1, 61) for step in (1, 3)]
assert len(triplets) == 120 and triplets[-1][-1] == 180
assert evaluate(triplet_guard, {'triplets': triplets, 'EPOCHS': 60})
assert not evaluate(triplet_guard, {'triplets': triplets[:-1], 'EPOCHS': 60})
check('exact_60_epoch_180_update_log_triplets')
shape_guard = next(node.test for node in ast.walk(fit) if isinstance(node, ast.Assert)
                   and ast.unparse(node.test).startswith('value.shape =='))
for size, accepted in ((98, True), (128, False), (97, False)):
    namespace = {'value': ArrayMetadata((size,)), 'saved': {'field': ArrayMetadata((size,))}, 'key': 'field',
                 'np': SimpleNamespace(isfinite=lambda value: SimpleNamespace(all=lambda: value.finite))}
    assert evaluate(shape_guard, namespace) is accepted
    check('all98_saved_field_shape_constraint')

# Run only Bash variable routing and the original argparse function; no trainer imports.
import argparse
import math
fit_runner = (ROOT / 'scripts/run_full_coverage_fit.sh').read_text(encoding='utf-8')
route = fit_runner[:fit_runner.index('/data/gb/envs/gola/bin/python -c')]
route += '\nepochs=60\nparent=/fixture/current_teacher46.pth\n'
route += next(line for line in fit_runner.splitlines() if line.startswith('run=')) + '\n'
command = fit_runner[fit_runner.index('/data/gb/envs/gola/bin/python -u -m research.train_recoverability'):]
command = command[:command.index('\ndate -Iseconds')]
route += command.replace('/data/gb/envs/gola/bin/python -u -m research.train_recoverability', 'printf "%s\\n"')
parser_scope = {'argparse': argparse, '__doc__': 'source fixture'}
execute([function(trainer, 'arguments')], parser_scope, ROOT / 'research/train_recoverability.py')
saved_argv = sys.argv
for gpu, arm in enumerate(arm_names):
    result = subprocess.run([BASH, '-c', route, 'fixture', str(gpu), arm, 'full'], check=True, capture_output=True, text=True)
    sys.argv = ['fixture'] + result.stdout.splitlines()
    parsed = parser_scope['arguments']()
    ranking, lr = ast.literal_eval(assigned(fit, 'ARMS').value)[arm]
    assert parsed.action_ranking == ranking and parsed.lr == lr
    assert parsed.epochs == 60 and parsed.batch_size == 416 and parsed.seed == 42
    assert parsed.epochs * math.ceil(881 / parsed.batch_size) == 180
    assert parsed.init_checkpoint == '/fixture/current_teacher46.pth'
    assert parsed.train == ['/data/gb/outputs/recoverability_full_coverage_merged_20261004/own/train']
    assert parsed.validation == '/data/gb/outputs/recoverability_full_coverage_merged_20261004/own/validation'
    assert parsed.output == '/data/gb/outputs/recoverability_full_coverage_' + arm + '_full_20261004'
    assert parsed.search_supervision == 'oracle' and parsed.write_verification == 'action'
    assert parsed.weight_decay == 1e-4 and parsed.threshold == .03 and parsed.frozen_modules == []
    assert not parsed.write_pair_calibration and not parsed.prefer_last_prefix
    check('four_matched_full_fit_Bash_and_argparse_routes')
sys.argv = saved_argv
for arm, stage in (('bad', 'full'), ('pairwise_lr4', 'm0'), ('pairwise_lr4', 'bad')):
    result = subprocess.run([BASH, '-c', route, 'fixture', '0', arm, stage], capture_output=True, text=True)
    assert result.returncode != 0 and not result.stdout
    check('new_M0_or_invalid_fit_route_rejected')

# Exercise every literal fit guard on metadata fixtures, including old partial coverage.
good = {'status': 'PASS', 'matched_partitions': {'train': 881, 'validation': 98},
        'all881_98_video_names_exact': True, 'all_ordered_jobs_exact': True,
        'all_GT_current_and_history_replayed': True, 'all_schema_and_sources_verified': True,
        'prefix_epoch': 46, 'future_epoch': 46, 'review_independence': 'same-family',
        'acceptance_status': 'provisional', 'all_four_lr_ranking_arms_passed': True,
        'batch_size': 416, 'optimizer_steps_per_arm': 9, 'epochs_per_arm': 3,
        'teacher': '/fixture/current_teacher46.pth'}
for line in fit_runner.splitlines():
    if "python -c 'import json;" not in line:
        continue
    literal = line.split("python -c '", 1)[1].split("'", 1)[0]
    tree = ast.parse(literal)
    def fit_guard(value):
        execute(tree.body, {'json': json, 'open': lambda path: io.StringIO(json.dumps(value))}, '<shell guard>')
    fit_guard(good)
    check('literal_full_fit_gate_positive')
    for key, value in (('status', 'PENDING'), ('matched_partitions', {'train': 902, 'validation': 128}),
                       ('all881_98_video_names_exact', False), ('all_ordered_jobs_exact', False),
                       ('all_GT_current_and_history_replayed', False), ('all_schema_and_sources_verified', False),
                       ('future_epoch', 4), ('review_independence', 'self'), ('acceptance_status', 'final'),
                       ('all_four_lr_ranking_arms_passed', False), ('batch_size', 384),
                       ('optimizer_steps_per_arm', 8), ('epochs_per_arm', 2)):
        if 'd["' + key + '"]' in literal:
            rejects(lambda key=key, value=value: fit_guard(dict(good, **{key: value})), 'literal_full_fit_gate_rejection')

# The current/history audit itself is unchanged; new manifests require all video names.
partition_path = HERE / 'actual_partition_merge_cpu.py'
partition = ast.parse(partition_path.read_text(encoding='utf-8'))
old_partition = ast.parse((COLLECTION_PRIOR / 'actual_partition_merge_cpu.py').read_text(encoding='utf-8'))
assert ast.dump(function(partition, 'overlaps')) == ast.dump(function(old_partition, 'overlaps'))
new_job_loop = next(node for node in ast.walk(partition) if isinstance(node, ast.For)
                    and ast.unparse(node.target) == '(index, job)')
old_job_loop = next(node for node in ast.walk(old_partition) if isinstance(node, ast.For)
                    and ast.unparse(node.target) == '(index, job)')
assert ast.dump(new_job_loop) == ast.dump(old_job_loop)
check('complete_current_and_valid_past_GT_audit_AST_unchanged')
gate_path = HERE / 'actual_completed_cache_gate_cpu.py'
gate_tree = ast.parse(gate_path.read_text(encoding='utf-8'))
common = ast.literal_eval(assigned(gate_tree, 'common').value)
assert common == ast.literal_eval(assigned(partition, 'common').value)
name_guard = next(node.test for node in partition.body if isinstance(node, ast.Assert)
                  and ast.unparse(node.test).startswith("{job['sequence']"))
ordered_guard = next(node.test for node in partition.body if isinstance(node, ast.Assert)
                     and ast.unparse(node.test).startswith('jobs == expected_jobs'))
synthetic_split = {'train': [f'train{i:03}' for i in range(881)],
                   'validation': [f'val{i:03}' for i in range(98)]}
synthetic_jobs = {part: [{'sequence': name, 'query_frame': 128} for name in names]
                  for part, names in synthetic_split.items()}
prepared_shards = {part + '_gpu' + str(gpu): {'clips': count}
                   for part, counts in (('train', (221, 220, 220, 220)), ('validation', (25, 25, 24, 24)))
                   for gpu, count in enumerate(counts)}
for part, count in (('train', 881), ('validation', 98)):
    namespace = {'expected_jobs': synthetic_jobs[part], 'split': synthetic_split,
                 'args': SimpleNamespace(partition=part), 'prepared': {'shards': prepared_shards}}
    assert evaluate(name_guard, namespace)
    shards = evaluate(assigned(partition, 'shards').value, namespace)
    assert sum(size for _, size in shards) == count and [name for name, _ in shards] == [part + '_gpu' + str(i) for i in range(4)]
    assert evaluate(ordered_guard, {'jobs': synthetic_jobs[part], 'expected_jobs': synthetic_jobs[part], 'expected_count': count})
    assert not evaluate(ordered_guard, {'jobs': synthetic_jobs[part][::-1], 'expected_jobs': synthetic_jobs[part], 'expected_count': count})
    assert not evaluate(name_guard, dict(namespace, expected_jobs=synthetic_jobs[part][:-1]))
    check('ordered_all_video_four_shard_positive_and_negative')

# Execute the complete cache gate over in-memory metadata and mock artifact bytes.
base_key = '/data/gb/outputs/recoverability_full_coverage_merged_20261004'
prepared_key = '/data/gb/setup/full_coverage_jobs_cpu_acceptance_20261004.json'
teacher_key = '/fixture/current_teacher46.pth'
fixture_review = {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional'}
fixture_prepared = {'status': 'PASS', 'teacher': teacher_key,
                    'teacher_sha256': hashlib.sha256(b'fixture teacher').hexdigest(),
                    'teacher_epoch': 46, 'jobs': synthetic_jobs}


def cache_case(change=None):
    configs, completions, acceptances = {}, {}, {}
    for part, count in (('train', 881), ('validation', 98)):
        configs[part] = dict.fromkeys(common, 'same fixture source')
        configs[part].update(partition=part, clips=count, jobs=copy.deepcopy(synthetic_jobs[part]),
                             prefix_model=teacher_key, prefix_checkpoint_epoch=46, future_checkpoint_epoch=46,
                             future_policy_mode='own', prefix_write_verification='action', split='/fixture/split.json')
        completions[part] = {'partition': part, 'clips': count, 'completed': True, 'strict_npz_reload_equal': True,
                             'decision_input_contains_future': False, 'official_tracking_accuracy': False}
        acceptances[part] = {'status': 'PASS', 'partition': part, 'clips': count, 'all_ordered_jobs_exact': True,
                             'all_arrays_strict_reload_exact': True, 'causal_prefix_and_copy_before_future': True,
                             'prefix_epoch': 46, 'future_epoch': 46, 'all_partition_video_names_exact': True,
                             'source_GT_evidence': [{'GT_current_iou_max_error': 0.0, 'GT_history_iou_max_error': 0.0,
                                'all_valid_past_frames_exact': True, 'root': '/fixture/source/' + part, 'artifact_sha256': {}}],
                             'decision_fields': list(decision_fields),
                             'schema': {'fixture': {'shape': [count, 7, 5], 'dtype': '<f4'}}}
    if change:
        change(configs, completions, acceptances)
    MemoryPath.fs = {prepared_key: json.dumps(fixture_prepared), teacher_key: 'fixture teacher',
                     review_key: json.dumps(fixture_review), '/fixture/split.json': json.dumps(synthetic_split)}
    for part in configs:
        root = base_key + '/own/' + part
        artifacts = {'config.json': json.dumps(configs[part]), 'completion.json': json.dumps(completions[part]),
                     'samples.npz': 'fixture byte string; never loaded as NPZ', 'job_completed.txt': 'fixture completion'}
        acceptances[part]['artifact_sha256'] = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in artifacts.items()}
        MemoryPath.fs.update({root + '/' + name: value for name, value in artifacts.items()})
        MemoryPath.fs[root + '/partition_cpu_acceptance.json'] = json.dumps(acceptances[part])
    namespace = {'Path': MemoryPath, 'os': SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES': ''}), 'hashlib': hashlib,
                 'json': json, 'datetime': datetime, 'timedelta': timedelta, 'timezone': timezone}
    execute(gate_tree.body, namespace, gate_path)
    result = json.loads(MemoryPath.fs[base_key + '/full_coverage_cache_cpu_gate.json'])
    assert result['status'] == 'PASS' and result['matched_partitions'] == {'train': 881, 'validation': 98}
    assert result['prefix_epoch'] == result['future_epoch'] == 46 and not result['training_started']
    return result


cache_case()
check('complete_cache_gate_metadata_positive')
for change in (
    lambda c, d, a: c['train']['jobs'].pop(),
    lambda c, d, a: c['validation']['jobs'][0].update(sequence='wrong_video'),
    lambda c, d, a: c['train'].update(prefix_checkpoint_epoch=4),
    lambda c, d, a: c['validation'].update(prefix_model='/fixture/old4.pth'),
    lambda c, d, a: a['train'].update(status='PENDING'),
    lambda c, d, a: d['validation'].update(completed=False),
    lambda c, d, a: d['train'].update(decision_input_contains_future=True),
    lambda c, d, a: a['train'].update(all_partition_video_names_exact=False),
    lambda c, d, a: a['validation'].update(all_ordered_jobs_exact=False),
    lambda c, d, a: a['train']['source_GT_evidence'][0].update(GT_current_iou_max_error=.01),
    lambda c, d, a: a['train']['source_GT_evidence'][0].update(GT_history_iou_max_error=.01),
    lambda c, d, a: a['train']['source_GT_evidence'][0].update(all_valid_past_frames_exact=False),
    lambda c, d, a: a['validation']['schema']['fixture'].update(shape=[98, 7, 4]),
    lambda c, d, a: a['validation']['schema']['fixture'].update(dtype='<f8'),
):
    rejects(lambda change=change: cache_case(change), 'invalid_complete_cache_gate_metadata_rejection')

# Fit launch uses exactly the four full arms and one non-neural monitor.
fit_launch_path = HERE / 'launch_fit.py'
fit_launch = ast.parse(fit_launch_path.read_text(encoding='utf-8'))
launch_key = '/data/gb/setup/full_coverage_fit_launch_20261004.json'
sanity_key = '/data/gb/setup/best_policy_fit_m0_acceptance_20261004.json'
MemoryPath.fs = {review_key: json.dumps(fixture_review), base_key + '/full_coverage_cache_cpu_gate.json': json.dumps(good),
                 sanity_key: json.dumps(good)}
spawned = []


def fit_popen(argv, **kwargs):
    index = len(spawned)
    if index < 4:
        assert argv == ['bash', '/data/gb/setup/run_full_coverage_fit_20261004.sh', str(index), arm_names[index], 'full']
    else:
        assert index == 4 and argv == [sys.executable, '/data/gb/setup/full_coverage_memory_monitor_20261004.py']
    assert kwargs['cwd'] == MemoryPath('/data/gb/experiments/recoverability_best_policy_fit_20261004') or str(kwargs['cwd']) == '/data/gb/experiments/recoverability_best_policy_fit_20261004'
    assert kwargs['start_new_session']
    spawned.append(argv)
    return SimpleNamespace(pid=2000 + index)


def fit_run(argv, **kwargs):
    assert argv == ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']
    return SimpleNamespace(stdout='\n'.join(f'{i}, 0' for i in range(4)))


execute(fit_launch.body, {'Path': MemoryPath, 'json': json, 'datetime': datetime, 'timedelta': timedelta,
                         'timezone': timezone, 'sys': sys,
                         'subprocess': SimpleNamespace(run=fit_run, Popen=fit_popen, STDOUT=-2)}, fit_launch_path)
assert len(spawned) == 5
fit_launched = json.loads(MemoryPath.fs[launch_key])
assert fit_launched['status'] == 'LAUNCHED_NOT_COMPLETED' and fit_launched['epochs'] == 60 and fit_launched['optimizer_steps_expected'] == 180
check('four_full_fit_queues_and_one_metadata_monitor_mock')
monitor = ast.parse((HERE / 'memory_monitor.py').read_text(encoding='utf-8'))
sleep_call = next(node for node in ast.walk(monitor) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == 'sleep')
assert ast.literal_eval(sleep_call.args[0]) == 180
check('memory_monitor180second_cadence')

# Automatic handoff must wait on the original owners and stop on genuine failure.
pipeline_path = HERE / 'complete_pipeline.py'
pipeline = ast.parse(pipeline_path.read_text(encoding='utf-8'))
wait_function = function(pipeline, 'wait_jobs')
mock_rows = [{'runner_pid': 3000 + index} for index in range(4)]
mock_groups = [[MemoryPath('/fixture/queue' + str(index) + '/' + stage) for stage in ('train', 'validation')]
               for index in range(4)]


def wait_case(all_done=False, dead=False):
    MemoryPath.fs = {str(marker): 'done' for group in mock_groups for marker in group} if all_done else {
        str(marker): 'done' for marker in mock_groups[0]}
    queried, sleeps = [], []
    def ps(argv, **kwargs):
        assert argv[:4] == ['ps', '-o', 'stat=', '-p'] and kwargs['check'] is True
        pid = int(argv[4])
        assert pid != 3000
        queried.append(pid)
        return SimpleNamespace(stdout='Z' if dead else 'S')
    def sleep(seconds):
        assert seconds == 180
        sleeps.append(seconds)
        MemoryPath.fs.update({str(marker): 'done' for group in mock_groups for marker in group})
    scope = {'subprocess': SimpleNamespace(run=ps), 'time': SimpleNamespace(sleep=sleep)}
    execute([wait_function], scope, pipeline_path)
    scope['wait_jobs'](mock_rows, mock_groups)
    assert queried == ([] if all_done else [3001, 3002, 3003])
    assert sleeps == ([] if all_done else [180])


wait_case()
check('original_owner_wait_skips_completed_queues_and_polls180')
wait_case(all_done=True)
check('all_completed_groups_require_no_process_query')
rejects(lambda: wait_case(dead=True), 'dead_unfinished_owner_rejected')
import unittest
MemoryPath.fs = {}
def gone(argv, **kwargs):
    assert kwargs['check'] is True
    raise subprocess.CalledProcessError(1, argv)
scope = {'subprocess': SimpleNamespace(run=gone), 'time': SimpleNamespace(sleep=lambda seconds: None)}
execute([wait_function], scope, pipeline_path)
with unittest.TestCase().assertRaises(subprocess.CalledProcessError):
    scope['wait_jobs'](mock_rows, mock_groups)
check('missing_unfinished_owner_ps_failure_propagates')

# Execute fixed handoff control flow with every command replaced by a metadata sink.
collection_key = '/data/gb/setup/full_coverage_collection_launch_20261004.json'
native_launch_key = '/data/gb/setup/full_coverage_native_launch_20261004.json'
progress_key = '/data/gb/setup/full_coverage_pipeline_progress_20261004.json'
fit_rows = [{'runner_pid': 4000 + i,
             'root': '/data/gb/outputs/recoverability_full_coverage_' + arm + '_full_20261004'}
            for i, arm in enumerate(arm_names)]
native_rows = [{'runner_pid': 5000 + i} for i in range(4)]
MemoryPath.fs = {review_key: json.dumps(fixture_review), collection_key: json.dumps({'runs': mock_rows}),
                 launch_key: json.dumps({'runs': fit_rows}), native_launch_key: json.dumps({'runs': native_rows})}
for gpu in range(4):
    for part in ('train', 'validation'):
        MemoryPath.fs['/data/gb/outputs/recoverability_full_coverage_collect_' + part + '_gpu' + str(gpu) + '_20261004/job_completed.txt'] = 'completed fixture'
    for dataset in ('lasher', 'rgbt234'):
        MemoryPath.fs['/data/gb/outputs/recoverability_full_coverage_native_full_20261004/' + dataset + '/shards/gpu' + str(gpu) + '/inference_completed.txt'] = 'completed fixture'
for row in fit_rows:
    MemoryPath.fs[row['root'] + '/job_completed.txt'] = 'completed fixture'
commands = []


def handoff_run(argv, **kwargs):
    assert kwargs['check'] is True and kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
    main_path = '/data/gb/GOLA'
    private_path = '/data/gb/experiments/recoverability_best_policy_fit_20261004'
    cwd = str(kwargs['cwd'])
    if argv[1].endswith('full_coverage_actual_full_fit_cpu_audit_20261004.py'):
        assert cwd == private_path and kwargs['env']['PYTHONPATH'] == private_path + ':' + main_path
    elif argv[0] == 'bash':
        assert cwd == main_path and kwargs['env']['PYTHONPATH'] == main_path
    else:
        assert cwd == main_path and kwargs['env']['PYTHONPATH'] == main_path + ':' + main_path
    commands.append([PurePosixPath(argv[1]).name, *argv[2:]])
    return SimpleNamespace(returncode=0)


handoff_scope = {'Path': MemoryPath, 'json': json, 'os': SimpleNamespace(environ={}),
                 'subprocess': SimpleNamespace(run=handoff_run), 'time': SimpleNamespace(sleep=lambda seconds: None),
                 'datetime': datetime, 'timedelta': timedelta, 'timezone': timezone}
execute(pipeline.body, handoff_scope, pipeline_path)
expected_commands = [
    ['full_coverage_actual_partition_merge_cpu_20261004.py', '--partition', 'train'],
    ['full_coverage_actual_partition_merge_cpu_20261004.py', '--partition', 'validation'],
    ['full_coverage_actual_completed_cache_gate_cpu_20261004.py'],
    ['launch_full_coverage_fit_20261004.py'],
    ['full_coverage_actual_full_fit_cpu_audit_20261004.py', '--stage', 'full'],
    ['launch_full_coverage_native_complete_20261004.py'],
    ['merge_full_coverage_native_complete_20261004.py', '--dataset', 'lasher'],
    ['report_full_coverage_native_complete_20261004.sh', 'lasher'],
    ['audit_full_coverage_native_complete_cpu_20261004.py', '--dataset', 'lasher'],
    ['merge_full_coverage_native_complete_20261004.py', '--dataset', 'rgbt234'],
    ['report_full_coverage_native_complete_20261004.sh', 'rgbt234'],
    ['audit_full_coverage_native_complete_cpu_20261004.py', '--dataset', 'rgbt234'],
]
assert commands == expected_commands
progress_value = json.loads(MemoryPath.fs[progress_key])
assert progress_value['stage'] == 'COMPLETE_TRAINING_AND_BOTH_FULL_NATIVE_CPU_REPORTS_READY_GOAL_REQUIRES_RESULT_AUDIT'
assert progress_value['automatic_restarts'] == 0 and progress_value['poll_interval_seconds'] == 180 and not progress_value['official_goal_completed']
assert not any(isinstance(node, ast.Try) for node in ast.walk(pipeline))
check('fixed_complete_handoff_command_order_and_CPU_private_import_paths')
rejects(lambda: execute(pipeline.body, handoff_scope, pipeline_path), 'duplicate_handoff_progress_rejected')

# Bounded operational helper review: SCP/Popen are replaced by fixture sinks.
stage_helper = ROOT.parent / 'stage_full_coverage_pipeline.ps1'
launch_helper = ROOT.parent / 'launch_full_coverage_controller_20261004.py'
controller_launcher = ast.parse(launch_helper.read_text(encoding='utf-8'))
compile(controller_launcher, str(launch_helper), 'exec')
check('operational_controller_launcher_python_compile')
pwsh = r'C:\Program Files\WindowsApps\Microsoft.PowerShell_7.6.6.0_x64__8wekyb3d8bbwe\pwsh.exe'
stage_fixture = r'''
$ErrorActionPreference = 'Stop'
$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile('__HELPER__', [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw 'Stager parse failed' }
$global:pipelineScpCalls = [System.Collections.Generic.List[object]]::new()
function scp {
    param([string]$Source, [string]$Destination)
    $global:pipelineScpCalls.Add(@{source=$Source; destination=$Destination})
}
& '__HELPER__' | Out-Null
if ($pipelineScpCalls.Count -ne 13) { throw 'Unexpected mock SCP count' }
foreach ($call in $pipelineScpCalls) {
    if (-not $call.destination.StartsWith('2027:/data/gb/setup/')) { throw 'Unexpected SCP destination' }
}
ConvertTo-Json -Compress -InputObject @($pipelineScpCalls)
'''.replace('__HELPER__', str(stage_helper).replace("'", "''"))
staged = subprocess.run([pwsh, '-NoProfile', '-Command', stage_fixture], capture_output=True, encoding='utf-8', check=True)
scp_calls = json.loads(staged.stdout)
assert len(scp_calls) == 13 and len({row['destination'] for row in scp_calls}) == 13
assert all(row['destination'].startswith('2027:/data/gb/setup/') for row in scp_calls)
assert {Path(row['source']).resolve() for row in scp_calls} == set(NEW) | {HERE / 'pipeline_source_review.json'}
staged_names = {row['destination'].rsplit('/', 1)[1] for row in scp_calls}
assert {row[0] for row in expected_commands} <= staged_names
assert 'complete_full_coverage_pipeline_20261004.py' in staged_names
assert 'run_full_coverage_fit_20261004.sh' in staged_names and 'run_full_coverage_native_complete_20261004.sh' in staged_names
assert 'full_coverage_memory_monitor_20261004.py' in staged_names
check('PowerShell_stager_parse_and13_mock_SCP_setup_only_routes')
controller_key = '/data/gb/setup/complete_full_coverage_pipeline_20261004.py'
controller_launch_key = '/data/gb/setup/full_coverage_pipeline_launch_20261004.json'
controller_text = pipeline_path.read_text(encoding='utf-8')


def controller_case(bad_handle=False, bad_source=False, reset=True):
    if reset:
        review = fixture_review | {'source_sha256': {'refine-logs/runs/recoverability_full_coverage/complete_pipeline.py':
                  'incorrect' if bad_source else hashlib.sha256(controller_text.encode()).hexdigest()}}
        MemoryPath.fs = {review_key: json.dumps(review), collection_key: json.dumps({'runs': mock_rows}), controller_key: controller_text}
    handles, spawned = [], []
    def ps(argv, **kwargs):
        assert argv[:4] == ['ps', '-o', 'pid=,stat=,args=', '-p'] and kwargs['check'] is True
        handles.append(int(argv[4]))
        return SimpleNamespace(stdout=argv[4] + ' S ' + ('wrong_script.sh' if bad_handle else 'bash /data/gb/setup/run_full_coverage_collect_20261004.sh'))
    def popen(argv, **kwargs):
        assert argv == ['/data/gb/envs/gola/bin/python', '-u', controller_key]
        assert kwargs['cwd'] == '/data/gb/GOLA' and kwargs['start_new_session']
        spawned.append(argv)
        return SimpleNamespace(pid=6000)
    scope = {'Path': MemoryPath, 'json': json, 'hashlib': hashlib, 'datetime': datetime, 'timedelta': timedelta,
             'timezone': timezone, 'subprocess': SimpleNamespace(run=ps, Popen=popen, STDOUT=-2)}
    execute(controller_launcher.body, scope, launch_helper)
    assert handles == [row['runner_pid'] for row in mock_rows] and len(spawned) == 1
    result = json.loads(MemoryPath.fs[controller_launch_key])
    assert result['status'] == 'ORIGINAL_CONTROLLER_LAUNCHED_NOT_COMPLETE' and result['automatic_restarts'] == 0
    assert result['poll_seconds'] == 180


controller_case()
check('reviewed_controller_once_original_four_handles_mock')
rejects(lambda: controller_case(reset=False), 'duplicate_operational_controller_launch_rejected')
rejects(lambda: controller_case(bad_handle=True), 'wrong_original_runner_handle_rejected')
rejects(lambda: controller_case(bad_source=True), 'unreviewed_controller_bytes_rejected')

assert len(NEW) == 12 and len(set(NEW)) == 12
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
record = {'status': 'PASS', 'counts': COUNTS,
          'scope': 'Exact source compilation, unchanged-source checks, Bash routes and stdlib scalar/in-memory metadata fixtures.',
          'limits': 'No real arrays, model import, checkpoint loading, ground-truth scoring, remote execution or new runtime acceptance.',
          'source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in NEW},
          'supporting_source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in UPSTREAM},
          'operational_helper_sha256': {str(path.resolve()).replace('\\', '/'): sha(path) for path in (stage_helper, launch_helper)},
          'existing_teacher_metadata': {'arm': teacher_arm, 'epoch': full['arms'][teacher_arm]['best_epoch']},
          'execution': {'torch_imported': False, 'numpy_imported': False, 'SSH_calls': 0, 'GPU_queries': 0,
                        'checkpoint_reads': 0, 'dataset_reads': 0, 'NPZ_reads': 0, 'NN_calls': 0,
                        'training_or_evaluation_jobs_started': 0, 'reviewed_source_files_modified': False},
          'runtime_attested': False, 'review_independence': 'same-family', 'acceptance_status': 'provisional',
          'python_version': sys.version, 'checked_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
(HERE / 'pipeline_source_checks.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': record['status'], 'counts': COUNTS, 'runtime_attested': False}))

