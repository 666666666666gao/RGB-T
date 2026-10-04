"""Source-only merge review: stdlib AST, manifest and in-memory numeric/gate fixtures."""
import ast
import contextlib
import copy
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import struct
import sys
from types import SimpleNamespace

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
MERGER = RUN / 'actual_partition_merge_cpu.py'
GATE = RUN / 'actual_completed_cache_gate_cpu.py'
COLLECTOR = ROOT / 'research/collect_recoverability.py'
TRAINER = ROOT / 'research/train_recoverability.py'
MODULES = ROOT / 'research/recoverability_modules.py'
CANDIDATES = ROOT / 'research/candidate_learning.py'
sources = [MERGER, GATE, COLLECTOR, TRAINER, MODULES, CANDIDATES]
trees = {path: ast.parse(path.read_text(encoding='utf-8')) for path in sources}
for path, tree in trees.items():
    compile(tree, str(path), 'exec')


def assignment(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def code(nodes, path):
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


prior = json.loads((RUN / 'shard_source_review.json').read_text(encoding='utf-8'))
unchanged = []
for name, expected in prior['source_sha256'].items():
    if name.startswith(('research/', 'scripts/', 'config/', 'trackit/')):
        assert digest(ROOT / name) == expected, name
        unchanged.append(name)
decision_fields = ast.literal_eval(assignment(trees[MODULES], 'DECISION_FIELDS').value)
label_fields = ast.literal_eval(assignment(trees[TRAINER], 'LABEL_FIELDS').value)
assert not set(decision_fields) & (set(label_fields) | {'motion_targets'})
forward = function(trees[TRAINER], 'forward')
assert 'for key in DECISION_FIELDS' in ast.unparse(forward)
load_data = ast.unparse(function(trees[TRAINER], 'load_data'))
assert 'DECISION_FIELDS + LABEL_FIELDS' in load_data
assert "config['prefix_policy']" in load_data and "receipt['completed']" in load_data
assert ast.literal_eval(assignment(trees[MERGER], 'common').value) == ast.literal_eval(assignment(trees[GATE], 'common').value)
for path in (MERGER, GATE):
    imports = [node for node in ast.walk(trees[path]) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any('torch' in ast.unparse(node) for node in imports)
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in ('backward', 'step', 'cuda', 'get_image_path')
                   for node in ast.walk(trees[path]))

jobs, shard_maps = {}, {}
manifest_paths = []
for part, count in (('train', 902), ('validation', 128)):
    original = ROOT / 'refine-logs/runs/recoverability_write_pair/future_policy_jobs' / (part + '.json')
    jobs[part] = json.loads(original.read_text(encoding='utf-8'))
    manifest_paths.append(original)
    assert len(jobs[part]) == len({(j['sequence'], j['query_frame']) for j in jobs[part]}) == count
    shard_maps[part] = eval(compile(ast.Expression(assignment(trees[MERGER], 'shards').value), str(MERGER), 'eval'),
                            {'args': SimpleNamespace(partition=part)})
    combined = []
    for shard, shard_count in shard_maps[part]:
        path = RUN / ('jobs_' + shard + '.json')
        manifest_paths.append(path)
        shard_jobs = json.loads(path.read_text(encoding='utf-8'))['jobs']
        assert len(shard_jobs) == shard_count
        combined.extend(shard_jobs)
    assert combined == jobs[part]
assert shard_maps == {'train': [('train_probe', 16), ('train0_rest', 296), ('train1_rest', 295), ('train2_rest', 295)],
                      'validation': [('validation_probe', 16), ('validation_rest', 112)]}
split_path = ROOT / 'refine-logs/runs/c1_seed42_split.json'
split = json.loads(split_path.read_text(encoding='utf-8'))
assert len(split['train']) == 881 and len(split['validation']) == 98
assert not set(split['train']) & set(split['validation'])
for part in jobs:
    assert {j['sequence'] for j in jobs[part]} <= set(split[part])
    assert all(1 <= j['query_frame'] <= 1024 for j in jobs[part])
assert not {j['sequence'] for j in jobs['train']} & {j['sequence'] for j in jobs['validation']}


class Number:
    """One scalar or vector: only arithmetic required by the extracted IoU functions."""
    def __init__(self, value):
        self.value = tuple(value) if isinstance(value, (list, tuple)) else value

    def __getitem__(self, item):
        if isinstance(item, tuple):
            assert item[0] is Ellipsis
            item = item[1]
        return Number(self.value[item])

    def binary(self, other, operation):
        other = other.value if isinstance(other, Number) else other
        a, b = self.value, other
        if isinstance(a, tuple) or isinstance(b, tuple):
            if not isinstance(a, tuple):
                a = (a,) * len(b)
            if not isinstance(b, tuple):
                b = (b,) * len(a)
            assert len(a) == len(b)
            return Number(tuple(operation(x, y) for x, y in zip(a, b)))
        return Number(operation(a, b))

    def __sub__(self, other): return self.binary(other, lambda a, b: a - b)
    def __add__(self, other): return self.binary(other, lambda a, b: a + b)
    def __truediv__(self, other): return self.binary(other, lambda a, b: a / b)
    def __gt__(self, other): return self.binary(other, lambda a, b: a > b)
    def __float__(self): return float(self.value)
    def prod(self, axis):
        assert axis == -1
        return Number(math.prod(self.value))
    def clamp(self, min): return self.binary(min, lambda a, b: max(a, b))
    def all(self): return all(self.value)


numeric = SimpleNamespace(
    maximum=lambda a, b: a.binary(b, max), minimum=lambda a, b: a.binary(b, min),
    isfinite=lambda a: Number(tuple(math.isfinite(v) for v in a.value)),
    flatnonzero=lambda a: [i for i, value in enumerate(a) if value])
numeric_scope = {'np': numeric, 'torch': numeric}
exec(code([function(trees[MERGER], 'overlaps')], MERGER), numeric_scope)
exec(code([function(trees[CANDIDATES], 'box_iou')], CANDIDATES), numeric_scope)


def measured(box, target, name):
    return float(numeric_scope[name](Number(box), Number(target)))


def exact_iou(box, target):
    box, target = list(map(Fraction, box)), list(map(Fraction, target))
    intersection = math.prod(max(min(box[i + 2], target[i + 2]) - max(box[i], target[i]), 0) for i in range(2))
    area = math.prod(max(box[i + 2] - box[i], 0) for i in range(2))
    target_area = math.prod(target[i + 2] - target[i] for i in range(2))
    return float(intersection / (area + target_area - intersection))


boxes = [(0, 0, 10, 10), (5, 5, 15, 15), (0, 0, 5, 5), (11, 0, 15, 10),
         (0, 0, 0, 10), (9, 3, 2, 7), (1.25, 2.5, 8.75, 9.5)]
for box in boxes:
    target = (0, 0, 10, 10)
    expected = exact_iou(box, target)
    assert abs(measured(box, target, 'overlaps') - expected) <= 1e-14
    assert abs(measured(box, target, 'box_iou') - expected) <= 1e-14
float32 = lambda value: struct.unpack('f', struct.pack('f', value))[0]
rng, max_error = random.Random(42), 0.0
for _ in range(2000):
    width, height = rng.randint(32, 4096), rng.randint(32, 4096)
    x, y = rng.uniform(0, width - 2), rng.uniform(0, height - 2)
    target = (x, y, min(x + rng.uniform(1, width), width), min(y + rng.uniform(1, height), height))
    proposal = (x + rng.uniform(-5, 5), y + rng.uniform(-5, 5),
                target[2] + rng.uniform(-5, 5), target[3] + rng.uniform(-5, 5))
    clipped = tuple(min(max(value, 0), (width, height)[index % 2]) for index, value in enumerate(proposal))
    label = float32(measured(clipped, target, 'box_iou'))
    replay = measured(tuple(map(float32, clipped)), target, 'overlaps')
    max_error = max(max_error, abs(label - replay))
assert max_error <= 1e-4

history = function(trees[COLLECTOR], 'history_arrays')
history_loop = next(node for node in history.body if isinstance(node, ast.For))
history_assert = next(node for node in history.body if isinstance(node, ast.Assert))
history_cases = []
for query in (1, 7, 8, 1023, 1024):
    entries = [{'descriptor': None, 'instance': None, 'evidence': [1.] * 10, 'quality': 1.,
                'box': [0., 0., 10., 10.], 'frame': frame, 'write': False} for frame in range(query)]
    row = {'history_' + key: [None] * 1024 for key in
           ('descriptors', 'instance_descriptors', 'evidence', 'quality', 'boxes', 'frames', 'write')}
    row['history_valid'] = [False] * 1024
    scope = {'row': row, 'context': {'history': entries, 'query': query}, 'capacity': 1024, 'anchor': [1., 2.]}
    exec(code([history_loop, history_assert], COLLECTOR), scope)
    assert row['history_valid'] == [False] * (1024 - query) + [True] * query
    assert [value for value, valid in zip(row['history_frames'], row['history_valid']) if valid] == list(range(query))
    assert row['history_frames'][-1] == query - 1
    history_cases.append({'query': query, 'right_aligned': True, 'frame_zero_retained': True})

collect = function(trees[COLLECTOR], 'collect_batch')
label_loop = next(node for node in ast.walk(collect) if isinstance(node, ast.For)
                  and ast.unparse(node.iter) == "np.flatnonzero(row['history_valid'])")
gt_values = [(0., 0., 10., 10.), (0., 0., 0., 1.), (5., 5., 1., 1.), (math.nan, 0., 10., 10.)]
sequence = [SimpleNamespace(get_bounding_box=lambda gt=gt: Number(gt)) for gt in gt_values]
label_row = {'history_valid': [False] + [True] * 4, 'history_frames': [0, 0, 1, 2, 3],
             'history_boxes': [Number((0., 0., 10., 10.))] * 5, 'history_iou': [-1.] * 5}
scope = {'np': numeric, 'sequence': sequence, 'row': label_row,
         'iou': lambda box, target: float(numeric_scope['box_iou'](box, target))}
exec(code([label_loop], COLLECTOR), scope)
assert label_row['history_iou'] == [-1., 1., -1., -1., -1.]

merger_text = MERGER.read_text(encoding='utf-8')
assert merger_text.index('assert jobs == expected_jobs') < merger_text.index('out.mkdir(parents=True)')
assert merger_text.index('np.savez_compressed') < merger_text.index("'strict_npz_reload_equal': True")
assert merger_text.index("assert sha(parent)", merger_text.index("(out / 'job_completed.txt').write_text")) < merger_text.index("(out / 'partition_cpu_acceptance.json').write_text")
assert 'np.concatenate([data[key] for data in arrays])' in merger_text
assert 'all(np.array_equal(archive[key], merged[key]) for key in merged)' in merger_text
assert 'np.array_equal(frames, np.arange(max(0, frame - 1024), frame))' in merger_text
assert "assert (labels[~gt_valid] == -1).all()" in merger_text

# Execute the exact gate statements with entirely in-memory files and hashes.
# The protected checkpoint digest below is a supplied fixture; no checkpoint is read.
teacher = '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
teacher_digest = 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
base = PurePosixPath('/data/gb/outputs/recoverability_best_policy_merged_20261004')
review_path = '/data/gb/setup/best_policy_merge_source_review_20261004.json'
gate_result = str(base / 'best_policy_cache_cpu_gate.json')
probe_cfg_path = RUN / 'probe_train/config.json'
template_cfg = json.loads(probe_cfg_path.read_text(encoding='utf-8'))
fixture = {review_path: json.dumps({'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional'}),
           template_cfg['split']: json.dumps(split)}
hash_text = lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
for part, count in (('train', 902), ('validation', 128)):
    root = base / 'own' / part
    cfg = template_cfg | {'partition': part, 'clips': count, 'jobs': jobs[part]}
    done = {'completed': True, 'strict_npz_reload_equal': True, 'partition': part, 'clips': count,
            'decision_input_contains_future': False, 'official_tracking_accuracy': False}
    for name, value in {'config.json': json.dumps(cfg), 'completion.json': json.dumps(done),
                        'samples.npz': 'synthetic-archive-placeholder-' + part,
                        'job_completed.txt': 'synthetic completed marker'}.items():
        fixture[str(root / name)] = value
    evidence = []
    for shard, shard_count in shard_maps[part]:
        shard_root = PurePosixPath('/data/gb/outputs/recoverability_best_policy_collect_' + shard + '_20261004')
        hashes = {}
        for name in ('config.json', 'completion.json', 'samples.npz', 'job_completed.txt'):
            value = 'synthetic source ' + shard + '/' + name
            fixture[str(shard_root / name)] = value
            hashes[name] = hash_text(value)
        evidence.append({'root': str(shard_root), 'clips': shard_count, 'GT_current_iou_max_error': 0.,
                         'GT_history_iou_max_error': 0., 'all_valid_past_frames_exact': True, 'artifact_sha256': hashes})
    accepted = {'status': 'PASS', 'partition': part, 'clips': count, 'all_ordered_jobs_exact': True,
                'all_arrays_strict_reload_exact': True, 'causal_prefix_and_copy_before_future': True,
                'prefix_epoch': 4, 'future_epoch': 4, 'source_GT_evidence': evidence, 'decision_fields': decision_fields,
                'schema': {key: {'dtype': '<f4', 'shape': [count, 1]} for key in decision_fields + label_fields},
                'artifact_sha256': {name: hash_text(fixture[str(root / name)])
                                    for name in ('config.json', 'completion.json', 'samples.npz', 'job_completed.txt')}}
    fixture[str(root / 'partition_cpu_acceptance.json')] = json.dumps(accepted)
    fixture['/data/gb/outputs/recoverability_current_policy_merged_20261004/own/' + part + '/config.json'] = json.dumps(cfg)


class MemoryPath(PurePosixPath):
    store = {}
    def read_text(self):
        if str(self) not in self.store:
            raise FileNotFoundError(str(self))
        return self.store[str(self)]
    def read_bytes(self): return self.read_text().encode('utf-8')
    def is_file(self): return str(self) in self.store
    def write_text(self, value):
        self.store[str(self)] = value
        return len(value)


gate_nodes = [node for node in trees[GATE].body if not isinstance(node, (ast.Import, ast.ImportFrom))
              and node is not assignment(trees[GATE], 'sha')]
gate_code = code(gate_nodes, GATE)


def run_gate(files, parent_hash=teacher_digest):
    MemoryPath.store = files
    def synthetic_sha(path):
        return parent_hash if str(path) == teacher else hash_text(MemoryPath(path).read_text())
    scope = {'Path': MemoryPath, 'os': SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES': ''}),
             'json': json, 'sha': synthetic_sha, 'datetime': datetime, 'timedelta': timedelta, 'timezone': timezone}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(gate_code, scope)
    return json.loads(files[gate_result])


passed = run_gate(copy.deepcopy(fixture))
assert passed['status'] == 'PASS' and not passed['training_started'] and not passed['official_goal_completed']
assert passed['execution'] == {'GPU_queries': 0, 'neural_forward_calls': 0, 'optimizer_steps': 0, 'native_test_reads': False}


def edit_json(files, path, edit):
    data = json.loads(files[str(path)])
    edit(data)
    files[str(path)] = json.dumps(data)


def acceptance_path(part): return base / 'own' / part / 'partition_cpu_acceptance.json'


def edit_config(files, part, edit):
    root = base / 'own' / part
    edit_json(files, root / 'config.json', edit)
    edit_json(files, acceptance_path(part), lambda data: data['artifact_sha256'].__setitem__('config.json', hash_text(files[str(root / 'config.json')])) )


negative_cases = [
    ('missing source review', lambda f: f.pop(review_path), FileNotFoundError),
    ('failed source review', lambda f: edit_json(f, review_path, lambda d: d.__setitem__('status', 'FAIL')), AssertionError),
    ('missing validation acceptance', lambda f: f.pop(str(acceptance_path('validation'))), FileNotFoundError),
    ('missing train completed marker', lambda f: f.pop(str(base / 'own/train/job_completed.txt')), AssertionError),
    ('incomplete collection', lambda f: edit_json(f, base / 'own/train/completion.json', lambda d: d.__setitem__('completed', False)), AssertionError),
    ('wrong accepted count', lambda f: edit_json(f, acceptance_path('train'), lambda d: d.__setitem__('clips', 901)), AssertionError),
    ('wrong teacher epoch', lambda f: edit_json(f, acceptance_path('train'), lambda d: d.__setitem__('future_epoch', 25)), AssertionError),
    ('future inputs claimed', lambda f: edit_json(f, base / 'own/train/completion.json', lambda d: d.__setitem__('decision_input_contains_future', True)), AssertionError),
    ('merged artifact changed', lambda f: f.__setitem__(str(base / 'own/train/samples.npz'), 'changed synthetic archive'), AssertionError),
    ('source artifact changed', lambda f: f.__setitem__('/data/gb/outputs/recoverability_best_policy_collect_train0_rest_20261004/samples.npz', 'changed synthetic source'), AssertionError),
    ('current GT error too large', lambda f: edit_json(f, acceptance_path('train'), lambda d: d['source_GT_evidence'][0].__setitem__('GT_current_iou_max_error', 2e-4)), AssertionError),
    ('history GT error too large', lambda f: edit_json(f, acceptance_path('train'), lambda d: d['source_GT_evidence'][0].__setitem__('GT_history_iou_max_error', 2e-4)), AssertionError),
    ('past frames not exact', lambda f: edit_json(f, acceptance_path('train'), lambda d: d['source_GT_evidence'][0].__setitem__('all_valid_past_frames_exact', False)), AssertionError),
    ('reordered original jobs', lambda f: edit_config(f, 'train', lambda d: d['jobs'].reverse()), AssertionError),
    ('mismatched common seed', lambda f: edit_config(f, 'validation', lambda d: d.__setitem__('seed', 43)), AssertionError),
    ('mismatched schema dtype', lambda f: edit_json(f, acceptance_path('validation'), lambda d: d['schema']['valid'].__setitem__('dtype', '|b1')), AssertionError),
    ('mismatched schema tail', lambda f: edit_json(f, acceptance_path('validation'), lambda d: d['schema']['valid'].__setitem__('shape', [128, 2])), AssertionError),
    ('cross-partition sequence leak', lambda f: edit_json(f, template_cfg['split'], lambda d: d['validation'].__setitem__(0, d['train'][0])), AssertionError),
]
rejected = []
for name, mutate, expected in negative_cases:
    files = copy.deepcopy(fixture)
    mutate(files)
    try:
        run_gate(files)
    except expected:
        assert gate_result not in files
        rejected.append(name)
    else:
        raise AssertionError('Gate accepted: ' + name)
files = copy.deepcopy(fixture)
try:
    run_gate(files, parent_hash='changed synthetic checkpoint hash')
except AssertionError:
    assert gate_result not in files
    rejected.append('changed parent checkpoint hash')
else:
    raise AssertionError('Gate accepted changed parent')

assert 'numpy' not in sys.modules and 'torch' not in sys.modules
record_paths = sources + manifest_paths + [split_path, probe_cfg_path,
    ROOT / 'research/collect_rollouts.py', ROOT / 'research/recoverability_tracker.py',
    ROOT / 'trackit/core/operator/numpy/bbox/utility/image.py',
    ROOT / 'trackit/datasets/MMOT/specialization/memory_mapped/dataset.py',
    RUN / 'remote_MMOT_dataset.py', RUN / 'remote_train_lasher.yaml',
    RUN / 'shard_source_review.json', Path(__file__).resolve()]
record = {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional',
          'runtime_independently_attested': False, 'fixture_scope': 'Source AST and synthetic in-memory arithmetic/files only; not real CPU cache acceptance.',
          'created_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'python_version': sys.version,
          'checks': {'compiled_source_files': len(sources), 'prior_main_sources_unchanged': unchanged,
                     'exact_partition_jobs': {part: len(value) for part, value in jobs.items()},
                     'exact_source_count_maps': shard_maps, 'split_counts': {part: len(split[part]) for part in jobs},
                     'decision_fields': decision_fields, 'label_fields': label_fields, 'decision_label_separation': True,
                     'exact_iou_fixtures': len(boxes), 'float32_storage_fixtures': 2000,
                     'float32_storage_max_iou_error': max_error, 'iou_tolerance': 1e-4,
                     'history_source_loop_cases': history_cases, 'history_label_sentinel_fixture': label_row['history_iou'],
                     'synthetic_full_gate_pass_cases': 1, 'synthetic_full_gate_rejections': rejected,
                     'exact_ordered_concat_and_reload_source_checks': True, 'gate_requires_final_partition_receipt': True},
          'source_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in dict.fromkeys(record_paths)},
          'execution': {'numpy_imported': False, 'torch_imported': False, 'dataset_reads': 0, 'checkpoint_reads': 0,
                        'actual_npz_reads': 0, 'actual_acceptance_receipt_reads': 0, 'collection_config_metadata_read': True,
                        'SSH_calls': 0, 'GPU_queries': 0, 'neural_forward_calls': 0, 'optimizer_steps': 0,
                        'native_test_reads': 0, 'production_edits': False}}
(RUN / 'merge_source_checks.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': record['status'], 'exact_jobs': record['checks']['exact_partition_jobs'],
                  'numeric_cases': len(boxes) + 2000, 'gate_rejections': len(rejected),
                  'float32_storage_max_iou_error': max_error, 'production_execution': False}))
