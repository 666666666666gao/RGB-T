"""Source-only final-merger fixtures; all runtime inputs and outputs stay in memory."""
import ast
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / 'merge_complete_goal_report.py'
CANONICAL = SOURCE.relative_to(ROOT).as_posix()
SOURCE_BYTES = SOURCE.read_bytes()
TREE = ast.parse(SOURCE_BYTES.decode('utf-8'), filename=str(SOURCE))
compile(TREE, str(SOURCE), 'exec')
COUNTS = {'canonical_source_compile': 1}


def check(name):
    COUNTS[name] = COUNTS.get(name, 0) + 1


class MemoryPath:
    """Implements only the reviewed file operations, without host filesystem access."""
    files = {}
    directories = set()
    writes = []
    reads = []

    def __init__(self, value):
        self.path = PurePosixPath(str(value))

    def __str__(self):
        return str(self.path)

    def __truediv__(self, value):
        return MemoryPath(self.path / value)

    def __eq__(self, other):
        return self.path == other.path

    def with_name(self, value):
        return MemoryPath(self.path.with_name(value))

    def read_text(self):
        self.reads.append(str(self))
        return self.files[str(self)]

    def read_bytes(self):
        return self.read_text().encode('utf-8')

    def exists(self):
        return str(self) in self.files or str(self) in self.directories

    def is_file(self):
        return str(self) in self.files

    def stat(self):
        return SimpleNamespace(st_size=len(self.files[str(self)].encode('utf-8')))

    def mkdir(self):
        assert not self.exists()
        self.directories.add(str(self))

    def write_text(self, value):
        self.files[str(self)] = value
        self.writes.append(str(self))

    @contextlib.contextmanager
    def open(self, mode='r', newline=None):
        assert mode in ('r', 'w')
        stream = io.StringIO(self.read_text() if mode == 'r' else '', newline=newline)
        yield stream
        if mode == 'w':
            self.write_text(stream.getvalue())
        stream.close()


# Import the actual stdlib-only reusable functions, then substitute their I/O only.
helper_path = ROOT / 'research/merge_abc_complete.py'
spec = importlib.util.spec_from_file_location('source_review_merge_abc_complete', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.Path = MemoryPath
assert helper.EXPECTED == {'lasher': (245, 220703, 19, ('PR', 'NPR', 'SR')),
                           'rgbt234': (234, 116649, 12, ('MPR', 'MSR'))}
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
check('real_stdlib_native_report_and_uncertainty_imported')

imports = [node for node in TREE.body if isinstance(node, (ast.Import, ast.ImportFrom))]
assert {node.module for node in imports if isinstance(node, ast.ImportFrom)} == {
    'datetime', 'pathlib', 'research.merge_abc_complete'}
assert {item.name for node in imports if isinstance(node, ast.Import) for item in node.names} == {
    'csv', 'hashlib', 'json', 'os'}
reuse = next(node for node in imports if isinstance(node, ast.ImportFrom)
             and node.module == 'research.merge_abc_complete')
assert [item.name for item in reuse.names] == ['EXPECTED', 'native_report', 'uncertainty']
assert not any(isinstance(node, ast.Try) for node in ast.walk(TREE))
BODY = ast.Module(body=[node for node in TREE.body if node not in imports], type_ignores=[])
CODE = compile(BODY, str(SOURCE), 'exec')
check('stdlib_only_merger_and_existing_helpers_reused')

SETUP = '/data/gb/setup/'
BASE = '/data/gb/outputs/recoverability_full_coverage_native_full_20261004'
OUT = BASE + '/complete_report'
REVIEW = SETUP + 'full_coverage_goal_report_source_review_20261004.json'
GATE = '/data/gb/outputs/recoverability_full_coverage_merged_20261004/full_coverage_cache_cpu_gate.json'
TRAINING = SETUP + 'full_coverage_actual_full_fit_cpu_acceptance_20261004.json'
SELECTION = SETUP + 'full_coverage_native_selection_20261004.json'
ARMS = ('pairwise_lr4', 'pairwise_lr5', 'budgeted_lr4', 'budgeted_lr5')
CHOSEN = ARMS[0]
CHECKPOINT = '/synthetic_only/' + CHOSEN + '/best.pth'
VARIANTS = ['baseline', 'c1', 'full_coverage_complete']
TEST = unittest.TestCase()


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def fixture(gains=(2.25, 2.5, 2.75, 3.0, 3.25), best_epoch=17):
    files = {CHECKPOINT: 'Synthetic checkpoint byte string; no tensor or real checkpoint.'}
    digest = hashlib.sha256(files[CHECKPOINT].encode()).hexdigest()
    documents = {
        REVIEW: {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional'},
        GATE: {'status': 'PASS', 'matched_partitions': {'train': 881, 'validation': 98},
               'all881_98_video_names_exact': True},
        TRAINING: {'status': 'PASS', 'all_four_lr_ranking_arms_passed': True,
                   'epochs_per_arm': 60, 'optimizer_steps_per_arm': 180, 'batch_size': 416, 'arms': {}},
        SELECTION: {'status': 'PASS', 'both_datasets_same_fixed_checkpoint': True,
                    'selected_arm': CHOSEN, 'selected_best_epoch': best_epoch,
                    'checkpoint': CHECKPOINT, 'checkpoint_sha256': digest,
                    'best0_is_parent_not_new_learning': best_epoch == 0,
                    'selection_rule': 'Maximum developer VAL98 utility; first arm on ties; no native-test selection.'},
    }
    for arm in ARMS:
        documents[TRAINING]['arms'][arm] = {
            'status': 'PASS', 'active_modules': ['A', 'B', 'C'], 'frozen_modules': [],
            'root': '/synthetic_only/' + arm, 'best_epoch': best_epoch,
            'C1_all8_retained_tensors_exact': True, 'strict25_tensor_CPU_load': True,
            'best0_is_parent_not_new_learning': best_epoch == 0, 'learning_gain_claimed': False,
            'artifact_sha256': {'best.pth': digest},
        }
    metric_index = 0
    for dataset, (count, frames, attribute_count, metrics) in helper.EXPECTED.items():
        root = BASE + '/' + dataset
        baseline = {metric: (40.125, 51.25, 62.5, 70.125, 60.75)[metric_index + i]
                    for i, metric in enumerate(metrics)}
        scores = {'baseline': baseline, 'c1': {m: v + 1.0 for m, v in baseline.items()},
                  VARIANTS[-1]: {m: v + gains[metric_index + i] for i, (m, v) in enumerate(baseline.items())}}
        metric_index += len(metrics)
        config = {'model': CHECKPOINT, 'head_epoch': best_epoch, 'write_verification': 'action',
                  'zero_init': False, 'validation_split': None, 'max_frames': 0, 'limit_sequences': 0,
                  'sequence_offset': 0, 'arbitrary_historical_state_restoration': False,
                  'timing_excludes_diagnostic_transfer_and_serialization': True}
        variants = {}
        for variant in VARIANTS:
            values = scores[variant]
            variants[variant] = {
                'overall_metrics_percent': values,
                'attributes': {'attribute_' + str(i): {'sequences': count, **values} for i in range(attribute_count)},
                'mean_curves': {m: {'thresholds': [0.0, 1.0], 'values': [v / 100, v / 100]} for m, v in values.items()},
                'original_inference_config': copy.deepcopy(config),
                'efficiency': {'tracking_fps': 20.0, 'includes_decode_crop_forward_selection_update': True},
            }
        report = {'dataset': dataset, 'all_actual_ground_truth_verified': True, 'sequences': count,
                  'frames': frames, 'variants': variants,
                  'diagnostic_protocol': {'event_is_localization_failure': 'not semantic identity ground truth'}}
        rows = [{'variant': variant, 'sequence': 'synthetic_' + str(i), **scores[variant]}
                for variant in VARIANTS for i in range(count)]
        documents[root + '/core_report/full_report.json'] = report
        files[root + '/core_report/per_sequence.csv'] = csv_text(rows)
        exact = {}
        for reference in ('baseline', 'c1'):
            folder = root + '/' + reference + '_paired_report'
            paired = copy.deepcopy(report)
            paired['variants'] = {v: paired['variants'][v] for v in (reference, VARIANTS[-1])}
            documents[folder + '/full_report.json'] = paired
            files[folder + '/per_sequence.csv'] = csv_text([row for row in rows if row['variant'] in paired['variants']])
            stats = {m: {'mean_delta_percentage_points': scores[VARIANTS[-1]][m] - scores[reference][m],
                         'percentile_95_interval_percentage_points': [-0.5, 4.0],
                         'improved_sequences': count, 'worsened_sequences': 0, 'tied_sequences': 0}
                     for m in metrics}
            exact[reference] = copy.deepcopy(stats)
            documents[folder + '/paired_bootstrap.json'] = {
                'args': {'iterations': 5000, 'seed': 42},
                'scope': 'fixed checkpoints, not repeated training-seed uncertainty',
                'datasets': {dataset: {'compared_variants': [reference, VARIANTS[-1]],
                                       'sequences': count, 'metrics': stats}},
            }
            for name in ('official_curves.png', 'official_curves.pdf', 'attribute_deltas.png', 'attribute_deltas.pdf'):
                files[folder + '/' + name] = 'Synthetic nonempty plot placeholder; not an image.'
        documents[root + '/independent_native_cpu_acceptance.json'] = {
            'status': 'PASS', 'dataset': dataset, 'sequences': count, 'frames': frames,
            'same_checkpoint': CHECKPOINT, 'all_actual_GT_sequence_metrics_attributes_and_curves_exact': True,
            'score_not_another_models_output': True, 'native_overall_percent': copy.deepcopy(scores),
            'paired_5000_seed42_bootstrap_exact': exact,
        }
        documents[root + '/mechanism_report/full_recoverability_report.json'] = {
            'completed': True, 'native_accuracy_completed': False, 'sequences': count,
            'definitions': {'template_pollution': 'localization proxy; no semantic distractor labels',
                            'timing': 'concurrent-load comparisons descriptive',
                            'harm_rescue': 'same-state selection; not independent future-rollout causality'},
            'variants': {VARIANTS[-1]: {'counters': {'wrong_template_updates_localization_proxy': 3}}},
        }
        files[root + '/report_completed.txt'] = 'Synthetic report completion.'
        files[root + '/mechanism_report/recoverability_metrics_completed.txt'] = 'Synthetic mechanism completion.'
    files.update({path: json.dumps(value) for path, value in documents.items()})
    return files


def modify_json(files, path, edit):
    value = json.loads(files[path])
    edit(value)
    files[path] = json.dumps(value)


def run_case(files, output_exists=False):
    MemoryPath.files = copy.deepcopy(files)
    MemoryPath.directories = {OUT} if output_exists else set()
    MemoryPath.reads, MemoryPath.writes = [], []
    namespace = {'Path': MemoryPath, 'os': SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES': ''}),
                 'hashlib': hashlib, 'json': json, 'csv': csv, 'datetime': datetime, 'timedelta': timedelta,
                 'timezone': timezone, 'EXPECTED': helper.EXPECTED, 'native_report': helper.native_report,
                 'uncertainty': helper.uncertainty}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(CODE, namespace)
    return json.loads(MemoryPath.files[OUT + '/complete_core_report.json'])


good = fixture()
result = run_case(good)
assert result['completed'] and result['official_tracking_accuracy']
assert result['acceptance']['all_five_overall_meet_plus_two']
assert result['goal_result'] == 'ALL_FIVE_PLUS_TWO_VERIFIED_REQUIRES_ROOT_COMPLETION_AUDIT'
assert not result['execution']['system_goal_status_changed']
assert result['execution'] == {'GPU_queries': 0, 'neural_forward_calls': 0, 'optimizer_steps': 0,
                                'system_goal_status_changed': False}
assert result['selected_checkpoint'] == json.loads(good[SELECTION])
assert result['training_CPU_acceptance'] == json.loads(good[TRAINING])
assert result['all881_98_video_coverage_CPU'] == json.loads(good[GATE])
assert MemoryPath.writes == [OUT + '/' + name for name in (
    'complete_core_report.json', 'acceptance.csv', 'per_sequence.csv', 'attributes.csv', 'complete_core_report_completed.txt')]
for name, count in (('acceptance.csv', 5), ('per_sequence.csv', 1437), ('attributes.csv', 93)):
    assert len(list(csv.DictReader(io.StringIO(MemoryPath.files[OUT + '/' + name])))) == count
expected_order = [('lasher', 'PR'), ('lasher', 'NPR'), ('lasher', 'SR'), ('rgbt234', 'MPR'), ('rgbt234', 'MSR')]
assert [(row['dataset'], row['metric']) for row in result['acceptance']['metrics']] == expected_order
for row in result['acceptance']['metrics']:
    source = result['datasets'][row['dataset']]['native']['variants']
    assert row['ABC_percent'] == source[VARIANTS[-1]]['overall_metrics_percent'][row['metric']]
    assert row['target_percent'] == row['baseline_percent'] + 2
    assert row['delta_vs_c1_percentage_points'] == row['ABC_percent'] - row['c1_percent']
    assert row['paired_95_lower_percentage_points'] == -0.5
check('complete_five_metric_csv_counts_aliases_and_no_system_goal_change')
for dataset in helper.EXPECTED:
    root = BASE + '/' + dataset
    saved = result['datasets'][dataset]
    assert saved['native'] == json.loads(good[root + '/core_report/full_report.json'])
    assert saved['mechanism'] == json.loads(good[root + '/mechanism_report/full_recoverability_report.json'])
    assert not saved['mechanism']['native_accuracy_completed']
    assert not saved['native']['variants'][VARIANTS[-1]]['original_inference_config']['arbitrary_historical_state_restoration']
    assert saved['independent_native_CPU'] == json.loads(good[root + '/independent_native_cpu_acceptance.json'])
    for reference in ('baseline', 'c1'):
        assert saved['paired'][reference] == json.loads(good[root + '/' + reference + '_paired_report/paired_bootstrap.json'])
    check('native_mechanism_uncertainty_and_source_limitations_preserved')
assert any('Concurrent four-GPU' in text for text in result['limitations'])
assert any('independent multiple training seeds are not required' in text for text in result['limitations'])
assert 'one eligible query per every881 TRAIN/98 developerVAL video' in result['scope']
assert 'no all-frame/end-to-end' in result['scope']
check('coverage_timing_and_fixed_checkpoint_uncertainty_scope')
assert 'full ABC training runs executed60epochs' in result['scope']
assert 'retained-weight provenance is separate' in result['scope']
assert result['selected_weight_provenance'] == (
    'Selected positive-epoch weights were learned during this completed full-coverage60epoch/180update training round.')
check('positive_epoch_full_coverage_selected_weight_provenance')

for gains, expected in (((2.0,) * 5, True), ((2.000001,) * 5, True)):
    boundary = run_case(fixture(gains=gains))
    assert boundary['acceptance']['all_five_overall_meet_plus_two'] is expected
    check('exact_two_and_above_point_threshold')
for index in range(5):
    gains = [3.0] * 5
    gains[index] = 1.999999
    below = run_case(fixture(gains=gains))
    row = below['acceptance']['metrics'][index]
    assert round(row['delta_vs_baseline_percentage_points'], 2) == 2.0
    assert not row['meets_plus_two'] and not below['acceptance']['all_five_overall_meet_plus_two']
    assert below['goal_result'] == 'ACTIVE_UNMET'
    check('each_below_two_metric_stays_unmet_without_display_rounding')
parent = run_case(fixture(best_epoch=0))
assert parent['selected_checkpoint']['best0_is_parent_not_new_learning']
assert not parent['training_CPU_acceptance']['arms'][CHOSEN]['learning_gain_claimed']
check('epoch0_selected_parent_evidence_preserved_without_new_learning_claim')
assert parent['selected_weight_provenance'] == (
    'The full-coverage60epoch/180update training runs were completed, but retained epoch0 is the earlier partial-coverage teacher; '
    'the selected weights contain no new learning from this full-coverage round.')
check('epoch0_partial_coverage_teacher_and_completed_full_fit_distinguished')


def reject_json(path, edit, name):
    files = copy.deepcopy(good)
    modify_json(files, path, edit)
    with TEST.assertRaises((AssertionError, KeyError)):
        run_case(files)
    assert not MemoryPath.writes
    check(name)


for path, edits in (
    (REVIEW, ({'status': 'PENDING'}, {'review_independence': 'cross-family'}, {'acceptance_status': 'final'})),
    (GATE, ({'status': 'PENDING'}, {'matched_partitions': {'train': 549, 'validation': 76}}, {'all881_98_video_names_exact': False})),
    (TRAINING, ({'status': 'PENDING'}, {'all_four_lr_ranking_arms_passed': False}, {'epochs_per_arm': 59},
                {'optimizer_steps_per_arm': 179}, {'batch_size': 384})),
    (SELECTION, ({'status': 'PENDING'}, {'both_datasets_same_fixed_checkpoint': False}, {'selected_best_epoch': 18},
                 {'checkpoint': '/synthetic_only/other/best.pth'}, {'checkpoint_sha256': 'wrong-existing-binding'})),
):
    for changes in edits:
        reject_json(path, lambda value, changes=changes: value.update(changes), 'source_coverage_fit_selection_gate_rejection')
for changes in ({'status': 'PENDING'}, {'active_modules': ['A', 'B']}, {'frozen_modules': ['C']},
                {'C1_all8_retained_tensors_exact': False}, {'artifact_sha256': {'best.pth': 'wrong-existing-binding'}}):
    reject_json(TRAINING, lambda value, changes=changes: value['arms'][CHOSEN].update(changes),
                'chosen_complete_ABC_C1_and_existing_checkpoint_binding_rejection')
for dataset, (count, frames, _, _) in helper.EXPECTED.items():
    root = BASE + '/' + dataset
    actual = root + '/independent_native_cpu_acceptance.json'
    for path in (actual, root + '/report_completed.txt', root + '/mechanism_report/recoverability_metrics_completed.txt'):
        files = copy.deepcopy(good)
        del files[path]
        with TEST.assertRaises((AssertionError, KeyError)):
            run_case(files)
        assert not MemoryPath.writes
        check('each_dataset_actual_completion_required')
    for changes in ({'status': 'PENDING'}, {'dataset': 'wrong'}, {'sequences': count - 1}, {'frames': frames - 1},
                    {'same_checkpoint': '/synthetic_only/other/best.pth'},
                    {'all_actual_GT_sequence_metrics_attributes_and_curves_exact': False},
                    {'score_not_another_models_output': False}):
        reject_json(actual, lambda value, changes=changes: value.update(changes), 'actual_GT_native_acceptance_rejection')
    core = root + '/core_report/full_report.json'
    reject_json(core, lambda value: value.update(all_actual_ground_truth_verified=False), 'native_report_helper_GT_gate_rejection')
    reject_json(core, lambda value: value['variants'][VARIANTS[-1]]['overall_metrics_percent'].update(unofficial=50),
                'native_report_helper_exact_metric_alias_rejection')
    for changes in ({'model': '/synthetic_only/different.pth'}, {'head_epoch': 18}, {'write_verification': 'raw'},
                    {'zero_init': True}, {'validation_split': 'developer_subset'}, {'max_frames': 20},
                    {'limit_sequences': 5}, {'sequence_offset': 1}):
        reject_json(core, lambda value, changes=changes: value['variants'][VARIANTS[-1]]['original_inference_config'].update(changes),
                    'same_selected_checkpoint_and_complete_native_config_rejection')
    sequence_path = root + '/core_report/per_sequence.csv'
    files = copy.deepcopy(good)
    rows = list(csv.DictReader(io.StringIO(files[sequence_path])))
    files[sequence_path] = csv_text(rows[:-1])
    with TEST.assertRaises(AssertionError):
        run_case(files)
    assert not MemoryPath.writes
    check('native_report_helper_partial_sequence_rejection')
    for reference in ('baseline', 'c1'):
        folder = root + '/' + reference + '_paired_report'
        bootstrap = folder + '/paired_bootstrap.json'
        for changes in ({'iterations': 4999}, {'seed': 43}):
            reject_json(bootstrap, lambda value, changes=changes: value['args'].update(changes), 'bootstrap_5000_seed42_rejection')
        metric = helper.EXPECTED[dataset][3][0]
        reject_json(bootstrap, lambda value, metric=metric: value['datasets'][dataset]['metrics'][metric].update(mean_delta_percentage_points=99),
                    'real_uncertainty_helper_mean_delta_rejection')
        reject_json(bootstrap, lambda value, metric=metric: value['datasets'][dataset]['metrics'][metric].update(percentile_95_interval_percentage_points=[99, 100]),
                    'independent_actual_bootstrap_interval_mismatch_rejection')
        for field in ('overall_metrics_percent', 'attributes', 'mean_curves'):
            reject_json(folder + '/full_report.json', lambda value, field=field: value['variants'][VARIANTS[-1]][field].update(extra=0),
                        'paired_core_metric_attribute_curve_identity_rejection')
        for contents in (None, ''):
            files = copy.deepcopy(good)
            plot = folder + '/official_curves.pdf'
            if contents is None:
                del files[plot]
            else:
                files[plot] = contents
            with TEST.assertRaises(AssertionError):
                run_case(files)
            assert not MemoryPath.writes
            check('missing_or_empty_required_plot_rejection')
    for changes in ({'completed': False}, {'sequences': count - 1}):
        reject_json(root + '/mechanism_report/full_recoverability_report.json',
                    lambda value, changes=changes: value.update(changes), 'mechanism_completion_rejection')
with TEST.assertRaises(AssertionError):
    run_case(good, output_exists=True)
assert not MemoryPath.writes
check('existing_complete_report_rejected_without_overwrite')

# Source evidence for the prerequisite semantics, without importing or running them.
fit_source = (HERE / 'actual_full_fit_cpu_audit.py').read_text(encoding='utf-8')
selection_source = (HERE / 'launch_native_complete.py').read_text(encoding='utf-8')
assert "[row['epoch'] for row in metrics] == list(range(EPOCHS + 1))" in fit_source
assert "len(ck['model']) == 25" in fit_source and "len(c1['head']) == 8" in fit_source
assert "best_epoch = max(range(EPOCHS + 1), key=lambda epoch: metrics[epoch]['utility'])" in fit_source
assert "selected = max(arms, key=lambda arm: accepted['arms'][arm]['best_validation']['utility'])" in selection_source
assert "first arm in declared order on ties; no native-test selection" in selection_source
assert "'strict25_tensor_CPU_load': True" in fit_source
assert "'learning_gain_claimed': False" in fit_source
check('producer_61_epoch_metrics_25_states_frozen8_and_first_VAL_max_source_contract')
assert SOURCE.read_bytes() == SOURCE_BYTES
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
record = {
    'status': 'PASS', 'counts': COUNTS,
    'scope': 'One canonical merger source, actual stdlib native_report/uncertainty functions, and full in-memory synthetic report fixtures.',
    'limits': 'Fixtures do not attest real collection, tensor state, training, checkpoint bytes, dataset GT, metrics, plots or runtime completion.',
    'source_sha256': {CANONICAL: hashlib.sha256(SOURCE_BYTES).hexdigest()},
    'execution': {'torch_imported': False, 'numpy_imported': False, 'SSH_calls': 0, 'GPU_queries': 0,
                  'checkpoint_reads': 0, 'dataset_reads': 0, 'NN_calls': 0, 'jobs_started': 0,
                  'reviewed_source_files_modified': False, 'system_goal_status_changed': False},
    'runtime_attested': False, 'review_independence': 'same-family', 'acceptance_status': 'provisional',
    'reviewer_model': 'gpt-6-astra', 'reasoning_effort': 'max',
    'python_version': sys.version, 'checked_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
}
(HERE / 'goal_report_source_checks.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': record['status'], 'counts': COUNTS, 'runtime_attested': False}))
