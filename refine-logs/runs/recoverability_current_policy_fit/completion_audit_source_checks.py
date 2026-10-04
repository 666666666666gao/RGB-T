"""Bounded source-only CPU fixtures; never execute actual completion audits."""
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NAMES = [f"{mode}_actual_{kind}_cpu_audit.py"
         for mode in ("oracle", "selector") for kind in ("full_fit", "full98")]
NAMES += ["four_factor_closure.py"]
payload = {
    "helpers": {name: (HERE / name).read_text(encoding="utf-8") for name in NAMES},
    "predecessors": {kind: (HERE.parent / "recoverability_warm_budgeted_action" /
                            f"actual_{kind}_cpu_audit.py").read_text(encoding="utf-8")
                     for kind in ("full_fit", "full98")},
    "runners": {mode: (REPO / "scripts" / filename).read_text(encoding="utf-8") for mode, filename in
                (("oracle", "run_recoverability_current_policy_fit.sh"),
                 ("selector", "run_recoverability_current_policy_selector_fit.sh"))},
    "reviews": {mode: json.loads((HERE / filename).read_text(encoding="utf-8")) for mode, filename in
                (("oracle", "source_review.json"), ("selector", "selector_source_review.json"))},
}

REMOTE = r'''
import ast
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

start = time.perf_counter()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
helpers = payload['helpers']
trees = {name: ast.parse(source, filename=name) for name, source in helpers.items()}
for name, source in helpers.items():
    compile(source, name, 'exec')
checks = {'all_five_remote_CPU_compile': True}

def functions(tree):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, ast.FunctionDef)}

def constants(tree, wanted):
    selected = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id in wanted for target in node.targets)]
    namespace = {'Path': Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<source-constants-only>', 'exec'), namespace)
    return namespace

oracle_fit = helpers['oracle_actual_full_fit_cpu_audit.py']
expected_selector = oracle_fit.replace('current_policy_fit', 'current_policy_selector_fit')
expected_selector = expected_selector.replace('/experiments/recoverability_current_policy_selector_fit_20261004',
                                             '/experiments/recoverability_current_policy_fit_20261004')
expected_selector = expected_selector.replace('search_supervision="oracle"', 'search_supervision="selector"')
expected_selector = expected_selector.replace('cfg["search_supervision"] == "oracle"', 'cfg["search_supervision"] == "selector"')
assert expected_selector == helpers['selector_actual_full_fit_cpu_audit.py']
assert helpers['oracle_actual_full98_cpu_audit.py'].replace('current_policy_fit', 'current_policy_selector_fit') == helpers['selector_actual_full98_cpu_audit.py']
checks['selector_exact_scoped_replacements'] = True
assert functions(trees['oracle_actual_full_fit_cpu_audit.py']) == functions(ast.parse(payload['predecessors']['full_fit']))
assert functions(trees['oracle_actual_full98_cpu_audit.py']) == functions(ast.parse(payload['predecessors']['full98']))
checks['oracle_all_function_ASTs_equal_to_accepted_predecessors'] = True

private = Path('/data/gb/experiments/recoverability_current_policy_fit_20261004')
source_hashes = {}
mode_constants = {}
for mode in ('oracle', 'selector'):
    stem = 'current_policy_fit' if mode == 'oracle' else 'current_policy_selector_fit'
    expected_review = payload['reviews'][mode]
    actual_review = json.loads(Path(f'/data/gb/setup/{stem}_source_review_20261004.json').read_text())
    assert actual_review == expected_review
    assert actual_review['status'] == 'PASS'
    verified = actual_review['verification']['local_remote_source_bytes_equal']
    for relative, field in [('research/train_recoverability.py', 'trainer_sha256'),
                            ('research/recoverability_modules.py', 'modules_sha256'),
                            (f'scripts/run_recoverability_{stem}.sh', 'runner_sha256')]:
        path = private / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == verified[field]
        source_hashes[str(path)] = digest
    assert (private / f'scripts/run_recoverability_{stem}.sh').read_text() == payload['runners'][mode]
    tree = trees[f'{mode}_actual_full98_cpu_audit.py']
    c = constants(tree, {'ROOTS', 'NEW', 'REFERENCES', 'FIT_AUDIT', 'SOURCES', 'OUT'})
    mode_constants[mode] = c
    assert c['FIT_AUDIT'] == Path(f'/data/gb/setup/{stem}_actual_full_fit_cpu_audit_20261004.json')
    assert c['OUT'] == Path(f'/data/gb/setup/{stem}_actual_full98_cpu_audit_20261004.json')
    assert len(c['ROOTS']) == 7 and len(c['NEW']) == 2
    for policy in ('c1', 'own'):
        label = f'{stem}_{policy}_b384'
        root = Path(f'/data/gb/outputs/recoverability_{label}_full_20261004')
        assert label in c['NEW'] and c['ROOTS'][label] == root / 'predictions'
        refs = ['baseline', 'c1', 'write_pair_reference_own_b384', f'warm_budgeted_action_{policy}_b384']
        assert c['REFERENCES'][label] == refs
        source = payload['runners'][mode].replace('\\\n', ' ')
        command = next(line for line in source.splitlines() if '-m research.collect_recoverability_metrics' in line)
        command = command.replace('${policy}', policy).replace('$run', str(root))
        args = shlex.split(command)
        def values(flag):
            offset = args.index(flag) + 1
            end = next((i for i in range(offset, len(args)) if args[i].startswith('--')), len(args))
            return args[offset:end]
        assert values('--labels') == [label]
        assert values('--runs') == [str(c['ROOTS'][label])]
        assert values('--reference-labels') == refs
        assert values('--references') == [str(c['ROOTS'][ref]) for ref in refs]
        assert values('--output') == [str(root / 'report')]
    fit_source = helpers[f'{mode}_actual_full_fit_cpu_audit.py']
    assert 'source["prefix_checkpoint_epoch"] == 25' in fit_source
    assert 'parent["epoch"] == 4' in fit_source and 'cfg["initial_checkpoint_epoch"] == 4' in fit_source
    assert 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72' in fit_source
    assert f'cfg["search_supervision"] == "{mode}"' in fit_source
    assert 'verified = review["verification"]["local_remote_source_bytes_equal"]' in fit_source
    for relative, digest in c['SOURCES'].items():
        path = Path('/data/gb/GOLA') / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        source_hashes[str(path)] = digest
for relative, digest in [('research/train_recoverability.py', '4b1327a541d426aeade308e4fc3d7c8e3110e35dae4612c11b93029783fd4e93'),
                         ('research/recoverability_modules.py', '371269db053cdbd38b510557e213f1acfb9c6427e6eabd2df5685c59bdfbb158')]:
    path = Path('/data/gb/GOLA') / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    source_hashes[str(path)] = digest
checks['actual_private_main_and_review_source_hashes_preserved'] = True
checks['both_runner_label_root_and_ordered_four_reference_contracts'] = True
checks['data_prefix25_and_training_parent4_distinct'] = True

def extracted(name, wanted, namespace):
    nodes = [node for node in trees[name].body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name + ':isolated-pure-functions', 'exec'), namespace)
    return namespace

for mode in ('oracle', 'selector'):
    class Recorder:
        def evaluate(self, *args, **kwargs):
            return args, kwargs
    ns = extracted(f'{mode}_actual_full_fit_cpu_audit.py', {'replay', 'metric_errors'}, {'trainer': Recorder()})
    args, kwargs = ns['replay']('SYNTHETIC_MODEL_TOKEN', 'SYNTHETIC_DATA_TOKEN')
    assert args == ('SYNTHETIC_MODEL_TOKEN', 'SYNTHETIC_DATA_TOKEN', 384, .03)
    assert kwargs == dict(details=True, search_supervision=mode, action_ranking='budgeted',
                          write_pair_calibration=False, write_verification='action')
    assert ns['metric_errors']({'utility': .5}, {'utility': .5}) == {'utility': 0.}
    try:
        ns['metric_errors']({'utility': .5}, {'utility': .500003})
    except AssertionError:
        pass
    else:
        raise AssertionError('metric mismatch unexpectedly accepted')
assert math.ceil(902 / 384) * 60 == 180
for utilities, expected in [([.8, .8, .7], 0), ([.7, .8, .8], 1)]:
    assert max(range(len(utilities)), key=utilities.__getitem__) == expected
checks['CPU_replay_call_contract_no_neural_execution_and_metric_tolerance'] = True
checks['180_update_arithmetic_and_strict_first_max_best_selection'] = True

ns = extracted('oracle_actual_full98_cpu_audit.py', {'xywh', 'overlap', 'known', 'failure_events', 'mechanism'}, {'np': np})
assert np.allclose(ns['overlap']([[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]],
                               [[0, 0, 2, 2], [1, 0, 2, 2], [3, 0, 2, 2]]), [1, 1/3, 0], atol=0, rtol=1e-15)
assert ns['known'](np.array([[0, 0, 2, 2], [0, 0, 0, 2], [np.nan, 0, 2, 2]])).tolist() == [True, False, False]
def event(start, confirm, recovery=None):
    return {'start_frame_zero_based': start, 'confirmed_frame_zero_based': confirm,
            'recovered': recovery is not None, 'recovery_start_frame_zero_based': recovery,
            'delay_frames': recovery-start if recovery is not None else None}
event_cases = [
    ([0, .1, .19, .199, .5, .6, .7], [], [event(1, 3, 4)]),
    ([0, .1, .1, .1, .5, .5, .9, .5, .5, .5], [6], [event(1, 3, 7)]),
    ([0, .1, .1, .1, .1, .1, .1], [3], [event(4, 6)]),
    ([0, .2, .2, .2], [], []), ([0, .1, .1], [], []),
    ([1, .1, .1, .1, .5, .5, .5, .1, .1, .1], [], [event(1, 3, 4), event(7, 9)]),
]
for q, invalid, expected in event_cases:
    valid = np.ones(len(q), dtype=bool)
    valid[invalid] = False
    assert ns['failure_events'](np.array(q), valid) == expected
checks['independent_IoU_and_six_failure_recovery_threshold_unknownGT_fixtures'] = True

n = 6
good, bad = np.array([0., 0., 10., 10.]), np.array([20., 20., 30., 30.])
gt = np.tile(good, (n+1, 1)); gt[5] = 0
boxes = np.tile(bad, (n, 7, 5, 1)); boxes[[1, 3, 5], 0, 0] = good; boxes[[0, 3, 4], 1, 0] = good
valid = np.zeros((n, 7, 5), dtype=bool); valid[:, 0, 0] = True; valid[1, 0, 1] = True; valid[[0, 3, 4], 1, 0] = True
choice = np.array([5, 1, 0, 0, 5, 0])
writes = np.array([True, True, False, False, True, False])
selected = boxes.reshape(n, 35, 4)[np.arange(n), choice]
source_boxes = np.stack([good, bad, bad, bad, good, good])
prior_boxes = np.concatenate([good[None], source_boxes[:-1]])
rates = np.zeros((n, 4, 2)); rates[writes, 0] = .5
raw = np.full((n, 7, 5), .9); raw[5] = .8
t = dict(boxes_xyxy=boxes, valid=valid, choice=choice, original_choice=np.zeros(n, dtype=int),
         region=choice//5, extra_executed=np.array([True, False, False, True, True, False]),
         search_requested=np.array([True, False, True, True, True, False]), searched_region=np.array([1, 0, 2, 1, 1, 0]),
         template_updated=writes, pause=np.array([False, False, True, True, False, False]), raw_score=raw,
         prior_template_frame=np.array([0, 1, 2, 2, 2, 5]), template_source_frame=np.array([1, 2, 2, 2, 5, 5]),
         prior_template_box=prior_boxes, template_source_box=source_boxes, memory_write_rates=rates,
         memory_target_commit=np.repeat(writes[:, None], 2, axis=1), requested_extra_area=np.array([100., 0, 200, 100, 100, 0]))
pred = np.concatenate([good[None], ns['xywh'](selected)])
expected_counts = dict(tracking_frames=6, valid_tracking_frames=5, candidate_evaluations=10,
                      local_candidate_recalled_frames=3, budget_candidate_recalled_frames=4, local_missing_frames=2,
                      missing_correct_candidates_reintroduced=1, reintroduced_candidates_selected_correctly=1,
                      c1_same_state_failed_frames=2, c1_same_state_correct_frames=3, selected_failed_frames=2,
                      changed_candidate_indices=3, valid_changed_candidate_indices=2, same_state_rescues=1, same_state_harms=1,
                      same_state_iou_improvements=1, same_state_iou_degradations=1, requested_extra_searches=4,
                      extra_visual_forwards=3, searches_without_new_correct_candidate=1, requested_extra_area_sum_pixels_squared=500.,
                      template_updates=3, known_template_updates=2, wrong_template_updates_localization_proxy=1,
                      paused_query_writes=2, known_paused_query_writes=2, wrong_query_writes_prevented_localization_proxy=1,
                      correct_query_writes_prevented=1, known_active_template_frames=4,
                      wrong_active_template_frames_localization_proxy=2)
counts, supplemental = ns['mechanism'](t, pred, gt)
assert counts == expected_counts, (counts, expected_counts)
assert supplemental == dict(chosen_extra_region_frames=2, chosen_extra_gt_correct_frames=1,
                            selected_regular_actions=4, selected_pause_actions=2, skipped_empty_extra_regions=1)
for key, index, value in [('template_source_frame', 2, 3), ('prior_template_frame', 3, 1)]:
    broken = {k: v.copy() for k, v in t.items()}; broken[key][index] = value
    try:
        ns['mechanism'](broken, pred, gt)
    except AssertionError:
        pass
    else:
        raise AssertionError('broken actual template lineage accepted')
checks['six_frame_actual_action_counters_and_template_lineage'] = {'counters_checked': len(expected_counts), 'supplemental_checked': 5,
                                                                'broken_lineage_cases_rejected': 2}

# The full closure program runs only against synthetic files in a temporary directory.
# Three top-level filesystem constants are redirected; no production path is executed.
closure_results = {}
with tempfile.TemporaryDirectory(prefix='current_policy_completion_CPU_FIXTURE_ONLY_') as temporary:
    sandbox = Path(temporary); setup = sandbox/'setup'; outputs = sandbox/'outputs'; setup.mkdir(); outputs.mkdir()
    names = [f'CPU_FIXTURE_ONLY_{i:03d}' for i in range(98)]
    split = sandbox/'split.json'; split.write_text(json.dumps({'validation': names}))
    closure_tree = copy.deepcopy(trees['four_factor_closure.py'])
    for node in closure_tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ('SETUP', 'OUTPUTS', 'SPLIT'):
                node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                                      args=[ast.Constant(str({'SETUP': setup, 'OUTPUTS': outputs, 'SPLIT': split}[key]))], keywords=[])
    closure_program = compile(ast.fix_missing_locations(closure_tree), 'four_factor_closure.py:CPU_FIXTURE_ONLY', 'exec')
    old = 'write_pair_reference_own_b384'
    labels = [f'{stem}_{policy}_b384' for stem in ('current_policy_fit', 'current_policy_selector_fit') for policy in ('c1', 'own')]
    base = np.linspace(.65, .75, 98)
    vectors = {old: base, 'baseline': base-.03, 'c1': base-.04}
    destination = setup/'current_policy_four_fit_factor_closure_20261004.json'
    def prepare(positive=True):
        for index, label in enumerate(labels):
            vectors[label] = base + ((index+1)*.005 if positive else -.025+index*.005) + np.sin(np.arange(98))*.001
        for mode, stem in [('oracle', 'current_policy_fit'), ('selector', 'current_policy_selector_fit')]:
            fit = dict(status='PASS', both_teacher_arms_passed=True, parent_epoch=4, epochs_per_arm=60, optimizer_steps_per_arm=180,
                       arms={policy: {'best_epoch': 3} for policy in ('c1', 'own')})
            audit = dict(status='PASS', scope={'frames_per_variant': 49418, 'formal_native_test_datasets_read': False}, variants={})
            for variant in [old, 'baseline', 'c1']+[f'{stem}_{policy}_b384' for policy in ('c1', 'own')]:
                audit['variants'][variant] = dict(per_sequence_metrics_exact_to_report_csv=True,
                    sequence_mean_iou=float(vectors[variant].mean()), frame_mean_iou=float(vectors[variant].mean()),
                    failure_event_count=0, failed_frames=0, median_recovery_delay_frames=None,
                    independently_recomputed_mechanism_counters={}, efficiency={})
            (setup/f'{stem}_actual_full_fit_cpu_audit_20261004.json').write_text(json.dumps(fit))
            (setup/f'{stem}_actual_full98_cpu_audit_20261004.json').write_text(json.dumps(audit))
            for policy in ('c1', 'own'):
                label = f'{stem}_{policy}_b384'; root = outputs/f'recoverability_{label}_full_20261004'
                (root/'report').mkdir(parents=True, exist_ok=True); (root/'job_completed.txt').write_text('CPU_FIXTURE_ONLY')
                with (root/'report/per_sequence.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream); writer.writerow(['variant', 'sequence', 'mean_valid_iou'])
                    for variant in [label, old, 'baseline', 'c1']:
                        writer.writerows((variant, name, float(value)) for name, value in zip(names, vectors[variant]))
    def execute():
        with contextlib.redirect_stdout(io.StringIO()):
            exec(closure_program, {'__name__': '__CPU_FIXTURE_ONLY__'})
        return json.loads(destination.read_text())
    prepare(True); result = execute()
    assert result['best_current_cell'] == labels[3]
    assert result['native_evaluation_candidate'].endswith(f'recoverability_{labels[3]}_full_20261004/best.pth')
    assert not result['global_checkpoint_promoted'] and not result['allfive_plus2_native_goal_proven']
    assert len(result['paired']) == 16
    samples = np.random.default_rng(42).integers(0, 98, size=(5000, 98))
    for name, actual in result['paired'].items():
        label, reference = name.split('_vs_'); differences = (vectors[label]-vectors[reference])*100
        manual_samples = [sum(float(differences[i]) for i in indices)/98 for indices in samples]
        assert abs(actual['delta_sequence_iou_percentage_points'] - sum(map(float, differences))/98) < 1e-12
        assert np.allclose(actual['paired_sequence_95_ci'], np.percentile(manual_samples, [2.5, 97.5]), atol=1e-12, rtol=0)
    closure_results['positive_candidate_and_all16_manual5000_bootstrap_pairs'] = True
    prepare(False); assert execute()['native_evaluation_candidate'] is None
    closure_results['all_four_below_old4_no_candidate'] = True
    prepare(True)
    audit_path = setup/'current_policy_fit_actual_full98_cpu_audit_20261004.json'
    broken = json.loads(audit_path.read_text()); broken['variants'][labels[0]]['sequence_mean_iou'] += .01
    audit_path.write_text(json.dumps(broken))
    try:
        execute()
    except AssertionError:
        closure_results['mismatched_audited_mean_rejected'] = True
    else:
        raise AssertionError('mismatched audited mean accepted')
    prepare(True)
    fit_path = setup/'current_policy_fit_actual_full_fit_cpu_audit_20261004.json'
    broken = json.loads(fit_path.read_text()); broken['status'] = 'PENDING'
    fit_path.write_text(json.dumps(broken))
    try:
        execute()
    except AssertionError:
        closure_results['pending_fit_acceptance_rejected'] = True
    else:
        raise AssertionError('pending fit acceptance accepted')
checks['synthetic_files_only_closure'] = closure_results
assert 'torch' not in sys.modules
receipt = {'status': 'PASS_SOURCE_AND_BOUNDED_CPU_FIXTURES_ONLY',
           'helper_sha256': {name: hashlib.sha256(source.encode()).hexdigest() for name, source in helpers.items()},
           'checks': checks, 'actual_remote_source_sha256': source_hashes,
           'scope': 'Static source review and synthetic CPU math/files only. No real completion audit, M0, full fit, full98 or native acceptance.',
           'execution': dict(remote_python=sys.executable, torch_imported=False, NN_forward=0, optimizer_steps=0,
                             GPU_queries=0, power_temperature_queries=0, process_signals=0, weights_read=False,
                             weights_modified=False, production_sources_modified=False, actual_completion_audits_executed=0,
                             runtime_seconds=round(time.perf_counter()-start, 3))}
print(json.dumps(receipt, indent=2, allow_nan=False))
'''

result = subprocess.run(
    ["ssh", "2027", "env CUDA_VISIBLE_DEVICES= /data/gb/envs/gola/bin/python -"],
    input="payload = " + repr(payload) + "\n" + REMOTE,
    text=True, capture_output=True,
)
assert result.returncode == 0, result.stderr
receipt = json.loads(result.stdout)
assert receipt["status"] == "PASS_SOURCE_AND_BOUNDED_CPU_FIXTURES_ONLY"
(HERE / "completion_audit_cpu_fixture_receipt.json").write_text(
    json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2, allow_nan=False))
