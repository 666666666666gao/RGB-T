"""Consolidate the two actually completed native benchmarks and all five goal metrics."""
import csv
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.merge_abc_complete import EXPECTED, native_report, uncertainty

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
setup = Path('/data/gb/setup')
base = Path('/data/gb/outputs/recoverability_best_native_policy_native_full_20261005')
read = lambda path: json.loads(Path(path).read_text())
review = read(setup / 'best_native_policy_goal_report_source_review_20261005.json')
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
gate = read('/data/gb/outputs/recoverability_best_native_policy_merged_20261005/best_native_policy_cache_cpu_gate.json')
assert gate['status'] == 'PASS' and gate['matched_partitions'] == {'train': 1762, 'validation': 196} and gate['all881_98_video_names_exact']
training = read(setup / 'best_native_policy_full_fit_cpu_acceptance_20261005.json')
assert training['status'] == 'PASS' and training['all_four_full_fits_CPU_passed']
assert training['epochs_per_arm'] == 60 and training['optimizer_steps_per_arm'] == 300 and training['batch_size'] == 384
selection = read(setup / 'best_native_policy_native_selection_20261005.json')
assert selection['status'] == 'PASS' and selection['both_datasets_same_fixed_checkpoint']
checkpoint = Path(selection['checkpoint'])
selected_arm = next(name for name, row in training['arms'].items() if Path(row['root']) == checkpoint.parent)
chosen = training['arms'][selected_arm]
weight = chosen['checkpoints'][checkpoint.name]
assert weight['epoch'] == selection['selected_best_epoch'] and weight['epoch'] > 0
assert weight['ABC_changed'] == {'A': True, 'B': True, 'C': True}
assert weight['all_C1_tensors_exact'] and weight['cached_CPU_replay_pass']
assert checkpoint.is_file()
checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
variants = ['baseline', 'c1', 'best_native_policy_complete']
datasets, acceptance, sequences, attributes = {}, [], [], []
for dataset, (count, frames, _, metrics) in EXPECTED.items():
    root = base / dataset
    assert (root / 'report_completed.txt').is_file()
    actual = read(root / 'independent_native_cpu_acceptance.json')
    assert actual['status'] == 'PASS' and actual['dataset'] == dataset
    assert actual['sequences'] == count and actual['frames'] == frames and actual['same_checkpoint'] == str(checkpoint)
    assert actual['all_actual_GT_sequence_metrics_attributes_and_curves_exact'] and actual['score_not_another_models_output']
    report, rows = native_report(root / 'core_report/full_report.json', dataset, variants)
    assert actual['native_overall_percent'] == {variant: report['variants'][variant]['overall_metrics_percent'] for variant in variants}
    config = report['variants']['best_native_policy_complete']['original_inference_config']
    assert config['model'] == str(checkpoint) and config['head_epoch'] == selection['selected_best_epoch']
    assert config['write_verification'] == 'action' and not config['zero_init'] and config['validation_split'] is None
    assert config['max_frames'] == config['limit_sequences'] == config['sequence_offset'] == 0
    paired = {}
    for reference in ('baseline', 'c1'):
        folder = root / (reference + '_paired_report')
        native, _ = native_report(folder / 'full_report.json', dataset, [reference, variants[-1]])
        for variant in (reference, variants[-1]):
            for field in ('overall_metrics_percent', 'attributes', 'mean_curves'):
                assert native['variants'][variant][field] == report['variants'][variant][field]
        paired[reference] = uncertainty(folder / 'paired_bootstrap.json', dataset, [reference, variants[-1]], native)
        assert paired[reference]['args']['seed'] == 42
        assert paired[reference]['datasets'][dataset]['metrics'] == actual['paired_5000_seed42_bootstrap_exact'][reference]
        assert all((folder / name).is_file() and (folder / name).stat().st_size > 0 for name in
                   ('official_curves.png', 'official_curves.pdf', 'attribute_deltas.png', 'attribute_deltas.pdf'))
    assert (root / 'mechanism_report/recoverability_metrics_completed.txt').is_file()
    mechanism = read(root / 'mechanism_report/full_recoverability_report.json')
    assert mechanism['completed'] and mechanism['sequences'] == count
    datasets[dataset] = {'native': report, 'paired': paired, 'independent_native_CPU': actual,
                         'mechanism': mechanism, 'source_root': str(root)}
    sequences.extend({'dataset': dataset, **row} for row in rows)
    for variant in variants:
        for name, values in report['variants'][variant]['attributes'].items():
            attributes.append({'dataset': dataset, 'variant': variant, 'attribute': name, **values})
    for metric in metrics:
        baseline, c1, model = [report['variants'][variant]['overall_metrics_percent'][metric] for variant in variants]
        result = actual['paired_5000_seed42_bootstrap_exact']['baseline'][metric]
        interval = result['percentile_95_interval_percentage_points']
        acceptance.append({'dataset': dataset, 'metric': metric, 'baseline_percent': baseline, 'c1_percent': c1,
                           'ABC_percent': model, 'delta_vs_baseline_percentage_points': model - baseline,
                           'delta_vs_c1_percentage_points': model - c1, 'target_percent': baseline + 2,
                           'meets_plus_two': model - baseline >= 2,
                           'paired_95_lower_percentage_points': interval[0], 'paired_95_upper_percentage_points': interval[1]})
assert len(acceptance) == 5 and len(sequences) == 479 * 3 and len(attributes) == (19 + 12) * 3
passed = all(row['meets_plus_two'] for row in acceptance)
record = {'completed': True, 'official_tracking_accuracy': True, 'created_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'selected_checkpoint': selection, 'training_CPU_acceptance': training, 'all881_98_video_coverage_CPU': gate,
          'datasets': datasets, 'acceptance': {'required_gain_percentage_points': 2, 'metrics': acceptance,
                                              'all_five_overall_meet_plus_two': passed},
          'goal_result': 'ALL_FIVE_PLUS_TWO_VERIFIED_REQUIRES_ROOT_COMPLETION_AUDIT' if passed else 'ACTIVE_UNMET',
          'scope': 'Frozen complete pretrained GOLA-B and C1; all ABC trained60epochs/300updates on1762 TRAIN states and evaluated on196 developerVAL states, two per every881/98 video, queries<=1024. Positive checkpoint selected by full98 continuous developer videos, not native tests. No all-frame/end-to-end or multi-seed stability claim.',
          'selected_weight_provenance': 'Positive-epoch weights learned during this completed60epoch/300update training round.',
          'selected_checkpoint_sha256_at_report_export': checkpoint_sha256,
          'limitations': ['Checkpoint selected on reused full98 continuous developer video IoU; native outcomes are separate evidence, but past native results have already informed development.',
                         'Bootstrap is fixed-checkpoint sequence uncertainty; independent multiple training seeds are not required.',
                         'Concurrent four-GPU I/O latency is descriptive, not an isolated algorithm speedup measurement.',
                         'Detailed mechanism counts come from the original collector; independent CPU replay covers native metrics, attributes, curves and paired intervals.',
                         'Eight A/B/C combinations, budget curves, isolated efficiency/FLOPs and a second baseline are not completed by this report.'],
          'execution': {'GPU_queries': 0, 'neural_forward_calls': 0, 'optimizer_steps': 0, 'system_goal_status_changed': False}}
out = base / 'complete_report'
assert not out.exists()
out.mkdir()
(out / 'complete_core_report.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
for name, rows in (('acceptance.csv', acceptance), ('per_sequence.csv', sequences), ('attributes.csv', attributes)):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
(out / 'complete_core_report_completed.txt').write_text(record['created_at_cst'] + '\n')
print(json.dumps({'completed': True, 'acceptance': record['acceptance'], 'output': str(out)}))
