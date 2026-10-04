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
base = Path('/data/gb/outputs/recoverability_full_coverage_native_full_20261004')
read = lambda path: json.loads(Path(path).read_text())
review = read(setup / 'full_coverage_goal_report_source_review_20261004.json')
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
gate = read('/data/gb/outputs/recoverability_full_coverage_merged_20261004/full_coverage_cache_cpu_gate.json')
assert gate['status'] == 'PASS' and gate['matched_partitions'] == {'train': 881, 'validation': 98} and gate['all881_98_video_names_exact']
training = read(setup / 'full_coverage_actual_full_fit_cpu_acceptance_20261004.json')
assert training['status'] == 'PASS' and training['all_four_lr_ranking_arms_passed']
assert training['epochs_per_arm'] == 60 and training['optimizer_steps_per_arm'] == 180 and training['batch_size'] == 416
selection = read(setup / 'full_coverage_native_selection_20261004.json')
assert selection['status'] == 'PASS' and selection['both_datasets_same_fixed_checkpoint']
chosen = training['arms'][selection['selected_arm']]
assert chosen['status'] == 'PASS' and chosen['active_modules'] == ['A', 'B', 'C'] and not chosen['frozen_modules']
assert chosen['best_epoch'] == selection['selected_best_epoch'] and chosen['C1_all8_retained_tensors_exact']
checkpoint = Path(selection['checkpoint'])
assert checkpoint == Path(chosen['root']) / 'best.pth'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selection['checkpoint_sha256'] == chosen['artifact_sha256']['best.pth']
variants = ['baseline', 'c1', 'full_coverage_complete']
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
    config = report['variants']['full_coverage_complete']['original_inference_config']
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
          'scope': 'Frozen complete pretrained GOLA-B and C1; full ABC training runs executed60epochs on one eligible query per every881 TRAIN/98 developerVAL video, history<=1024; retained-weight provenance is separate; no all-frame/end-to-end or multi-seed stability claim.',
          'selected_weight_provenance': ('The full-coverage60epoch/180update training runs were completed, but retained epoch0 is the earlier partial-coverage teacher; the selected weights contain no new learning from this full-coverage round.'
                                        if selection['best0_is_parent_not_new_learning'] else
                                        'Selected positive-epoch weights were learned during this completed full-coverage60epoch/180update training round.'),
          'limitations': ['Checkpoint selected on reused developer VAL98 utility; final native outcomes are separate evidence.',
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
