"""Merge both completed core benchmarks and the matched state-repair controls."""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {'lasher': (245, 220703, 19, ('PR', 'NPR', 'SR')),
            'rgbt234': (234, 116649, 12, ('MPR', 'MSR'))}


def load(path):
    return json.loads(Path(path).read_text())


def native_report(path, dataset, variants):
    report = load(path)
    sequences, frames, attributes, metrics = EXPECTED[dataset]
    assert report['dataset'] == dataset and report['all_actual_ground_truth_verified']
    assert (report['sequences'], report['frames']) == (sequences, frames)
    assert list(report['variants']) == variants
    for variant in variants:
        entry = report['variants'][variant]
        assert set(entry['overall_metrics_percent']) == set(metrics)
        assert len(entry['attributes']) == attributes
        assert all(math.isfinite(value) for value in entry['overall_metrics_percent'].values())
    with path.with_name('per_sequence.csv').open(newline='') as source:
        rows = list(csv.DictReader(source))
    names = [{row['sequence'] for row in rows if row['variant'] == variant} for variant in variants]
    assert len(rows) == len(variants) * sequences and all(len(group) == sequences for group in names)
    assert all(group == names[0] for group in names)
    return report, rows


def uncertainty(path, dataset, variants, native):
    report = load(path)
    assert set(report['datasets']) == {dataset} and report['args']['iterations'] == 5000
    entry = report['datasets'][dataset]
    assert entry['compared_variants'] == variants and entry['sequences'] == EXPECTED[dataset][0]
    assert set(entry['metrics']) == set(EXPECTED[dataset][3])
    for metric, result in entry['metrics'].items():
        delta = native['variants'][variants[1]]['overall_metrics_percent'][metric] - native['variants'][variants[0]]['overall_metrics_percent'][metric]
        assert math.isclose(delta, result['mean_delta_percentage_points'], rel_tol=0, abs_tol=1e-8)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-root', required=True)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--parameter-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, training = Path(args.full_root), Path(args.training_run)
    assert (training / 'training_and_audit_completed.txt').is_file()
    audit = load(training / 'reload_and_calibration.json')
    assert audit['completed'] and audit['initial_last_c1_state_exact'] and audit['all_epochs_retained']
    report = {'completed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'official_tracking_accuracy': True, 'datasets': {}, 'training_audit': audit,
              'training_config': load(training / 'config.json'), 'training_completion': load(training / 'completion.json'),
              'training_epoch_metrics': load(training / 'metrics.json'),
              'parameter_counts': load(args.parameter_report),
              'train_search_audit': load(training / 'train_candidate_audit_v2.json'),
              'acceptance': {'required_gain_percentage_points_per_overall_metric': 2., 'metrics': []},
              'remaining_experiments': ['Eight A/B/C combinations have not been evaluated.',
                                        'Memory-budget and matched-search-budget curves have not been evaluated.',
                                        'Repeated training-seed uncertainty has not been evaluated.',
                                        'FLOPs have not been measured; concurrent I/O prevents a fair speedup claim.']}
    parameters = report['parameter_counts']
    assert report['train_search_audit']['completed'] and report['train_search_audit']['checkpoint_epoch'] == 30
    modules = report['train_search_audit']['training_parameters']
    report['new_module_parameters'] = modules
    assert parameters['actual_pretrained_model_loaded'] and parameters['C1_parameters'] == modules['frozen_C1']
    report['current_system_parameter_count'] = (parameters['base_parameters_after_adapter_loading']
                                                + sum(modules.values()))
    report['parameter_count_scope'] = 'Sum of recorded actual base and current A/B/C/frozen-C1 module counts; cached C3 prototype is excluded.'
    report['baseline_and_c1_uncertainty'] = load(root.parent / 'full_reports_v2/paired_sequence_bootstrap.json')
    sequence_rows, attribute_rows = [], []
    for dataset, (_, _, _, metrics) in EXPECTED.items():
        dataset_reports, diagnostics, intervals = {}, {}, {}
        for variant in ('abc', 'abc_box_only'):
            run = root / f'{dataset}_{variant}'
            assert (run / 'full_evaluation_completed.txt').is_file()
            native, rows = native_report(run / 'report/full_report.json', dataset, ['baseline', variant])
            native['original_pending_notes_superseded_by_abc_diagnostics'] = native['mechanism_measurements_pending']
            native['mechanism_measurements_pending'] = report['remaining_experiments']
            dataset_reports[variant] = native
            intervals[variant] = uncertainty(run / 'report/paired_sequence_bootstrap.json', dataset, ['baseline', variant], native)
            branch = load(run / 'report/abc_diagnostics.json')
            assert branch['dataset'] == dataset and branch['actual_data_root'] == native['actual_data_root']
            assert set(branch['variants']) == {variant}
            assert branch['variants'][variant]['inference_config'] == native['variants'][variant]['original_inference_config']
            config = branch['variants'][variant]['inference_config']
            assert config['model'] == str(training / 'last.pth') and config['head_epoch'] == 30
            assert not config['smoke_only'] and config['validation_split'] is None
            assert (config['sequence_offset'], config['limit_sequences'], config['max_frames']) == (0, 0, 0)
            assert (config['branches'], config['window'], config['switch_patience'], config['seed']) == (3, 5, 2, 42)
            diagnostics[variant] = branch
            for row in rows:
                if variant == 'abc' or row['variant'] != 'baseline':
                    sequence_rows.append({'dataset': dataset, **row})
            for name, entry in native['variants'].items():
                if variant == 'abc' or name != 'baseline':
                    for attribute, values in entry['attributes'].items():
                        attribute_rows.append({'dataset': dataset, 'variant': name, 'attribute': attribute, **values})
        assert dataset_reports['abc']['variants']['baseline'] == dataset_reports['abc_box_only']['variants']['baseline']
        state = root / 'state_compare' / dataset
        assert (state / 'state_comparison_completed.txt').is_file()
        control, _ = native_report(state / 'full_report.json', dataset, ['abc_box_only', 'abc'])
        control_ci = uncertainty(state / 'paired_sequence_bootstrap.json', dataset, ['abc_box_only', 'abc'], control)
        control['original_pending_notes_superseded_by_abc_diagnostics'] = control['mechanism_measurements_pending']
        control['mechanism_measurements_pending'] = report['remaining_experiments']
        for variant in ('abc', 'abc_box_only'):
            assert control['variants'][variant] == dataset_reports[variant]['variants'][variant]
        reference = load(root.parent / 'full_reports_v2' / dataset / 'complete_report.json')
        assert reference['all_actual_ground_truth_verified'] and reference['dataset'] == dataset
        assert (reference['sequences'], reference['frames']) == EXPECTED[dataset][:2]
        assert set(reference['variants']) == {'baseline', 'c1'}
        for key in ('overall_metrics_percent', 'attributes', 'mean_curves'):
            assert reference['variants']['baseline'][key] == dataset_reports['abc']['variants']['baseline'][key]
        report['datasets'][dataset] = {'native': dataset_reports, 'branch_diagnostics': diagnostics,
                                       'paired_uncertainty_against_baseline': intervals,
                                       'state_repair_control': control, 'state_repair_uncertainty': control_ci,
                                       'original_full_baseline_and_c1_reference': reference,
                                       'sources': {variant: str(root / f'{dataset}_{variant}' / 'report')
                                                   for variant in ('abc', 'abc_box_only')}}
        baseline = dataset_reports['abc']['variants']['baseline']['overall_metrics_percent']
        main_scores = dataset_reports['abc']['variants']['abc']['overall_metrics_percent']
        for metric in metrics:
            delta = main_scores[metric] - baseline[metric]
            report['acceptance']['metrics'].append({'dataset': dataset, 'metric': metric,
                                                    'baseline_percent': baseline[metric], 'abc_percent': main_scores[metric],
                                                    'delta_percentage_points': delta, 'meets_plus_two': delta >= 2.})
    assert len(report['acceptance']['metrics']) == 5 and len(sequence_rows) == 479 * 3
    report['acceptance']['all_five_overall_meet_plus_two'] = all(row['meets_plus_two'] for row in report['acceptance']['metrics'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'complete_core_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for filename, rows in (('acceptance.csv', report['acceptance']['metrics']),
                           ('per_sequence.csv', sequence_rows), ('attributes.csv', attribute_rows)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / filename).open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output / 'complete_core_report_completed.txt').write_text(datetime.now(timezone.utc).isoformat())
    print(json.dumps({'completed': True, 'acceptance': report['acceptance'], 'output': str(output)}))


if __name__ == '__main__':
    main()
