"""Copy the actually completed native evidence; perform no inference or scoring."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

setup = Path('/data/gb/setup')
base = Path('/data/gb/outputs/recoverability_train_events_native_full_20261005')
target = Path('/data/gb/GOLA/refine-logs/runs/recoverability_train_events/native_complete')
read = lambda p: json.loads(p.read_text())
report = read(base / 'complete_report/complete_core_report.json')
assert report['completed'] and report['official_tracking_accuracy']
assert len(report['acceptance']['metrics']) == 5
assert (base / 'complete_report/complete_core_report_completed.txt').is_file()
continuation = read(setup / 'train_events_complete_evaluation_progress_20261005.json')
assert continuation['stage'] == 'COMPLETE420_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY'
assert continuation['acceptance'] == report['acceptance']
assert not target.exists()
target.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
entries = []


def copy(source, relative):
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert sha(source) == sha(destination)
    entries.append({'path': relative, 'source': str(source),
                    'sha256': sha(destination), 'bytes': destination.stat().st_size})


for path in sorted((base / 'complete_report').iterdir()):
    assert path.is_file() and path.suffix in ('.json', '.csv', '.txt')
    copy(path, 'complete_report/' + path.name)
for dataset in ('lasher', 'rgbt234'):
    root = base / dataset
    actual = read(root / 'independent_native_cpu_acceptance.json')
    assert actual['status'] == 'PASS'
    assert actual['all_actual_GT_sequence_metrics_attributes_and_curves_exact']
    for name in ('full_inference_merge_acceptance.json',
                 'independent_native_cpu_acceptance.json', 'report_completed.txt'):
        copy(root / name, dataset + '/' + name)
    for folder in ('core_report', 'baseline_paired_report', 'c1_paired_report', 'mechanism_report'):
        for path in sorted((root / folder).rglob('*')):
            if path.is_file():
                assert path.suffix in ('.json', '.csv', '.txt', '.png', '.pdf')
                copy(path, dataset + '/' + str(path.relative_to(root)))
    for name in ('inference_config.json', 'inference_completion.json', 'progress.jsonl'):
        copy(root / 'predictions' / name, dataset + '/inference/' + name)
for name in ('train_events_native_selection_20261005.json',
             'train_events_full_fit_cpu_acceptance_20261005.json',
             'train_events_full_cpu_acceptance_20261005.json',
             'train_events_m0_fit_cpu_acceptance_20261005.json',
             'train_events_complete_fit_progress_20261005.json',
             'train_events_complete_evaluation_progress_20261005.json',
             'train_events_complete_fit_controller_20261005.log',
             'train_events_complete_evaluation_controller_20261005.log',
             'train_events_complete_fit_cpu_audit_20261005.log'):
    copy(setup / name, 'runtime/' + name)
for arm, info in report['training_CPU_acceptance']['arms'].items():
    for name in ('config.json', 'completion.json', 'metrics.json', 'train.jsonl', 'best_validation.npz'):
        copy(Path(info['output']) / name, 'training/' + arm + '/' + name)
assert 12 <= len(report['selected_checkpoint']['candidates']) <= 16
for row in report['selected_checkpoint']['candidates']:
    assert row['epoch'] > 0
    prefix = 'full98_selection/' + row['arm'] + '/' + Path(row['model']).stem
    prediction = Path(row['predictions'])
    copy(prediction.parent / 'full98_selection_score.json', prefix + '/full98_selection_score.json')
    for name in ('inference_config.json', 'inference_completion.json'):
        copy(prediction / name, prefix + '/' + name)
history = Path('/data/gb/outputs/recoverability_train_events_training_history_20261005')
for name in ('training_history.png', 'training_history.pdf', 'training_history_summary.json'):
    copy(history / name, 'training/' + name)
manifest = {'status': 'PASS', 'file_count': len(entries),
            'total_bytes': sum(row['bytes'] for row in entries), 'files': entries,
            'all_export_bytes_equal_to_actual_outputs': True,
            'no_weights_datasets_or_raw_predictions_exported': True,
            'actual_two_native_GT_audits_passed': True}
(target / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
with tarfile.open(setup / 'train_events_complete_results_20261005.tar.gz', 'w:gz') as archive:
    for row in entries:
        archive.add(target / row['path'], arcname=row['path'])
    archive.add(target / 'export_manifest.json', arcname='export_manifest.json')
print(json.dumps({'status': 'PASS', 'file_count': len(entries),
                  'total_bytes': manifest['total_bytes'],
                  'five_metrics': report['acceptance']['metrics']}))
