"""Collect both completed native benchmarks and all four training histories, without NN work."""
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile

sys.stdout.reconfigure(encoding='utf-8')
work = pathlib.Path(r'C:\Users\gb\.codex_tmp\gola_setup_20261002')
root = work / 'GOLA-source'
sources = root / 'refine-logs/runs/recoverability_train_events'
target = sources / 'native_complete'
assert not target.exists()


def remote(code):
    result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                            input=code, text=True, encoding='utf-8', capture_output=True, check=True)
    return json.loads(result.stdout)


preflight = remote("""
import json,pathlib
base=pathlib.Path('/data/gb/outputs/recoverability_train_events_native_full_20261005')
report=json.loads((base/'complete_report/complete_core_report.json').read_text())
state=json.loads(pathlib.Path('/data/gb/setup/train_events_complete_evaluation_progress_20261005.json').read_text())
assert report['completed'] and report['official_tracking_accuracy'] and len(report['acceptance']['metrics'])==5
assert state['stage']=='COMPLETE420_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY'
assert state['acceptance']==report['acceptance']
assert all(row['optimizer_steps']==420 for row in report['training_CPU_acceptance']['arms'].values())
assert report['selected_checkpoint']['both_datasets_same_fixed_checkpoint']
assert report['TRAIN_transition_716_GT_CPU']['new_clips']==716
assert (base/'complete_report/complete_core_report_completed.txt').is_file()
print(json.dumps({'status':'PASS','five_metrics':report['acceptance']['metrics']}))
""")
assert preflight['status'] == 'PASS'
review = json.loads((sources / 'complete_delivery_source_review.json').read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and review['runtime_attested'] is False
plotted = remote("import os\nos.environ['CUDA_VISIBLE_DEVICES']=''\n" + (sources / 'plot_completed_training_history_cpu.py').read_text(encoding='utf-8'))
assert plotted['completed'] and plotted['actual_optimizer_steps_per_arm'] == 420
exported = remote((sources / 'export_completed_results_cpu.py').read_text(encoding='utf-8'))
assert exported['status'] == 'PASS' and exported['five_metrics'] == preflight['five_metrics']
packet = work / 'train_events_complete_results_20261005.tar.gz'
subprocess.run(['scp', '2027:/data/gb/setup/' + packet.name, str(packet)], check=True, capture_output=True, text=True)
target.mkdir()
with tarfile.open(packet) as archive:
    manifest = json.load(archive.extractfile('export_manifest.json'))
    assert manifest['status'] == 'PASS' and manifest['actual_two_native_GT_audits_passed']
    assert sorted(member.name for member in archive.getmembers()) == sorted([row['path'] for row in manifest['files']] + ['export_manifest.json'])
    archive.extractall(target)
for row in manifest['files']:
    path = target / row['path']
    assert path.stat().st_size == row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
report = json.loads((target / 'complete_report/complete_core_report.json').read_text(encoding='utf-8'))
assert report['completed'] and report['official_tracking_accuracy']
assert report['acceptance']['metrics'] == exported['five_metrics']
training = report['training_CPU_acceptance']
assert training['status'] == 'PASS' and training['all_four_ABC_fits_passed']
assert {name: row['optimizer_steps'] for name, row in training['arms'].items()} == {
    'base_pairwise': 420, 'aug_pairwise': 420, 'base_budgeted': 420, 'aug_budgeted': 420}
assert {name: (row['epochs'], row['train_queries']) for name, row in training['arms'].items()} == {
    'base_pairwise': (84, 1762), 'aug_pairwise': (60, 2478),
    'base_budgeted': (84, 1762), 'aug_budgeted': (60, 2478)}
assert report['selected_checkpoint']['both_datasets_same_fixed_checkpoint']
assert [(name, value['independent_native_CPU']['sequences'], value['independent_native_CPU']['frames'])
        for name, value in report['datasets'].items()] == [('lasher', 245, 220703), ('rgbt234', 234, 116649)]
assert all(value['independent_native_CPU']['status'] == 'PASS' for value in report['datasets'].values())
summary = {'status': 'PASS', 'file_count': manifest['file_count'], 'total_bytes': manifest['total_bytes'],
           'five_metrics': exported['five_metrics'], 'goal_result': report['goal_result'], 'source': str(target),
           'full_report_sha256': hashlib.sha256((target / 'complete_report/complete_core_report.json').read_bytes()).hexdigest(),
           'weights_retired': False, 'neural_forward_calls': 0, 'optimizer_steps': 0}
(sources / 'actual_complete_report_local_intake.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False), flush=True)
