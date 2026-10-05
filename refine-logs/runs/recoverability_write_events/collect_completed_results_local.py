"""Collect the actual complete five-metric report, curves and training evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

WORK = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
ROOT = WORK / 'GOLA-source'
SOURCES = ROOT / 'refine-logs/runs/recoverability_write_events'
TARGET = SOURCES / 'native_complete'


def remote(code):
    r = subprocess.run(['ssh', '-T', '2027', 'CUDA_VISIBLE_DEVICES= /data/gb/envs/gola/bin/python -'],
                       input=code, text=True, encoding='utf-8', capture_output=True)
    (SOURCES / 'actual_complete_delivery_last_remote.log').write_text(r.stdout + r.stderr)
    r.check_returncode()
    return json.loads(r.stdout)


def main():
    assert not TARGET.exists()
    review = json.loads((SOURCES / 'complete_delivery_source_review.json').read_text())
    assert review['status'] == 'PASS' and review['runtime_attested'] is False
    checked = remote(r'''
import json,pathlib
base=pathlib.Path('/data/gb/outputs/recoverability_write_events_native_full_20261005')
report=json.loads((base/'complete_report/complete_core_report.json').read_text())
state=json.loads(pathlib.Path('/data/gb/setup/write_events_complete_evaluation_progress_20261005.json').read_text())
assert report['completed'] and report['official_tracking_accuracy'] and len(report['acceptance']['metrics'])==5
assert state['stage']=='COMPLETE480_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY'
assert state['acceptance']==report['acceptance']
assert all(row['epochs']==60 and row['optimizer_steps']==480 and row['train_queries']==2576 for row in report['training_CPU_acceptance']['arms'].values())
assert report['selected_checkpoint']['both_datasets_same_fixed_checkpoint']
assert report['TRAIN_transition_716_GT_CPU']['new_clips']==716 and report['TRAIN_write_episode_98_GT_CPU']['new_queries']==98
assert (base/'complete_report/complete_core_report_completed.txt').is_file()
plot=json.loads(pathlib.Path('/data/gb/outputs/recoverability_write_events_training_history_20261005/training_history_summary.json').read_text())
assert plot['completed'] and all(row['optimizer_steps']==480 for row in plot['arms'].values())
print(json.dumps({'status':'PASS','five_metrics':report['acceptance']['metrics']}))
''')
    assert checked['status'] == 'PASS'
    exported = remote((SOURCES / 'export_completed_results_cpu.py').read_text())
    assert exported['status'] == 'PASS' and exported['five_metrics'] == checked['five_metrics']
    packet = WORK / 'write_events_complete_results_20261005.tar.gz'
    subprocess.run(['scp', '2027:/data/gb/setup/' + packet.name, str(packet)], check=True, capture_output=True)
    TARGET.mkdir()
    with tarfile.open(packet) as archive:
        manifest = json.load(archive.extractfile('export_manifest.json'))
        assert manifest['status'] == 'PASS' and manifest['actual_two_native_GT_audits_passed']
        assert sorted(m.name for m in archive.getmembers()) == sorted([row['path'] for row in manifest['files']] + ['export_manifest.json'])
        archive.extractall(TARGET)
    for row in manifest['files']:
        path = TARGET / row['path']
        assert path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    report = json.loads((TARGET / 'complete_report/complete_core_report.json').read_text())
    assert report['acceptance']['metrics'] == checked['five_metrics']
    training = report['training_CPU_acceptance']
    assert training['status'] == 'PASS' and training['all_four_ABC_fits_passed']
    assert set(training['arms']) == {'pairwise_lr4', 'budgeted_lr4', 'pairwise_lr5', 'budgeted_lr5'}
    assert all((row['epochs'], row['optimizer_steps'], row['train_queries']) == (60, 480, 2576) for row in training['arms'].values())
    assert [(name, value['independent_native_CPU']['sequences'], value['independent_native_CPU']['frames'])
            for name, value in report['datasets'].items()] == [('lasher', 245, 220703), ('rgbt234', 234, 116649)]
    assert all(value['independent_native_CPU']['status'] == 'PASS' for value in report['datasets'].values())
    result = {'status': 'PASS', 'file_count': manifest['file_count'], 'total_bytes': manifest['total_bytes'],
              'five_metrics': checked['five_metrics'], 'goal_result': report['goal_result'],
              'same_fixed_checkpoint': report['selected_checkpoint']['checkpoint'],
              'actual_native_GT_audits_passed': True, 'neural_forward_calls': 0, 'optimizer_steps': 0,
              'model_weights_exported': False}
    (SOURCES / 'actual_complete_report_local_intake.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
