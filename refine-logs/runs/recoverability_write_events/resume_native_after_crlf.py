"""Resume only the unstarted native phase after the recorded CRLF shell failure."""
import os
import subprocess
from pathlib import Path

import evaluation_pipeline as pipeline


def main():
    assert Path.cwd() == pipeline.ROOT and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    review = pipeline.read(pipeline.FOLDER / 'native_crlf_rescue_source_review.json')
    assert review['status'] == 'PASS' and review['runtime_attested'] is False
    failed = pipeline.read(pipeline.FOLDER / 'actual_native_launch_failure_primary.json')
    assert failed['state']['stage'] == 'BOTH_COMPLETE_NATIVE_BENCHMARKS_RUNNING'
    assert all(not row['live'] and 'invalid option name' in row['log'] for row in failed['children_primary'])
    for pid in [failed['state']['pid']] + [row['pid'] for row in failed['state']['children']]:
        assert not (Path('/proc') / str(pid)).exists()
    selection = pipeline.read(pipeline.SETUP / 'write_events_native_selection_20261005.json')
    assert selection == failed['selection'] and len(selection['candidates']) == 16
    assert selection['selected_best_epoch'] == 4 and selection['both_datasets_same_fixed_checkpoint']
    assert Path(selection['checkpoint']).is_file()
    base = Path('/data/gb/outputs/recoverability_write_events_native_full_20261005')
    assert not base.exists()
    for name in ('run_native_complete.sh', 'report_native_complete.sh'):
        script = pipeline.FOLDER / name
        assert b'\r' not in script.read_bytes()
        subprocess.run(['bash', '-n', str(script)], check=True)
    env = os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(pipeline.ROOT),
                        'LD_LIBRARY_PATH': '/data/gb/envs/gola/lib', 'OMP_NUM_THREADS': '4',
                        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'TORCH_HOME': '/data/gb/cache/torch',
                        'XDG_CACHE_HOME': '/data/gb/cache', 'TMPDIR': '/data/gb/cache/tmp',
                        'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    pipeline.wave([(gpu, ['bash', str(pipeline.FOLDER / 'run_native_complete.sh'), str(gpu)])
                   for gpu in range(4)], 'BOTH_COMPLETE_NATIVE_CRLF_RESUME', env)
    pipeline.record('BOTH_NATIVE_INFERENCE_COMPLETE_ALL_METRICS_SCORING')
    for dataset in ('lasher', 'rgbt234'):
        subprocess.run([pipeline.PYTHON, '-u', str(pipeline.FOLDER / 'merge_native_complete.py'),
                        '--dataset', dataset], cwd=pipeline.ROOT, env=env, check=True)
        subprocess.run(['bash', str(pipeline.FOLDER / 'report_native_complete.sh'), dataset],
                       cwd=pipeline.ROOT, env=env, check=True)
        subprocess.run([pipeline.PYTHON, '-u', str(pipeline.FOLDER / 'audit_native_complete_cpu.py'),
                        '--dataset', dataset], cwd=pipeline.ROOT, env=env, check=True)
    subprocess.run([pipeline.PYTHON, '-u', str(pipeline.FOLDER / 'merge_complete_goal_report.py')],
                   cwd=pipeline.ROOT, env=env, check=True)
    report = base / 'complete_report/complete_core_report.json'
    result = pipeline.read(report)
    assert result['completed'] and len(result['acceptance']['metrics']) == 5
    pipeline.record('COMPLETE480_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY',
                    complete_report=str(report), goal_result=result['goal_result'],
                    acceptance=result['acceptance'])


if __name__ == '__main__':
    main()
