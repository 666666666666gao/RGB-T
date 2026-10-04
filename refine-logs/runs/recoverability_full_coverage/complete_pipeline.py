"""Continue the fixed collect→full train→both native benchmarks chain, without retries."""
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

setup = Path('/data/gb/setup')
main = Path('/data/gb/GOLA')
private = Path('/data/gb/experiments/recoverability_best_policy_fit_20261004')
python = '/data/gb/envs/gola/bin/python'
read = lambda path: json.loads(Path(path).read_text())
review = read(setup / 'full_coverage_pipeline_source_review_20261004.json')
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
env = os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(main), 'OMP_NUM_THREADS': '4',
                    'LD_LIBRARY_PATH': '/data/gb/envs/gola/lib'}
out = setup / 'full_coverage_pipeline_progress_20261004.json'
assert not out.exists()


def progress(stage):
    record = {'stage': stage, 'observed_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'automatic_restarts': 0, 'poll_interval_seconds': 180, 'official_goal_completed': False}
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


def wait_jobs(rows, groups):
    while not all(marker.is_file() for group in groups for marker in group):
        for row, owned in zip(rows, groups):
            # The original runner remains the owner, including TRAIN→VAL transitions.
            if all(marker.is_file() for marker in owned):
                continue
            state = subprocess.run(['ps', '-o', 'stat=', '-p', str(row['runner_pid'])],
                                   text=True, capture_output=True, check=True).stdout.strip()
            assert state and not state.startswith('Z'), row
        time.sleep(180)


def run(script, *args, cwd=main):
    subprocess.run([python, str(setup / script), *args], cwd=cwd,
                   env=env | {'PYTHONPATH': str(cwd) + ':' + str(main)}, check=True)


progress('WAIT_ORIGINAL_FOUR_COLLECTOR_QUEUES')
collection = read(setup / 'full_coverage_collection_launch_20261004.json')
groups = [[Path('/data/gb/outputs/recoverability_full_coverage_collect_' + part + '_gpu' + str(gpu) + '_20261004/job_completed.txt')
           for part in ('train', 'validation')] for gpu in range(4)]
wait_jobs(collection['runs'], groups)
progress('ALL881_98_COLLECTION_COMPLETE_CPU_GT_MERGE')
for part in ('train', 'validation'):
    run('full_coverage_actual_partition_merge_cpu_20261004.py', '--partition', part)
run('full_coverage_actual_completed_cache_gate_cpu_20261004.py')
progress('DIRECT_FULL60_EPOCH_ABC_FITS')
run('launch_full_coverage_fit_20261004.py')
fits = read(setup / 'full_coverage_fit_launch_20261004.json')
wait_jobs(fits['runs'], [[Path(row['root']) / 'job_completed.txt'] for row in fits['runs']])
progress('FULL60_EPOCH_TRAINING_COMPLETE_CPU_ACCEPTANCE')
run('full_coverage_actual_full_fit_cpu_audit_20261004.py', '--stage', 'full', cwd=private)
progress('ONE_CHECKPOINT_BOTH_COMPLETE_NATIVE_BENCHMARKS')
run('launch_full_coverage_native_complete_20261004.py')
native = read(setup / 'full_coverage_native_launch_20261004.json')
base = Path('/data/gb/outputs/recoverability_full_coverage_native_full_20261004')
wait_jobs(native['runs'], [[base / dataset / 'shards' / ('gpu' + str(gpu)) / 'inference_completed.txt'
                           for dataset in ('lasher', 'rgbt234')] for gpu in range(4)])
progress('BOTH_DATASETS_COMPLETE_ACTUAL_GT_REPORTS')
for dataset in ('lasher', 'rgbt234'):
    run('merge_full_coverage_native_complete_20261004.py', '--dataset', dataset)
    subprocess.run(['bash', str(setup / 'report_full_coverage_native_complete_20261004.sh'), dataset],
                   cwd=main, env=env, check=True)
    run('audit_full_coverage_native_complete_cpu_20261004.py', '--dataset', dataset)
progress('COMPLETE_TRAINING_AND_BOTH_FULL_NATIVE_CPU_REPORTS_READY_GOAL_REQUIRES_RESULT_AUDIT')
