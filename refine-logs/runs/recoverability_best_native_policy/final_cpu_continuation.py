"""Finalize complete native reports once the original reviewed controller really finishes."""
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and Path.cwd() == Path('/data/gb/GOLA')
setup = Path('/data/gb/setup')
base = Path('/data/gb/outputs/recoverability_best_native_policy_native_full_20261005')
read = lambda path: json.loads(Path(path).read_text())
now = lambda: datetime.now(timezone(timedelta(hours=8))).isoformat()
owner_pid = 632884
ready_stage = 'COMPLETE_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY'
progress_path = setup / 'best_native_policy_pipeline_progress_20261005.json'
receipt_path = setup / 'best_native_policy_final_cpu_continuation_20261005.json'
review = read(setup / 'best_native_policy_final_cpu_continuation_source_review_20261005.json')
assert review['status'] == 'PASS' and review['same_family_provisional'] and not review['runtime_attested']
assert not receipt_path.exists() and not (base / 'complete_report').exists()
original = read(progress_path)
assert original['stage'] == 'FULL98_best'
state = subprocess.run(['ps', '-o', 'stat=', '-p', str(owner_pid)], text=True, capture_output=True).stdout.strip()
assert state and state[0] != 'Z'
first_check = datetime(2026, 10, 5, 7, 30, tzinfo=timezone(timedelta(hours=8)))
record = {'status': 'WAIT_ORIGINAL_NATIVE_REPORTS', 'pid': os.getpid(), 'original_controller_pid': owner_pid,
          'started_at_cst': now(), 'first_completion_check_cst': first_check.isoformat(), 'poll_interval_seconds': 180,
          'original_controller_stage_at_start': original['stage'], 'GPU_queries': 0, 'neural_calls': 0,
          'optimizer_updates': 0, 'automatic_restarts': 0, 'goal_status_changed': False}
receipt_path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
time.sleep(max(0, first_check.timestamp() - time.time()))
while True:
    progress = read(progress_path)
    state = subprocess.run(['ps', '-o', 'stat=', '-p', str(owner_pid)], text=True, capture_output=True).stdout.strip()
    if progress['stage'] == ready_stage:
        original_report = read(base / 'complete_core_report.json')
        assert original_report['completed'] and len(original_report['acceptance']) == 5
        for dataset in ('lasher', 'rgbt234'):
            assert read(base / dataset / 'independent_native_cpu_acceptance.json')['status'] == 'PASS'
        break
    assert state and state[0] != 'Z', ('Original controller stopped before full native CPU acceptance', progress['stage'])
    time.sleep(180)
subprocess.run(['/data/gb/envs/gola/bin/python', str(setup / 'best_native_policy_merge_complete_goal_report_20261005.py')],
               cwd='/data/gb/GOLA', check=True)
report = read(base / 'complete_report/complete_core_report.json')
assert report['completed'] and report['official_tracking_accuracy'] and len(report['acceptance']['metrics']) == 5
assert (base / 'complete_report/complete_core_report_completed.txt').is_file()
record.update(status='COMPLETE_FULL_NATIVE_REPORT_CONSOLIDATION', completed_at_cst=now(),
              complete_report=str(base / 'complete_report/complete_core_report.json'),
              actual_acceptance=report['acceptance'], original_all_native_CPU_passed=True,
              selected_checkpoint=report['selected_checkpoint'], goal_result=report['goal_result'])
receipt_path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
