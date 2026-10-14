from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

project = Path('/home/gaob/CADTrack-ABC')
deadline = datetime(2026, 10, 14, 22, 46, 31, tzinfo=timezone.utc)
time.sleep(max(0, deadline.timestamp() - time.time()))
admission = json.loads((project / 'outputs/r1_main_full98_20261015/remaining_candidates_admission.json').read_text())
jobs = []
for original in admission['jobs']:
    row = dict(original)
    process = subprocess.run(['ps', '-p', str(row['pid']), '-o', 'args='], capture_output=True, text=True)
    row['actual_pid_live'] = process.returncode == 0
    if row['actual_pid_live']:
        assert 'cadtrack_abc.validate_future_actions' in process.stdout
        assert row['checkpoint'] in process.stdout
        assert f"--shard {row['shard']}" in process.stdout
    receipt = Path(row['output']) / f"development_shard{row['shard']}.json"
    row['completed_receipt_present'] = receipt.exists()
    if receipt.exists():
        completed = json.loads(receipt.read_text())
        assert completed['completed'] and completed['epoch'] == row['epoch']
        row['completed_sequences'] = len(completed['records'])
        row['actual_shard_seconds'] = completed['seconds']
    jobs.append(row)
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(), deadline_utc=deadline.isoformat(),
    jobs=jobs, one_time_observation=True, neural_jobs_started=False,
    root_must_consume_original_native_exit_codes=True)
(project / 'outputs/r1_main_full98_20261015/remaining_candidates_observation_0646.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
