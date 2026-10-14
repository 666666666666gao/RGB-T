from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

deadline = datetime.fromisoformat('2026-10-14T22:39:00+00:00')
time.sleep(max(0, (deadline - datetime.now(timezone.utc)).total_seconds()))
project = Path('/home/gaob/CADTrack-ABC')
admission = json.loads((project / 'outputs/r1_main_full98_20261015/remaining_candidates_admission.json').read_text())
rows = []
for job in admission['jobs']:
    process = subprocess.run(['ps', '-p', str(job['pid']), '-o', 'pid=,args='],
        capture_output=True, text=True)
    assert process.returncode in (0, 1)
    live = process.returncode == 0
    if live:
        assert 'cadtrack_abc.validate_future_actions' in process.stdout
        assert job['output'] in process.stdout
    source = Path(job['output']) / f"development_shard{job['shard']}.json"
    receipt = json.loads(source.read_text()) if source.is_file() else None
    if receipt is not None:
        assert receipt['completed'] and receipt['epoch'] == job['epoch']
        assert receipt['shard'] == job['shard'] and receipt['shards'] == 2
        assert receipt['checkpoint'] == job['checkpoint']
    rows.append(dict(job, actual_pid_live=live, completed_receipt_present=receipt is not None,
        completed_sequences=receipt['sequences'] if receipt is not None else None,
        actual_shard_seconds=receipt['seconds'] if receipt is not None else None))
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(), deadline_utc=deadline.isoformat(),
    jobs=rows, one_time_observation=True, neural_jobs_started=False,
    root_must_consume_original_native_exit_codes=True)
output = project / 'outputs/r1_main_full98_20261015/remaining_candidates_observation_0639.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
