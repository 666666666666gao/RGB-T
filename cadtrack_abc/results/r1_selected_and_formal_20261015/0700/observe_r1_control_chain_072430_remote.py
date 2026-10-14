from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

deadline = datetime(2026, 10, 14, 23, 24, 30, tzinfo=timezone.utc)
time.sleep(max(0, deadline.timestamp() - time.time()))
root = Path('/home/gaob/CADTrack-ABC/outputs/r1_control_reports_20261015')
state = json.loads((root / 'benchmark_state.json').read_text())
process = subprocess.run(['ps', '-p', str(state['controller_pid']), '-o', 'args='],
    capture_output=True, text=True)
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(), controller_pid=state['controller_pid'],
    controller_actual_pid_live=process.returncode == 0, status=state['status'],
    parent_observation=state.get('parent_epoch00_observation'), phases=state['phases'],
    reports=state['reports'], child_queries_owned_by_existing_chain=True,
    additional_neural_pid_queries=False, new_neural_jobs_started=False)
if result['controller_actual_pid_live']:
    assert 'python3 -' in process.stdout
(root / 'observation_072430.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
