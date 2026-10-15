from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

deadline = datetime(2026, 10, 15, 3, 50, tzinfo=timezone.utc)
time.sleep(max(0, deadline.timestamp() - time.time()))
root = Path('/home/gaob/CADTrack-ABC/outputs/r1_main_reports_20261015')
state = json.loads((root / 'benchmark_state.json').read_text())
process = subprocess.run(['ps', '-p', str(state['controller_pid']), '-o', 'args='],
    capture_output=True, text=True)
active = {}
for name, phase in state['phases'].items():
    if phase['status'] == 'RUNNING':
        jobs = []
        for row in phase['jobs']:
            current = subprocess.run(['ps', '-p', str(row['pid']), '-o', 'args='], capture_output=True, text=True)
            live = current.returncode == 0
            if live:
                assert row['command'][0] in current.stdout
            jobs.append(dict(row, actual_pid_live=live))
        active[name] = jobs
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(), controller_pid=state['controller_pid'],
    controller_actual_pid_live=process.returncode == 0, status=state['status'],
    checkpoint=state['checkpoint'], checkpoint_sha256=state['checkpoint_sha256'],
    current_jobs=active, reports=state['reports'], one_time_observation=True,
    new_neural_jobs_started=False, temperature_or_power_queried=False)
(root / 'observation_1150.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
