from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

deadline = datetime(2026, 10, 15, 5, 0, tzinfo=timezone.utc)
time.sleep(max(0, deadline.timestamp() - time.time()))
root = Path('/home/gaob/CADTrack-ABC/outputs/r1_control_reports_20261015')
state = json.loads((root / 'benchmark_state.json').read_text())
process = subprocess.run(['ps', '-p', str(state['controller_pid']), '-o', 'args='],
    capture_output=True, text=True)
jobs = []
for name, phase in state['phases'].items():
    if phase['status'] == 'RUNNING':
        for job in phase['jobs']:
            actual = subprocess.run(['ps', '-p', str(job['pid']), '-o', 'args='],
                capture_output=True, text=True)
            if actual.returncode == 0:
                assert job['command'][0] in actual.stdout
            jobs.append(dict(phase=name, pid=job['pid'], physical_gpu=job['physical_gpu'],
                actual_pid_live=actual.returncode == 0, actual_command=actual.stdout.strip()))
progress = []
for name, phase in state['phases'].items():
    if phase['status'] == 'RUNNING' and name.startswith('infer_'):
        for shard in range(len(phase['jobs'])):
            records = [json.loads(line) for line in (root / f'{name}_shard{shard}.log').read_text().splitlines() if line.startswith('{')]
            records = [row for row in records if row.get('event') == 'sequence_complete']
            progress.append(dict(phase=name, shard=shard, completed_sequences=len(records),
                completed_frame_lower_bound=sum(row['frames'] for row in records)))
result = dict(completed_sequence_progress=progress, observed_utc=datetime.now(timezone.utc).isoformat(), controller_pid=state['controller_pid'],
    controller_actual_pid_live=process.returncode == 0, status=state['status'],
    phases=state['phases'], reports=state['reports'], actual_running_phase_jobs=jobs,
    new_neural_jobs_started=False, model_or_environment_changed=False)
(root / 'observation_1300.json').write_text(json.dumps(result, indent=2))
print(json.dumps(dict(result, phases={name: phase for name, phase in result['phases'].items() if phase['status'] == 'RUNNING'}), indent=2), flush=True)
