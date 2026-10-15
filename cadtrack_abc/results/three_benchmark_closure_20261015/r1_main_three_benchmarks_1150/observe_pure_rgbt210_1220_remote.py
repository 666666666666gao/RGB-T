from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

deadline = datetime(2026, 10, 15, 4, 20, tzinfo=timezone.utc)
time.sleep(max(0, deadline.timestamp() - time.time()))
root = Path('/home/gaob/CADTrack-ABC/outputs/pure_cad_rgbt210_20261015')
state = json.loads((root / 'pipeline_state.json').read_text())
controller = subprocess.run(['ps', '-p', str(state['controller_pid']), '-o', 'args='],
    capture_output=True, text=True)
jobs = []
progress = []
if state['status'] == 'PURE_RGBT210_INFERENCE_RUNNING':
    for job in state['jobs']:
        actual = subprocess.run(['ps', '-p', str(job['pid']), '-o', 'args='],
            capture_output=True, text=True)
        if actual.returncode == 0:
            assert 'cadtrack_abc.infer_baseline' in actual.stdout and '--dataset rgbt210' in actual.stdout
        jobs.append(dict(job, actual_pid_live=actual.returncode == 0))
    for shard in (0, 1):
        records = []
        for line in (root / f'infer_shard{shard}.log').read_text().splitlines():
            if line.startswith('{'):
                record = json.loads(line)
                if record.get('event') == 'sequence_complete':
                    records.append(record)
        progress.append(dict(shard=shard, completed_sequences=len(records),
            completed_frame_pairs=sum(record['frames'] for record in records)))
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(), status=state['status'],
    controller_pid=state['controller_pid'], controller_actual_pid_live=controller.returncode == 0,
    state=state, actual_running_jobs=jobs, completed_sequence_progress=progress,
    one_time_observation=True, new_neural_jobs_started=False, baseline_lasher_rerun=False)
(root / 'observation_1220.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
