from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

root = Path('/home/gaob/CADTrack-ABC/outputs/pure_cad_rgbt210_20261015')
state = json.loads((root / 'pipeline_state.json').read_text())
assert state['status'] == 'PURE_RGBT210_INFERENCE_RUNNING'
assert len(state['jobs']) == 2 and state['physical_gpu'] == 2
jobs = []
for job in state['jobs']:
    actual = subprocess.run(['ps', '-p', str(job['pid']), '-o', 'args='],
        capture_output=True, text=True, check=True).stdout.strip()
    assert 'cadtrack_abc.infer_baseline' in actual and '--dataset rgbt210' in actual
    jobs.append(dict(pid=job['pid'], actual_command=actual, physical_gpu=2, live=True))
gpu = subprocess.run(['nvidia-smi', '-i', '2',
    '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
    capture_output=True, text=True, check=True).stdout.strip()
result = dict(status='PURE_RGBT210_LAUNCH_VERIFIED', observed_utc=datetime.now(timezone.utc).isoformat(),
    controller_pid=state['controller_pid'], started_utc=state['started_utc'],
    jobs=jobs, gpu2_memory_csv=gpu, baseline_lasher_rerun=False,
    environment_rebuilt=False, temperature_or_power_queried=False,
    model_role='Pure CAD baseline, not a new enhanced-model training')
(root / 'launch_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
