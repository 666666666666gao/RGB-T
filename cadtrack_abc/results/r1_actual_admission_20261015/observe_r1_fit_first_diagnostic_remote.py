from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

project = Path('/home/gaob/CADTrack-ABC')
root = project / 'outputs/r1_matched_fits_20261015'
launch = json.loads((root / 'launch.json').read_text())
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
    controller_pid=4089586, controller_exists=Path('/proc/4089586').exists(), jobs=[])
for job in launch['jobs']:
    fit = root / f"fit_{job['policy']}"
    history_path = fit / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    complete_path = fit / 'fit_complete.json'
    complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
    proc = Path('/proc') / str(job['pid'])
    device = ([entry.decode() for entry in (proc / 'environ').read_bytes().split(b'\0')
               if entry.startswith(b'CUDA_VISIBLE_DEVICES=')] if proc.exists() else [])
    candidate_zero = fit / 'candidate_epoch00.pth'
    result['jobs'].append(dict(pid=job['pid'], process_exists=proc.exists(), physical_gpu=job['physical_gpu'],
        policy=job['policy'], command=job['command'], cuda_device_setting=device,
        completed_epochs=len(history), latest_epoch=history[-1] if history else None,
        history_elapsed_seconds=(history_path.stat().st_mtime - candidate_zero.stat().st_mtime) if history else None,
        completed_fit=complete))
for name in ['child_exits.json', 'fits_complete.json']:
    path = root / name
    result[name] = json.loads(path.read_text()) if path.exists() else None
result['gpu_memory_utilization_csv'] = subprocess.run(['nvidia-smi', '-i', '2,3',
    '--query-gpu=index,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
    capture_output=True, text=True, check=True).stdout
result['visual_sanity'] = json.loads((project / 'outputs/search_action_visual_sanity_20261014/sanity_complete.json').read_text())
(root / 'first_diagnostic.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
