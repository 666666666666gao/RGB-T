from datetime import datetime, timezone
from pathlib import Path
import json
import os

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import cv2

processes = []
for pid in (81006, 81007, 134761):
    status = dict(line.split(':', 1) for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines() if ':' in line)
    processes.append(dict(pid=pid, actual_threads=int(status['Threads']),
        resident_memory_kib=int(status['VmRSS'].split()[0])))
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(), opencv_version=cv2.__version__,
    logical_cores=os.cpu_count(), same_launcher_omp_threads=4,
    standalone_default_opencv_threads=cv2.getNumThreads(), processes=processes,
    scope='Read-only current process thread counts and separate warm OpenCV default; no inference process modified.',
    new_neural_jobs_started=False)
root = Path('/home/gaob/CADTrack-ABC/profile_output')
(root / 'opencv_thread_inspection_20261015_0824.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
