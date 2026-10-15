from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

root=Path('/home/gaob/CADTrack-ABC/outputs/r2_dense_pair_20261015')
pair=json.loads((root/'pair_state.json').read_text())
result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),pair_state=pair,jobs={},new_neural_jobs_started=False)
for label in ('H','HA'):
    folder=root/label
    queue=json.loads((folder/'queue_state.json').read_text())
    job=dict(queue_state=queue,progress=json.loads((folder/'fit/progress.json').read_text()),
        progress_mtime_utc=datetime.fromtimestamp((folder/'fit/progress.json').stat().st_mtime,timezone.utc).isoformat(),
        actual_training_log_tail=(folder/'full_training_0.log').read_text().splitlines()[-6:])
    phase=queue['phases']['full_training']
    job['actual_processes']=[dict(pid=pid,ps=subprocess.run(['ps','-p',str(pid),'-o','pid=,ppid=,stat=,pcpu=,pmem=,etime=,args='],capture_output=True,text=True).stdout.strip()) for pid in phase['pids']]
    if (folder/'fit/history.json').exists():
        job['completed_epochs']=json.loads((folder/'fit/history.json').read_text())
    result['jobs'][label]=job
result['pair_exit']=json.loads((root/'pair_exit.json').read_text()) if (root/'pair_exit.json').exists() else None
(root/'followup1442_observation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
