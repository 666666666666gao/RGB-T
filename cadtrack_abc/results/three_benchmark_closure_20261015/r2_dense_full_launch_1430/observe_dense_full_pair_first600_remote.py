from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import time

root=Path('/home/gaob/CADTrack-ABC/outputs/r2_dense_pair_20261015')


def cpu():
    values=[int(value) for value in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(values),values[3],values[4]


begin=cpu()
samples=[]
for _ in range(30):
    output=subprocess.run(['nvidia-smi','-i','2,3','--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    rows=[[int(value.strip()) for value in line.split(',')] for line in output.splitlines()]
    samples.append(rows)
    time.sleep(1)
end=cpu()
total=end[0]-begin[0]
pair=json.loads((root/'pair_state.json').read_text())
result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),pair_state=pair,
    resource_scope='30second first-necessary-diagnostic sample during formal training; only physical GPU2/3.',
    cpu_busy_percent=100*(total-(end[1]-begin[1])-(end[2]-begin[2]))/total,
    cpu_iowait_percent=100*(end[2]-begin[2])/total,logical_cpu_cores=os.cpu_count(),
    load_average=os.getloadavg(),meminfo=Path('/proc/meminfo').read_text(),
    gpu_samples=samples,jobs={},new_neural_jobs_started=False,temperature_power_queried=False)
for label in ('H','HA'):
    folder=root/label
    queue=json.loads((folder/'queue_state.json').read_text())
    job=dict(queue_state=queue,progress=json.loads((folder/'fit/progress.json').read_text()) if (folder/'fit/progress.json').exists() else None,
        pipeline_log_tail=(root/f'{label}_pipeline.log').read_text().splitlines()[-20:])
    pids=[pid for phase in queue['phases'].values() for pid in phase.get('pids',[])]
    job['actual_processes']=[dict(pid=pid,command=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True).stdout.strip()) for pid in pids]
    if (folder/'fit/history.json').exists():
        job['actual_completed_epochs']=json.loads((folder/'fit/history.json').read_text())
    result['jobs'][label]=job
exit_file=root/'pair_exit.json'
result['pair_exit']=json.loads(exit_file.read_text()) if exit_file.exists() else None
(root/'first600_observation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
