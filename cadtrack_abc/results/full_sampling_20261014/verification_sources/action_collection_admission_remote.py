from pathlib import Path
from datetime import datetime,timezone
import json

root=Path('/home/gaob/CADTrack-ABC/outputs/native_action_collection_20261014')
launch=json.loads((root/'launch.json').read_text())
jobs=[]
for job in launch['jobs']:
    children=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        status=proc/'status'
        if status.exists() and 'PPid:\t'+str(job['pid']) in status.read_text().splitlines():
            command=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
            if 'cadtrack_abc.collect_actions' in command:
                children.append(dict(pid=int(proc.name),cmdline=command))
    driver_exists=(Path('/proc')/str(job['pid'])).exists()
    log=Path(job['log']).read_text()
    records=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
    jobs.append(dict(driver_pid=job['pid'],physical_gpu=job['physical_gpu'],shard=job['shard'],
                     driver_exists=driver_exists,children=children,
                     completed_queries=len(records),first_record=records[0] if records else None,
                     last_record=records[-1] if records else None,log_tail=log[-1500:]))
assert all(job['driver_exists'] and len(job['children'])==1 for job in jobs),jobs
receipt=dict(status='TWO_ACTUAL_NATIVE_ACTION_COLLECTORS_ADMITTED',observed_utc=datetime.now(timezone.utc).isoformat(),
             jobs=jobs,updates=0,neural_fit_started=False,baseline_rerun=False)
(root/'admission.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
