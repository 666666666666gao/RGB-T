from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

root = Path('/home/gaob/CADTrack-ABC/outputs/r1_control_reports_20261015')
state = json.loads((root/'recovery_1308_state.json').read_text())
rows, progress = [], []
for dataset in ('rgbt234','rgbt210'):
    for job in state['phases']['infer_'+dataset]['jobs']:
        process = subprocess.run(['ps','-p',str(job['pid']),'-o','args='],capture_output=True,text=True)
        rows.append(dict(dataset=dataset,pid=job['pid'],physical_gpu=job['physical_gpu'],
            actual_pid_live=process.returncode==0,actual_command=process.stdout.strip()))
        records = [json.loads(line) for line in (root/f"recovery_1308_infer_{dataset}_shard{job['shard']}.log").read_text().splitlines() if line.startswith('{')]
        completed = [r for r in records if r.get('event')=='sequence_complete']
        progress.append(dict(dataset=dataset,shard=job['shard'],completed_sequences=len(completed),
            completed_frame_lower_bound=sum(r['frames'] for r in completed)))
exit_path = root/'recovery_1308_exit.json'
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(),status=state['status'],
    controller_pid=state['controller_pid'],
    controller_actual_pid_live=subprocess.run(['ps','-p',str(state['controller_pid'])],capture_output=True).returncode==0,
    actual_jobs=rows,completed_sequence_progress=progress,reports=state['reports'],
    exit_receipt=json.loads(exit_path.read_text()) if exit_path.exists() else None,
    new_neural_jobs_started=False,production_source_changed=False)
(root/'observation_resume_1350.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
