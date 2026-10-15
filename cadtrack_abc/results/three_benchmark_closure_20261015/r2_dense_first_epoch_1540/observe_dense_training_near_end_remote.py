from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

root = Path('/home/gaob/CADTrack-ABC/outputs/r2_dense_pair_20261015')
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
    pair_state=json.loads((root/'pair_state.json').read_text()), jobs={},
    new_neural_jobs_started=False, source_or_configuration_changed=False)
for name in ('H', 'HA'):
    folder = root/name
    queue = json.loads((folder/'queue_state.json').read_text())
    job = dict(queue_state=queue,
        progress=json.loads((folder/'fit/progress.json').read_text()),
        completed_epochs=json.loads((folder/'fit/history.json').read_text()),
        active_processes=[], active_log_tails={}, development_completed_shards=[])
    for phase_name, phase in queue['phases'].items():
        if phase['status'] == 'RUNNING':
            for index, pid in enumerate(phase['pids']):
                process = subprocess.run(['ps','-p',str(pid),'-o',
                    'pid=,ppid=,stat=,pcpu=,pmem=,etime=,args='],capture_output=True,text=True)
                job['active_processes'].append(dict(phase=phase_name,pid=pid,
                    ps_returncode=process.returncode,ps=process.stdout.strip()))
                log = folder/f'{phase_name}_{index}.log'
                job['active_log_tails'][str(log)] = log.read_text().splitlines()[-10:]
    for filename in ('fit/fit_complete.json','fit/selection.json','pipeline_complete.json'):
        path = folder/filename
        if path.exists():
            job[filename] = json.loads(path.read_text())
    for path in sorted((folder/'development').glob('candidate_epoch*/development_shard*.json')):
        receipt = json.loads(path.read_text())
        job['development_completed_shards'].append(dict(path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shard=receipt['shard'],
            sequences=receipt['sequences'],frames=sum(row['frames'] for row in receipt['records']),
            checkpoint_sha256=receipt['checkpoint_sha256'],epoch=receipt['epoch'],
            completed=receipt['completed'],scope='One completed shard; not a full98 selection result.'))
    job['formal_reports'] = {}
    for dataset in ('lasher','rgbt234','rgbt210'):
        path = folder/'formal'/f'{dataset}_report'/'qualified_full_report.json'
        if path.exists():
            report = json.loads(path.read_text())
            job['formal_reports'][dataset] = dict(path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),completed=report['completed'],
                sequences=report['sequences'],frames=report['frames'],
                checkpoint_sha256=report['checkpoint_sha256'],
                overall_metrics_percent=report['overall_metrics_percent'],
                baseline_metrics_percent=report['baseline_metrics_percent'],
                delta_percentage_points=report['delta_percentage_points'],
                plus_two_reached=report['plus_two_reached'])
    result['jobs'][name] = job
path = root/'pair_exit.json'
result['pair_exit'] = json.loads(path.read_text()) if path.exists() else None
(root/'training_near_end_observation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
