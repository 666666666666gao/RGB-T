from pathlib import Path
from datetime import datetime, timezone
import json

root=Path('/home/gaob/CADTrack-ABC/outputs/native_future_sampling_20261014')
launch=json.loads((root/'launch.json').read_text())
receipts=[]
for split in ['train','validation']:
    for shard in [0,1]:
        path=root/('queries_'+split)/('sampling_shard'+str(shard)+'.json')
        if path.exists():
            data=json.loads(path.read_text())
            assert data['program_complete'] and data['full_shard_queries_complete'] and data['full_sequence_rollouts']
            assert not data['sanity'] and data['teacher_sha256']==launch['teacher_sha256']
            receipts.append(dict(split=split,shard=shard,sequences=data['sequences'],queries=data['queries'],
                                 seconds=data['seconds'],frames=sum(r['frames'] for r in data['records']),
                                 search_gates={key:sum(r['search_gates'][key] for r in data['records'])
                                               for key in data['records'][0]['search_gates']}))
jobs=[dict(driver_pid=job['pid'],physical_gpu=job['physical_gpu'],shard=job['shard'],
           proc_exists=(Path('/proc')/str(job['pid'])).exists(),
           log_tail=Path(job['log']).read_text()[-1500:]) for job in launch['jobs']]
complete=len(receipts)==4 and all(not job['proc_exists'] for job in jobs)
if complete:
    assert sum(r['sequences'] for r in receipts)==979 and sum(r['frames'] for r in receipts)==514083
result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),full_sampling_complete=complete,
            receipts=receipts,jobs=jobs,training_updates=0,teacher_sha256=launch['teacher_sha256'])
print(json.dumps(result))
