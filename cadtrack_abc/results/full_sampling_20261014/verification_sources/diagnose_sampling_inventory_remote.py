from pathlib import Path
from datetime import datetime,timezone
import json

project=Path('/home/gaob/CADTrack-ABC')
root=project/'outputs/native_future_sampling_20261014'
config=json.loads((project/'cadtrack_abc/config.json').read_text())
inventory=json.loads((project/'outputs/cad_abc_20261013/manifest.json').read_text())
launch=json.loads((root/'launch.json').read_text())
rows=[]
coverage={}
for split in ['train','validation']:
    expected=set(inventory['train_sequences' if split=='train' else 'validation_sequences'])
    observed=[]
    for shard in [0,1]:
        path=root/('queries_'+split)/f'sampling_shard{shard}.json'
        data=json.loads(path.read_text())
        records=data['records']
        observed.extend(record['sequence'] for record in records)
        mismatches=[]
        actual_frames=0
        for record in records:
            sequence=Path(config['train_root'])/record['sequence']
            rgb_count=sum(1 for _ in (sequence/'visible').iterdir())
            tir_count=sum(1 for _ in (sequence/'infrared').iterdir())
            gt_count=sum(bool(line.strip()) for line in (sequence/'init.txt').read_text().splitlines())
            actual_frames+=rgb_count
            if not rgb_count==tir_count==gt_count==record['frames']:
                mismatches.append(dict(sequence=record['sequence'],record_frames=record['frames'],rgb=rgb_count,tir=tir_count,gt=gt_count))
        queries=json.loads((path.parent/f'queries_shard{shard}.json').read_text())
        rows.append(dict(split=split,shard=shard,receipt_shards=data['shards'],sequences=data['sequences'],
                         unique_sequences=len({r['sequence'] for r in records}),records=len(records),
                         queries=data['queries'],query_file_length=len(queries),
                         receipt_frames=sum(record['frames'] for record in records),actual_frames=actual_frames,
                         mismatches=mismatches,seconds=data['seconds'],teacher_sha256=data['teacher_sha256'],
                         program_complete=data['program_complete'],full_shard_queries_complete=data['full_shard_queries_complete'],
                         full_sequence_rollouts=data['full_sequence_rollouts'],sanity=data['sanity']))
    coverage[split]=dict(expected=len(expected),observed=len(observed),unique=len(set(observed)),
                         missing=sorted(expected-set(observed)),unexpected=sorted(set(observed)-expected))
result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),rows=rows,coverage=coverage,
            total_sequences=sum(row['sequences'] for row in rows),total_receipt_frames=sum(row['receipt_frames'] for row in rows),
            total_real_frames=sum(row['actual_frames'] for row in rows),
            drivers=[dict(pid=job['pid'],proc_exists=(Path('/proc')/str(job['pid'])).exists()) for job in launch['jobs']],
            forecast_frame_inventory=launch['sampled_frame_inventory'],updates=0,neural_jobs_started=0)
(root/'inventory_diagnosis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
