from pathlib import Path
from datetime import datetime, timezone
import json

root = Path('/home/gaob/CADTrack-ABC/outputs/native_action_collection_20261014')
launch = json.loads((root / 'launch.json').read_text())
jobs, receipts = [], []
for job in launch['jobs']:
    shard = job['shard']
    finished = set()
    for split in ['train', 'validation']:
        path = root / ('collection_' + split) / f'shard{shard}' / 'collection_complete.json'
        if path.exists():
            data = json.loads(path.read_text())
            assert data['completed'] and data['teacher_sha256'] == launch['teacher_sha256']
            assert data['action_batch'] == 64 and data['model_family'] == 'CADTrackABC-native-current-quality'
            # Read the --queries argument from the original admitted command.
            command = job['commands'][0 if split == 'train' else 1]
            query_path = Path(command[command.index('--queries') + 1])
            expected = {(q['sequence'], q['frame']) for q in json.loads(query_path.read_text())}
            records = data['records']
            assert data['queries'] == len(records) == len(expected)
            assert {(r['sequence'], r['frame']) for r in records} == expected
            assert all(r['action_labels'] == 210 and r['native_state_reconstruction_equal']
                       and r['full_predicted_prefix_from_frame0'] for r in records)
            receipts.append(dict(split=split, shard=shard, queries=len(records),
                                 action_labels=sum(r['action_labels'] for r in records)))
            finished.add(split)
    proc = Path('/proc') / str(job['pid'])
    children = []
    if proc.exists():
        ids = (proc / 'task' / str(job['pid']) / 'children').read_text().split()
        for pid in ids:
            child = Path('/proc') / pid
            if child.exists():
                children.append(dict(pid=int(pid), cmdline=(child / 'cmdline').read_bytes().replace(b'\0', b' ').decode()))
    log = Path(job['log']).read_text()
    records = [json.loads(line) for line in log.splitlines() if line.startswith('{"event": "query_complete"')]
    jobs.append(dict(driver_pid=job['pid'], driver_exists=proc.exists(), physical_gpu=job['physical_gpu'],
                     shard=shard, finished_splits=sorted(finished), children=children,
                     completed_query_records=len(records), last_record=records[-1] if records else None,
                     log_tail=log[-4000:]))
complete = len(receipts) == 4
if complete:
    assert sum(r['queries'] for r in receipts) == 5697
    assert sum(r['action_labels'] for r in receipts) == 1196370
failed = any(not j['driver_exists'] and len(j['finished_splits']) != 2 for j in jobs)
result = dict(observed_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs, receipts=receipts,
              full_action_collection_complete=complete, incomplete_driver_exited=failed,
              optimizer_updates=0, neural_fit_started=False, new_formal_metrics=None)
print(json.dumps(result))
