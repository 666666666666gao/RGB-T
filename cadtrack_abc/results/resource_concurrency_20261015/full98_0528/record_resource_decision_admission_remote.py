from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import subprocess

project = Path('/home/gaob/CADTrack-ABC')
gpu_rows = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=index,uuid',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
gpu_by_uuid = {row[1].strip(): int(row[0]) for row in csv.reader(io.StringIO(gpu_rows))}
app_rows = subprocess.run(['nvidia-smi', '-i', '2,3',
    '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'],
    capture_output=True, text=True, check=True).stdout
expected = {(2, 2, 1): 47291, (2, 5, 1): 3800, (3, 2, 0): 6168, (3, 5, 0): 39558}
jobs = []
for row in csv.reader(io.StringIO(app_rows)):
    pid, uuid, memory = int(row[0]), row[1].strip(), int(row[2])
    arguments = (Path('/proc') / str(pid) / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    assert 'cadtrack_abc.validate_future_actions' in arguments, (pid, arguments)
    def argument(flag):
        return arguments[arguments.index(flag) + 1]
    assert argument('--policy') == 'search_actions'
    epoch = int(Path(argument('--checkpoint')).stem.split('epoch')[1])
    gpu, shard = gpu_by_uuid[uuid], int(argument('--shard'))
    assert (gpu, epoch, shard) in expected
    assert argument('--output') == str(project / 'outputs/r1_main_full98_20261015' /
        f'candidate_epoch{epoch:02d}')
    jobs.append(dict(pid=pid, physical_gpu=gpu, epoch=epoch, shard=shard,
        used_mib=memory, original_native_session=expected[gpu, epoch, shard],
        checkpoint=argument('--checkpoint'), output=argument('--output')))
assert len(jobs) == 4 and {(row['physical_gpu'], row['epoch'], row['shard']) for row in jobs} == set(expected)
result = dict(status='MEASURED_TWO_WORKER_PER_GPU_ADMISSION_VERIFIED',
    observed_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs,
    concurrency_per_physical_gpu={'2': 2, '3': 2},
    next_observation_bjt='2026-10-15 06:10:00',
    timing_basis='Prior full98 shard runtimes815/1340seconds, short measured individual slowdown about20-30%; first necessary check shortly before smaller shards are expected to finish.',
    source_changed=False, checkpoints_changed=False, new_training_started=False,
    temperature_or_power_queried=False, formal_metrics=None)
(project / 'outputs/r1_main_full98_20261015/resource_decision_admission.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
