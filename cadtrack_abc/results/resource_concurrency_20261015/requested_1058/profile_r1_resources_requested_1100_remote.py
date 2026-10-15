"""Read-only 30-second resource sample for server 28, physical GPUs 2 and 3."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import os
import subprocess
import time

project = Path('/home/gaob/CADTrack-ABC')
output = project / 'profile_output/r1_resources_requested_20261015_1100.json'
assert not output.exists()

def cpu_stat():
    return [int(value) for value in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]

def processes():
    result = subprocess.run(['nvidia-smi', '-i', '2,3',
        '--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = []
    for fields in csv.reader(io.StringIO(result.stdout)):
        pid = int(fields[0])
        command_path = Path('/proc') / str(pid) / 'cmdline'
        command = command_path.read_bytes().replace(b'\0', b' ').decode() if command_path.exists() else None
        own = command is not None and ('cadtrack_abc.' in command or str(project) in command)
        rows.append(dict(pid=pid, used_mib=int(fields[1]), gpu_uuid=fields[2].strip(),
                         project_job=own, command=command if own else None))
    return rows

before_processes = processes()
cpu_before = cpu_stat()
started_utc = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
sampler = subprocess.Popen(['nvidia-smi', '-i', '2,3',
    '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory',
    '--format=csv,noheader,nounits', '--loop-ms=1000'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(30)
sampler.terminate()
gpu_text, gpu_error = sampler.communicate()
assert not gpu_error, gpu_error
elapsed = time.perf_counter() - started
delta = [after - before for after, before in zip(cpu_stat(), cpu_before)]
total = sum(delta)
samples = {}
for row in csv.reader(io.StringIO(gpu_text)):
    if len(row) == 6:
        gpu = int(row[0])
        used, capacity, activity, bandwidth = map(int, row[2:])
        samples.setdefault(gpu, []).append(dict(uuid=row[1].strip(), used_mib=used,
            total_mib=capacity, gpu_utilization_percent=activity,
            memory_utilization_percent=bandwidth))
assert set(samples) == {2, 3}
summary = {}
for gpu, rows in samples.items():
    summary[gpu] = dict(uuid=rows[-1]['uuid'], samples=len(rows),
        total_mib=rows[-1]['total_mib'], peak_used_mib=max(row['used_mib'] for row in rows),
        minimum_free_mib=min(row['total_mib'] - row['used_mib'] for row in rows),
        mean_gpu_utilization_percent=sum(row['gpu_utilization_percent'] for row in rows) / len(rows),
        min_gpu_utilization_percent=min(row['gpu_utilization_percent'] for row in rows),
        max_gpu_utilization_percent=max(row['gpu_utilization_percent'] for row in rows))
meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
result = dict(status='READ_ONLY_RESOURCE_SAMPLE_COMPLETE', started_utc=started_utc,
    observed_utc=datetime.now(timezone.utc).isoformat(), window_seconds=elapsed,
    logical_cpu_cores=os.cpu_count(), load_average=list(os.getloadavg()),
    cpu_busy_percent=100 * (total - delta[3] - delta[4]) / total,
    cpu_iowait_percent=100 * delta[4] / total,
    available_system_memory_kib=int(meminfo['MemAvailable'].split()[0]),
    gpu_summary=summary, gpu_samples=samples, processes_before=before_processes,
    processes_after=processes(), temperature_or_power_queried=False,
    production_instrumentation_changes=False, new_neural_jobs_started=False)
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key != 'gpu_samples'}, indent=2))
