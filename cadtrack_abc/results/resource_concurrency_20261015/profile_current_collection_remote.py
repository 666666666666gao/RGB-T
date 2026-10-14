"""Read-only30s resource window on physical GPU2/3; no model or job changes."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import os
import shutil
import subprocess
import time

root = Path('/home/gaob/CADTrack-ABC/outputs/native_action_collection_20261014')
launch = json.loads((root / 'launch.json').read_text())

def cpu_stat():
    return [int(value) for value in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]

def process_stat(pid):
    folder = Path('/proc') / str(pid)
    fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    status = dict(line.split(':', 1) for line in (folder / 'status').read_text().splitlines() if ':' in line)
    counters = dict(line.split(': ', 1) for line in (folder / 'io').read_text().splitlines())
    return dict(cpu_ticks=int(fields[11]) + int(fields[12]),
                rss_kib=int(status['VmRSS'].split()[0]), threads=int(status['Threads']),
                read_bytes=int(counters['read_bytes']), write_bytes=int(counters['write_bytes']))

def query_records(job):
    return [json.loads(line) for line in Path(job['log']).read_text().splitlines()
            if line.startswith('{"event": "query_complete"')]

jobs = []
for job in launch['jobs']:
    pid = job['pid']
    driver = Path('/proc') / str(pid)
    children = (driver / 'task' / str(pid) / 'children').read_text().split() if driver.exists() else []
    for child in children:
        folder = Path('/proc') / child
        if folder.exists():
            command = (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'cadtrack_abc.collect_actions' in command:
                jobs.append(dict(driver_pid=pid, collector_pid=int(child), physical_gpu=job['physical_gpu'],
                                 shard=job['shard'], command=command, before=process_stat(int(child)),
                                 records_before=len(query_records(job)), job=job))
assert len(jobs) == 2, 'Read actual process state: this window expects the two admitted active collectors.'
start_time = datetime.now(timezone.utc).isoformat()
cpu_before = cpu_stat()
started = time.perf_counter()
sampler = subprocess.Popen(['nvidia-smi', '-i', '2,3', '--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory',
                            '--format=csv,noheader,nounits', '--loop-ms=1000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(30)
sampler.terminate()
gpu_text, gpu_error = sampler.communicate()
assert not gpu_error, gpu_error
elapsed = time.perf_counter() - started
cpu_after = cpu_stat()
cpu_delta = [after - before for after, before in zip(cpu_after, cpu_before)]
total = sum(cpu_delta)
gpu_rows = {}
for row in csv.reader(io.StringIO(gpu_text)):
    if len(row) == 5:
        index, used, capacity, activity, bandwidth = map(int, row)
        gpu_rows.setdefault(index, []).append(dict(memory_used_mib=used, memory_total_mib=capacity,
                                                 gpu_utilization_percent=activity, memory_utilization_percent=bandwidth))
assert set(gpu_rows) == {2, 3}
gpu_summary = {}
for index, rows in gpu_rows.items():
    gpu_summary[index] = dict(samples=len(rows), memory_used_min_mib=min(row['memory_used_mib'] for row in rows),
                             memory_used_max_mib=max(row['memory_used_mib'] for row in rows),
                             memory_total_mib=rows[-1]['memory_total_mib'],
                             minimum_free_mib=min(row['memory_total_mib'] - row['memory_used_mib'] for row in rows),
                             mean_gpu_utilization_percent=sum(row['gpu_utilization_percent'] for row in rows) / len(rows),
                             minimum_gpu_utilization_percent=min(row['gpu_utilization_percent'] for row in rows),
                             maximum_gpu_utilization_percent=max(row['gpu_utilization_percent'] for row in rows),
                             mean_memory_utilization_percent=sum(row['memory_utilization_percent'] for row in rows) / len(rows))
for job in jobs:
    after = process_stat(job['collector_pid'])
    job['cpu_percent_one_core_basis'] = 100 * (after['cpu_ticks'] - job['before']['cpu_ticks']) / os.sysconf('SC_CLK_TCK') / elapsed
    job['after'] = after
    records = query_records(job.pop('job'))
    job['records_after'] = len(records)
    job['window_completed_queries'] = len(records) - job['records_before']
    recent = records[-100:]
    job['last_record'] = records[-1] if records else None
    if len(recent) >= 2 and recent[0]['split'] == recent[-1]['split']:
        seconds = recent[-1]['seconds_since_start'] - recent[0]['seconds_since_start']
        job['recent_query_interval_seconds'] = seconds / (len(recent) - 1)
        job['recent_scope'] = 'Last up to100 query completions including their intervening prefix work; not constant-cost events.'
meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
result = dict(status='READ_ONLY_30SECOND_COLLECTION_RESOURCE_PROFILE', started_utc=start_time,
              observed_utc=datetime.now(timezone.utc).isoformat(), window_seconds=elapsed,
              cpu_logical_cores=os.cpu_count(), load_average=list(os.getloadavg()),
              cpu_busy_percent=100 * (total - cpu_delta[3] - cpu_delta[4]) / total,
              cpu_idle_percent=100 * cpu_delta[3] / total, cpu_iowait_percent=100 * cpu_delta[4] / total,
              system_memory_total_kib=int(meminfo['MemTotal'].split()[0]),
              system_memory_available_kib=int(meminfo['MemAvailable'].split()[0]),
              gpu_summary=gpu_summary, gpu_samples=gpu_rows, jobs=jobs,
              home_free_bytes=shutil.disk_usage(root).free,
              additional_neural_jobs_started=0, current_jobs_modified=False,
              temperature_or_power_queried=False,
              limitation='GPU utilization is sampled activity, not measured SM throughput; concurrency speedup untested.')
out = root / 'resource_profile_20261015.json'
assert not out.exists()
out.write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key != 'gpu_samples'}, indent=2))
