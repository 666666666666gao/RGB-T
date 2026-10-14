"""Read-only resource and progress window for the four admitted GPU2/3 jobs."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import os
import subprocess
import time

project = Path('/home/gaob/CADTrack-ABC')
collection = project / 'outputs/native_action_collection_20261014'
evaluation = project / 'outputs/native_parent_rgbt234_20261015'

def cpu_stat():
    return [int(value) for value in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]

def process_info(pid):
    folder = Path('/proc') / str(pid)
    if not folder.exists():
        return dict(pid=pid, present=False)
    status = dict(line.split(':', 1) for line in (folder / 'status').read_text().splitlines() if ':' in line)
    command = (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, present=True, command=command, state=status['State'].strip(),
                cpu_ticks=int(fields[11]) + int(fields[12]),
                rss_kib=int(status['VmRSS'].split()[0]), threads=int(status['Threads']))

def events(log, event):
    prefix = '{"event": "' + event + '"'
    return [json.loads(line) for line in log.read_text().splitlines() if line.startswith(prefix)]

jobs = []
for job in json.loads((collection / 'launch.json').read_text())['jobs']:
    pid = job['pid']
    folder = Path('/proc') / str(pid)
    children = (folder / 'task' / str(pid) / 'children').read_text().split() if folder.exists() else []
    for child in children:
        info = process_info(int(child))
        if info['present'] and 'cadtrack_abc.collect_actions' in info['command']:
            jobs.append(dict(kind='action_collection', physical_gpu=job['physical_gpu'],
                             shard=job['shard'], log=job['log'], event='query_complete', before=info))
for job in json.loads((evaluation / 'launch.json').read_text())['jobs']:
    info = process_info(job['pid'])
    if info['present']:
        assert 'cadtrack_abc.infer' in info['command'] and str(evaluation / 'predictions') in info['command']
    jobs.append(dict(kind='parent_rgbt234_evaluation', physical_gpu=job['physical_gpu'], shard=job['shard'],
                     log=str(evaluation / f'infer_shard{job["shard"]}.log'), event='sequence_complete', before=info))

for job in jobs:
    job['records_before'] = len(events(Path(job['log']), job['event']))
cpu_before = cpu_stat()
started_utc = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
sampler = subprocess.Popen(['nvidia-smi', '-i', '2,3',
    '--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory',
    '--format=csv,noheader,nounits', '--loop-ms=1000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(30)
sampler.terminate()
gpu_text, gpu_error = sampler.communicate()
assert not gpu_error, gpu_error
elapsed = time.perf_counter() - started
cpu_delta = [after - before for after, before in zip(cpu_stat(), cpu_before)]
total = sum(cpu_delta)
samples = {}
for row in csv.reader(io.StringIO(gpu_text)):
    if len(row) == 5:
        gpu, used, capacity, activity, bandwidth = map(int, row)
        samples.setdefault(gpu, []).append(dict(used_mib=used, total_mib=capacity,
                                              gpu_utilization_percent=activity, memory_utilization_percent=bandwidth))
assert set(samples) == {2, 3}
gpu_summary = {}
for gpu, rows in samples.items():
    gpu_summary[gpu] = dict(samples=len(rows), peak_used_mib=max(row['used_mib'] for row in rows),
                           minimum_free_mib=min(row['total_mib'] - row['used_mib'] for row in rows),
                           total_mib=rows[-1]['total_mib'],
                           mean_gpu_utilization_percent=sum(row['gpu_utilization_percent'] for row in rows) / len(rows),
                           min_gpu_utilization_percent=min(row['gpu_utilization_percent'] for row in rows),
                           max_gpu_utilization_percent=max(row['gpu_utilization_percent'] for row in rows))
for job in jobs:
    job['after'] = process_info(job['before']['pid'])
    if job['before']['present'] and job['after']['present']:
        job['cpu_percent_one_core_basis'] = 100 * (job['after']['cpu_ticks'] - job['before']['cpu_ticks']) / os.sysconf('SC_CLK_TCK') / elapsed
    records = events(Path(job['log']), job['event'])
    job['records_after'] = len(records)
    job['completed_during_window'] = len(records) - job['records_before']
    job['last_record'] = records[-1] if records else None
    if job['kind'] == 'parent_rgbt234_evaluation':
        job['completed_frames'] = sum(row['frames'] for row in records)
        job['summed_inference_seconds'] = sum(row['seconds'] for row in records)
    else:
        recent = records[-100:]
        if len(recent) >= 2 and recent[0]['split'] == recent[-1]['split']:
            job['recent_seconds_per_query'] = (recent[-1]['seconds_since_start'] - recent[0]['seconds_since_start']) / (len(recent) - 1)
            job['recent_query_scope'] = 'Variable prefix lengths; descriptive progress, not matched workload throughput.'
    del job['log']
meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
result = dict(status='READ_ONLY_PARALLEL_RESOURCE_WINDOW_COMPLETE', started_utc=started_utc,
              observed_utc=datetime.now(timezone.utc).isoformat(), window_seconds=elapsed,
              logical_cpu_cores=os.cpu_count(), load_average=list(os.getloadavg()),
              cpu_busy_percent=100 * (total - cpu_delta[3] - cpu_delta[4]) / total,
              cpu_iowait_percent=100 * cpu_delta[4] / total,
              available_system_memory_kib=int(meminfo['MemAvailable'].split()[0]),
              gpu_summary=gpu_summary, gpu_samples=samples, jobs=jobs,
              original_jobs_modified=False, temperature_or_power_queried=False,
              limitation='Short equal-work collection benchmark showed 1.2913x throughput; heterogeneous evaluation plus collection throughput is not independently established by this resource window.')
output = evaluation / 'parallel_resource_profile_20261015.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key != 'gpu_samples'}, indent=2))
