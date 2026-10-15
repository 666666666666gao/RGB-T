from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import time

root = Path('/home/gaob/CADTrack-ABC')
output = root / 'profile_output/dense_resources_requested_20261015_1830'
output.mkdir(exist_ok=False)
jobs = {'H': 826053, 'HA': 826055}


def cpu_snapshot():
    return {row.split()[0]: [int(x) for x in row.split()[1:9]]
            for row in Path('/proc/stat').read_text().splitlines()
            if row.startswith('cpu')}


def process_snapshot(pid):
    raw = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
    status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
    return dict(state=raw[0], cpu_ticks=int(raw[11]) + int(raw[12]),
                rss_kib=int(status['VmRSS'].split()[0]), threads=int(status['Threads']),
                io={key: int(value) for key, value in
                    (line.split(': ') for line in Path(f'/proc/{pid}/io').read_text().splitlines())},
                command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode())


def disks_snapshot():
    return {f'{row.split()[0]}:{row.split()[1]}':
            dict(name=row.split()[2], values=[int(x) for x in row.split()[3:]])
            for row in Path('/proc/diskstats').read_text().splitlines()}


def progress_snapshot():
    return {name: dict(value=json.loads((root / f'outputs/r2_dense_pair_20261015/{name}/fit/progress.json').read_text()),
                       mtime_utc=datetime.fromtimestamp((root / f'outputs/r2_dense_pair_20261015/{name}/fit/progress.json').stat().st_mtime,
                                                       timezone.utc).isoformat()) for name in jobs}


dataset = Path('/data/wangwj/dataset/LasHeR/trainingset')
device = dataset.stat().st_dev
dataset_device = f'{os.major(device)}:{os.minor(device)}'
children = {name: [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
            for name, pid in jobs.items()}
all_pids = list(jobs.values()) + [pid for values in children.values() for pid in values]
start_cpu = cpu_snapshot()
start_process = {str(pid): process_snapshot(pid) for pid in all_pids}
start_disks = disks_snapshot()
start_progress = progress_snapshot()
started = datetime.now(timezone.utc).isoformat()
begin = time.monotonic()
with (output / 'gpu_samples.csv').open('w') as stream:
    sampler = subprocess.Popen(['nvidia-smi', '-i', '2,3',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory',
        '--format=csv,noheader,nounits', '-l', '1'], stdout=stream, stderr=subprocess.PIPE, text=True)
    time.sleep(60)
    sampler.terminate()
    _, error = sampler.communicate()
assert not error, error
elapsed = time.monotonic() - begin
end_cpu = cpu_snapshot()
end_process = {str(pid): process_snapshot(pid) for pid in all_pids}
end_disks = disks_snapshot()
end_progress = progress_snapshot()
cpu = {}
for name, initial in start_cpu.items():
    delta = [b - a for a, b in zip(initial, end_cpu[name])]
    total = sum(delta)
    cpu[name] = dict(busy_percent=100 * (total - delta[3] - delta[4]) / total,
                     iowait_percent=100 * delta[4] / total)
hz = os.sysconf('SC_CLK_TCK')
process = {}
for pid, initial in start_process.items():
    final = end_process[pid]
    process[pid] = dict(cpu_percent_one_core=100 * (final['cpu_ticks'] - initial['cpu_ticks']) / hz / elapsed,
                        rss_gib=final['rss_kib'] / 1024**2, state=final['state'], threads=final['threads'],
                        read_bytes=final['io']['read_bytes'] - initial['io']['read_bytes'],
                        read_chars=final['io']['rchar'] - initial['io']['rchar'], command=final['command'])
samples = [[int(x.strip()) for x in row.split(',')]
           for row in (output / 'gpu_samples.csv').read_text().splitlines()]
gpu = {}
for index in (2, 3):
    rows = [row for row in samples if row[0] == index]
    gpu[str(index)] = dict(samples=len(rows), total_gib=rows[0][2] / 1024,
        used_min_gib=min(row[1] for row in rows) / 1024,
        used_max_gib=max(row[1] for row in rows) / 1024,
        free_min_gib=min(row[2] - row[1] for row in rows) / 1024,
        utilization_mean_percent=sum(row[3] for row in rows) / len(rows),
        utilization_min_percent=min(row[3] for row in rows),
        utilization_max_percent=max(row[3] for row in rows),
        zero_utilization_samples=sum(row[3] == 0 for row in rows),
        at_least_90_utilization_samples=sum(row[3] >= 90 for row in rows),
        memory_utilization_mean_percent=sum(row[4] for row in rows) / len(rows))
disk_delta = [b - a for a, b in zip(start_disks[dataset_device]['values'], end_disks[dataset_device]['values'])]
disk = dict(device=dataset_device, name=end_disks[dataset_device]['name'],
            read_mib_per_second=disk_delta[2] * 512 / 1024**2 / elapsed,
            write_mib_per_second=disk_delta[6] * 512 / 1024**2 / elapsed,
            read_await_ms=disk_delta[3] / disk_delta[0] if disk_delta[0] else None,
            busy_percent=100 * disk_delta[9] / (elapsed * 1000),
            average_queue_depth=disk_delta[10] / (elapsed * 1000),
            scope='Whole backing device, includes other users and processes; not training-only I/O.')
memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
result = dict(started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(), duration_seconds=elapsed,
    cpu_logical_count=os.cpu_count(), cpu_total=cpu['cpu'],
    busiest_cores=sorted(((name, value) for name, value in cpu.items() if name != 'cpu'),
                         key=lambda item: item[1]['busy_percent'], reverse=True)[:8],
    load_average=os.getloadavg(), available_ram_gib=int(memory['MemAvailable'].split()[0]) / 1024**2,
    gpu=gpu, dataset_disk=disk, processes=process, direct_children=children,
    progress_start=start_progress, progress_end=end_progress,
    new_neural_jobs_started=0, training_sources_or_configuration_changed=False,
    temperature_or_power_queried=False,
    scope='User-requested read-only60second resource and training-throughput assessment; no concurrency benchmark.')
(output / 'resource_report.json').write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key not in ('processes', 'direct_children')}, indent=2), flush=True)
