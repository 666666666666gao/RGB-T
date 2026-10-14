"""Bounded equal-work inference benchmark on currently idle physical GPU3.

Uses the existing validation CLI unchanged. No training, checkpoint changes,
official test, suspension of existing work, or production instrumentation.
"""
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import subprocess
import time

project = Path('/home/gaob/CADTrack-ABC')
output = project / 'profile_output/full98_concurrency_20261015_0530'
output.mkdir(exist_ok=False)
manifest_path = project / 'outputs/cad_abc_20261013/manifest.json'
config_path = project / 'cadtrack_abc/config.json'
checkpoint = project / 'outputs/r1_matched_fits_20261015/fit_search_actions/candidate_epoch01.pth'
manifest = json.loads(manifest_path.read_text())
config = json.loads(config_path.read_text())
names = sorted(manifest['validation_sequences'])
assert len(names) == 98
work = {}
for shard in (3, 33):
    selected = names[shard::49]
    assert len(selected) == 2
    work[shard] = dict(names=selected, frames=sum(len(
        (Path(config['train_root']) / name / 'init.txt').read_text().splitlines())
        for name in selected))
assert sum(row['frames'] for row in work.values()) == 1334
memory = subprocess.run(['nvidia-smi', '-i', '3', '--query-gpu=memory.used,memory.total',
    '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
used, capacity = map(int, memory.split(','))
assert used < 100 and capacity - used > 4000, memory
apps = subprocess.run(['nvidia-smi', '-i', '3', '--query-compute-apps=pid',
    '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
assert not apps.strip(), apps
environment = dict(os.environ, PYTHONPATH=str(project), CUDA_VISIBLE_DEVICES='3',
    OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
base = ['/data/wangwj/envs/CADTrack/bin/python', '-m', 'cadtrack_abc.validate_future_actions',
    '--config', str(config_path), '--checkpoint', str(checkpoint),
    '--manifest', str(manifest_path), '--policy', 'search_actions', '--shards', '49']
processes, streams, phases = [], [], []
started_utc = datetime.now(timezone.utc).isoformat()
gpu_stream = (output / 'gpu_samples.csv').open('w')
gpu_errors = (output / 'gpu_samples.stderr.txt').open('w')
sampler = subprocess.Popen(['nvidia-smi', '-i', '3',
    '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu',
    '--format=csv,noheader,nounits', '--loop-ms=1000'], stdout=gpu_stream, stderr=gpu_errors)

def cpu_stat():
    return list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))

def launch(phase, shard):
    folder = output / phase / ('worker' + str(shard))
    stream = (output / (phase + '_worker' + str(shard) + '.log')).open('w')
    streams.append(stream)
    process = subprocess.Popen(base + ['--output', str(folder), '--shard', str(shard)],
        cwd=project, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    processes.append(process)
    return process, folder, shard

def finish(job):
    process, folder, shard = job
    code = process.wait()
    assert code == 0, (folder, code)
    receipt = json.loads((folder / ('development_shard' + str(shard) + '.json')).read_text())
    assert receipt['completed'] and receipt['epoch'] == 1
    assert receipt['checkpoint'] == str(checkpoint) and receipt['policy_family'] == 'search_actions'
    assert receipt['shards'] == 49 and receipt['shard'] == shard
    assert [row['sequence'] for row in receipt['records']] == work[shard]['names']
    assert sum(row['frames'] for row in receipt['records']) == work[shard]['frames']
    return dict(shard=shard, child_exit_code=code, frames=work[shard]['frames'],
        tracking_seconds=receipt['seconds'], extra_forwards=sum(
            row['extra_forwards'] for row in receipt['records']), folder=str(folder))

try:
    for phase, concurrent in [('serial1', False), ('parallel1', True),
                              ('parallel2', True), ('serial2', False)]:
        tick = time.perf_counter()
        cpu_before = cpu_stat()
        phase_started = datetime.now(timezone.utc).isoformat()
        if concurrent:
            jobs = [launch(phase, shard) for shard in (3, 33)]
            children = [finish(job) for job in jobs]
        else:
            children = [finish(launch(phase, shard)) for shard in (3, 33)]
        elapsed = time.perf_counter() - tick
        delta = [after - before for after, before in zip(cpu_stat(), cpu_before)]
        total = sum(delta)
        phases.append(dict(name=phase, concurrent_workers=2 if concurrent else 1,
            started_utc=phase_started, ended_utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=elapsed, frames=1334, frames_per_second=1334 / elapsed,
            cpu_busy_percent=100 * (total - delta[3] - delta[4]) / total,
            cpu_iowait_percent=100 * delta[4] / total, children=children))
        (output / 'progress.json').write_text(json.dumps(dict(phases=phases), indent=2))
finally:
    for process in processes:
        if process.poll() is None:
            process.terminate()
            process.wait()
    sampler.terminate()
    sampler.wait()
    for stream in streams:
        stream.close()
    gpu_stream.close()
    gpu_errors.close()

assert sampler.returncode in (0, -15)
assert not (output / 'gpu_samples.stderr.txt').read_text()
samples = list(csv.reader((output / 'gpu_samples.csv').open()))
assert samples and all(len(row) == 5 for row in samples)
used_samples = [int(row[2]) for row in samples]
util_samples = [int(row[4]) for row in samples]
serial = sum(row['wall_seconds'] for row in phases if row['concurrent_workers'] == 1)
parallel = sum(row['wall_seconds'] for row in phases if row['concurrent_workers'] == 2)
serial_tracking = sum(sum(child['tracking_seconds'] for child in row['children'])
    for row in phases if row['concurrent_workers'] == 1)
parallel_tracking = sum(max(child['tracking_seconds'] for child in row['children'])
    for row in phases if row['concurrent_workers'] == 2)
equal_predictions = True
for shard in (3, 33):
    for name in work[shard]['names']:
        first = (output / 'serial1' / ('worker' + str(shard)) / (name + '.txt')).read_bytes()
        equal_predictions &= all((output / phase / ('worker' + str(shard)) /
            (name + '.txt')).read_bytes() == first for phase in ('parallel1', 'parallel2', 'serial2'))
result = dict(status='FULL_VIDEO_EQUAL_WORK_CONCURRENCY_BENCHMARK_COMPLETE',
    started_utc=started_utc, ended_utc=datetime.now(timezone.utc).isoformat(), physical_gpu=3,
    checkpoint=str(checkpoint), epoch=1, work=work, phases=phases,
    frames_per_condition=2668, serial_seconds=serial, parallel_seconds=parallel,
    aggregate_throughput_speedup=serial / parallel,
    serial_tracking_seconds=serial_tracking, parallel_tracking_seconds=parallel_tracking,
    approximate_tracking_window_speedup=serial_tracking / parallel_tracking,
    tracking_window_scope='Receipt timing excludes model load. Concurrent max is an approximate window; workers have no common launch barrier.',
    peak_memory_used_mib=max(used_samples), minimum_free_mib=capacity - max(used_samples),
    mean_gpu_utilization_percent=sum(util_samples) / len(util_samples),
    predictions_byte_equal_across_all_phases=bool(equal_predictions),
    original_running_jobs_changed=False, production_source_changed=False,
    optimizer_updates=0, temperature_or_power_queried=False, formal_metrics=None,
    scope='Four complete DEV videos per phase, same checkpoint and work; serial/parallel/parallel/serial order. End-to-end timing includes process/model startup and decoding. Short throughput study, not accuracy result.',
    recommendation='Two workers per card' if serial / parallel > 1.10
        and serial_tracking / parallel_tracking > 1.10 and equal_predictions
        else 'Keep one worker per card')
(output / 'benchmark_complete.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
