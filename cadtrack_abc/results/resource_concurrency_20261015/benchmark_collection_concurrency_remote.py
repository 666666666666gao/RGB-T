"""Equal real query workload, serial vs two concurrent workers on GPU2.

Only the owned GPU2 collector is temporarily stopped; its state is resumed in
finally. GPU3 and original drivers remain unchanged. No optimizer runs.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import signal
import subprocess
import time

project = Path('/home/gaob/CADTrack-ABC')
collection = project / 'outputs/native_action_collection_20261014'
launch = json.loads((collection / 'launch.json').read_text())
job = next(job for job in launch['jobs'] if job['physical_gpu'] == 2)
driver = Path('/proc') / str(job['pid'])
children = (driver / 'task' / str(job['pid']) / 'children').read_text().split()
collectors = []
for child in children:
    folder = Path('/proc') / child
    if folder.exists():
        command = (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if 'cadtrack_abc.collect_actions' in command and '--split train' in command:
            collectors.append((int(child), command))
assert len(collectors) == 1
primary_pid, primary_command = collectors[0]
memory = subprocess.run(['nvidia-smi', '-i', '2', '--query-gpu=memory.used,memory.total',
                         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
used, capacity = [int(value) for value in memory.split(',')]
assert capacity - used >= 12000, memory
sample = project / 'outputs/native_future_sampling_20261014/queries_train/queries_shard0.json'
all_queries = json.loads(sample.read_text())
first = '1strowrightgirl'
queries = sorted([query for query in all_queries if query['sequence'] == first], key=lambda query: query['frame'])[:5]
assert [query['frame'] for query in queries] == [38, 39, 40, 41, 43]
out = project / 'outputs/collection_concurrency_benchmark_20261015'
out.mkdir(exist_ok=False)
(out / 'queries.json').write_text(json.dumps(queries, indent=2))
python = '/data/wangwj/envs/CADTrack/bin/python'
base = [python, '-m', 'cadtrack_abc.collect_actions', '--config', str(project / 'cadtrack_abc/config.json'),
        '--teacher', launch['teacher'], '--teacher-policy', 'native',
        '--manifest', str(project / 'outputs/cad_abc_20261013/manifest.json'),
        '--queries', str(out / 'queries.json'), '--split', 'train', '--device', 'cuda', '--horizon', '3', '--action-batch', '64']
env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
processes, streams, phase_results = [], [], []
start = datetime.now(timezone.utc).isoformat()

def worker(name):
    stream = (out / (name + '.log')).open('w')
    streams.append(stream)
    command = base + ['--output', str(out / name)]
    process = subprocess.Popen(command, cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT)
    processes.append(process)
    return process

def finish(name, elapsed):
    receipt = json.loads((out / name / 'collection_complete.json').read_text())
    assert receipt['completed'] and receipt['queries'] == 5 and receipt['teacher_sha256'] == launch['teacher_sha256']
    assert {(record['sequence'], record['frame']) for record in receipt['records']} == {(query['sequence'], query['frame']) for query in queries}
    assert all(record['action_labels'] == 210 and record['native_state_reconstruction_equal'] for record in receipt['records'])
    phase_results.append(dict(name=name, queries=5, labels=1050, wall_seconds=elapsed,
                              collector_seconds=receipt['records'][-1]['seconds_since_start']))

os.kill(primary_pid, signal.SIGSTOP)
try:
    time.sleep(1)
    state = (Path('/proc') / str(primary_pid) / 'status').read_text()
    assert any(line.startswith('State:') and '\tT' in line for line in state.splitlines())
    for name in ['single_a', 'single_b']:
        tick = time.perf_counter()
        process = worker(name)
        assert process.wait() == 0, name
        finish(name, time.perf_counter() - tick)
    tick = time.perf_counter()
    parallel = [worker(name) for name in ['parallel_a', 'parallel_b']]
    codes = [process.wait() for process in parallel]
    parallel_seconds = time.perf_counter() - tick
    assert codes == [0, 0], codes
    for name in ['parallel_a', 'parallel_b']:
        finish(name, parallel_seconds)
finally:
    for process in processes:
        if process.poll() is None:
            process.terminate()
            process.wait()
    for stream in streams:
        stream.close()
    os.kill(primary_pid, signal.SIGCONT)
    (out / 'primary_resumed.json').write_text(json.dumps(dict(primary_pid=primary_pid,
        original_command=primary_command, resumed_utc=datetime.now(timezone.utc).isoformat(), signal='SIGCONT'), indent=2))

serial_seconds = sum(row['wall_seconds'] for row in phase_results if row['name'].startswith('single_'))
speedup = serial_seconds / parallel_seconds
result = dict(status='EQUAL_REAL_QUERY_SERIAL_VS_TWO_WORKER_BENCHMARK_COMPLETE',
              started_utc=start, observed_utc=datetime.now(timezone.utc).isoformat(), physical_gpu=2,
              primary_pid=primary_pid, original_primary_resumed=True, gpu3_unchanged=True,
              equal_queries_per_condition=10, equal_action_labels_per_condition=2100,
              serial_wall_seconds=serial_seconds, parallel_wall_seconds=parallel_seconds,
              aggregate_throughput_speedup=speedup, phases=phase_results,
              queries=queries, original_memory_used_mib=used, gpu_capacity_mib=capacity,
              recommendation='Two workers fit and improved this workload.' if speedup > 1.10 else 'Keep one worker; speedup insufficient on this workload.',
              full_collection_modified=False, new_optimizer_updates=0, formal_metrics=None,
              scope='Same five real TRAIN queries per worker, including initialization, full-prefix decoding and H3 batch64 labels. Short window; production tail still requires monitoring.')
(out / 'benchmark_complete.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
