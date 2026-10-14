"""One real-prefix sanity, then a distinct full baseline beside current collectors."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import os
import subprocess
import time

project = Path('/home/gaob/CADTrack-ABC')
python = '/data/wangwj/envs/CADTrack/bin/python'
output = project / 'outputs/pure_cad_rgbt234_20261015'
assert not output.exists()
assert not (project / 'outputs/pure_baseline_protocol_sanity_20261015_r2').exists()
assert len([path for path in (project / 'dataset/rgbt234').iterdir() if path.is_dir()]) == 234
memory = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=index,memory.used,memory.total',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
assert all(int(row[2]) - int(row[1]) >= 12000 for row in csv.reader(io.StringIO(memory))), memory
processes = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-compute-apps=pid,gpu_uuid',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
existing = []
for row in csv.reader(io.StringIO(processes)):
    pid = int(row[0])
    command = Path('/proc', str(pid), 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert 'cadtrack_abc.collect_actions' in command and str(project) in command, command
    existing.append(dict(pid=pid, gpu_uuid=row[1].strip(), command=command))
assert len(existing) == 2 and len({row['gpu_uuid'] for row in existing}) == 2
env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='4',
           OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sanity_command = [python, str(project / 'tools/sanity_pure_cad_native_protocol_r2_remote.py')]
with (project / 'outputs/pure_baseline_protocol_sanity_20261015_r2.log').open('w') as stream:
    sanity_child = subprocess.run(sanity_command, cwd=project, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=True)
sanity = json.loads((project / 'outputs/pure_baseline_protocol_sanity_20261015_r2/sanity.json').read_text())
assert sanity['status'] == 'REAL_TRAIN_NATIVE_PROTOCOL_PARITY_PASS'
assert sanity['checkpoint_sha256'] == 'cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731'
command = [python, '-m', 'cadtrack_abc.baseline_pipeline',
    '--checkpoint', '/home/gaob/CADTrack_ep0014.pth.tar', '--dataset', 'rgbt234',
    '--root', str(project / 'dataset/rgbt234'), '--output', str(output)]
controller_env = dict(env, CUDA_VISIBLE_DEVICES='')
with (project / 'outputs/pure_cad_rgbt234_controller_20261015.log').open('w') as stream:
    controller = subprocess.Popen(command, cwd=project, env=controller_env, stdout=stream,
        stderr=subprocess.STDOUT, start_new_session=True)
time.sleep(15)
assert controller.poll() is None, controller.returncode
launch = json.loads((output / 'launch.json').read_text())
for job in launch['jobs']:
    command_actual = Path('/proc', str(job['pid']), 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert 'cadtrack_abc.infer_baseline' in command_actual and str(output) in command_actual
after = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
result = dict(status='PURE_CAD_RGBT234_FULL_BASELINE_ADMITTED',
    observed_utc=datetime.now(timezone.utc).isoformat(), controller_pid=controller.pid,
    command=command, jobs=launch['jobs'], existing_collectors=existing,
    sanity_exit_code=sanity_child.returncode, sanity=sanity,
    initial_gpu_memory_csv=memory, admission_gpu_memory_utilization_csv=after,
    maximum_neural_jobs_per_card=2, original_collectors_modified=False,
    baseline_lasher_rerun=False, expected_sequences=234, expected_frames=116649,
    estimated_duration_minutes=[45, 60],
    estimate_basis='Previously completed native enhanced parent on same dataset and concurrent cards; approximate, pure baseline has fewer decision operations.',
    formal_scores=None, new_optimizer_updates=0)
(output / 'admission.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
