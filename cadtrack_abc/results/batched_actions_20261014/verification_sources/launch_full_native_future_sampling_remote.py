from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

project = Path('/home/gaob/CADTrack-ABC')
parent = project / 'outputs/cad_abc_20261013'
sys.path.insert(0, str(project))
from cadtrack_abc.future_pipeline import validate_parent
teacher, parent_report = validate_parent(parent)
teacher_sha = hashlib.sha256(teacher.read_bytes()).hexdigest()
assert teacher_sha == '5c6b6fcbf18b11918ae3495ccfa927cf56a3f5c7f3cecd76aabaaa94d80cc988'
gpu_memory = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=index,memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
assert all(int(row.split(',')[1]) < 500 for row in gpu_memory.splitlines()), gpu_memory
output = project / 'outputs/native_future_sampling_20261014'
output.mkdir(exist_ok=False)
python = '/data/wangwj/envs/CADTrack/bin/python'
manifest = parent / 'manifest.json'
inventory = json.loads(manifest.read_text())
assert len(inventory['train_sequences']) == 881 and len(inventory['validation_sequences']) == 98
records = []
for shard, physical_gpu in enumerate([2, 3]):
    commands = []
    for split in ['train', 'validation']:
        commands.append([python, '-m', 'cadtrack_abc.sample_action_queries', '--config', str(project / 'cadtrack_abc/config.json'),
                         '--teacher', str(teacher), '--teacher-policy', 'native', '--manifest', str(manifest), '--split', split,
                         '--output', str(output / ('queries_' + split)), '--device', 'cuda', '--horizon', '3', '--shard', str(shard)])
    driver = output / ('driver_shard' + str(shard) + '.py')
    driver.write_text('import subprocess\n' + '\n'.join(f'subprocess.run({command!r},cwd={str(project)!r},check=True)' for command in commands) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(physical_gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    log = output / ('sampling_shard' + str(shard) + '.log')
    with log.open('w') as stream:
        process = subprocess.Popen([python, str(driver)], cwd=project, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    records.append({'pid': process.pid, 'physical_gpu': physical_gpu, 'shard': shard,
                    'commands': commands, 'log': str(log),
                    'actual_cmdline': Path('/proc/' + str(process.pid) + '/cmdline').read_bytes().replace(b'\x00', b' ').decode()})
receipt = {'status': 'FULL_NATIVE_TEACHER_TRAJECTORY_SAMPLING_LAUNCHED', 'started_utc': datetime.now(timezone.utc).isoformat(),
           'jobs': records, 'teacher': str(teacher), 'teacher_sha256': teacher_sha,
           'train_sequences': 881, 'train_frames': 461883, 'development_sequences': 98, 'development_frames': 52200,
           'sampled_frame_inventory': [244684, 269399], 'physical_gpus': [2, 3],
           'estimated_hours': [7.5, 9.], 'native_formal_results_unchanged': True,
           'training_updates': 0, 'action_label_collection_started': False, 'future_head_fit_started': False,
           'sampling_policy_batch': 1, 'batch_action_collector_qualification_required_for_this_stage': False,
           'parent_full_lasher_metrics': parent_report['overall_metrics_percent'],
           'official_test_used': False, 'baseline_rerun': False, 'launch_gpu_memory_only': gpu_memory,
           'source_snapshot_sha256': {name: hashlib.sha256((project / 'cadtrack_abc' / name).read_bytes()).hexdigest()
                                    for name in ['sample_action_queries.py', 'model.py', 'ported_modules.py', 'baseline.py']},
           'scope': 'Full881TRAIN/98DEV causal teacher trajectories and event queries; optimizer fitting follows later. No automatic next training.'}
(output / 'launch.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
