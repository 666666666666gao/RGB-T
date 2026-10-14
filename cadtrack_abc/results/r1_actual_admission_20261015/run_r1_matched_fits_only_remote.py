from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

project = Path('/home/gaob/CADTrack-ABC')
sys.path.insert(0, str(project))
from cadtrack_abc.future_pipeline import validate_collection, validate_parent, validate_sampling

config_path = project / 'cadtrack_abc/config.json'
config = json.loads(config_path.read_text())
assert config['physical_gpus'] == [2, 3] and config['seed'] == 42
parent = Path(config['output'])
teacher, _ = validate_parent(parent)
teacher_sha = hashlib.sha256(teacher.read_bytes()).hexdigest()
manifest_path = parent / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
sampling = project / 'outputs/native_future_sampling_20261014'
collection = project / 'outputs/native_action_collection_20261014'
launch = json.loads((collection / 'launch.json').read_text())
assert all(not (Path('/proc') / str(job['pid'])).exists() for job in launch['jobs'])
for split in ['train', 'validation']:
    allowed = set(manifest['train_sequences' if split == 'train' else 'validation_sequences'])
    validate_sampling(sampling / f'queries_{split}', allowed, teacher_sha)
    folders = [collection / f'collection_{split}' / f'shard{shard}' for shard in range(2)]
    validate_collection(folders, sampling / f'queries_{split}', teacher_sha)
baseline = json.loads((project / 'outputs/pure_cad_rgbt234_20261015/pipeline_complete.json').read_text())
assert baseline['completed'] and baseline['inference_exit_codes'] == [0, 0] and baseline['score_exit_code'] == 0
sanity = json.loads((project / 'outputs/search_action_visual_sanity_20261014/sanity_complete.json').read_text())
assert sanity['status'] == 'ACTUAL_GPU_NEUTRAL_AND_PROBABILITY_RANK1_VISUAL_STATE_SANITY_PASS'
memory = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=index,memory.used',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
assert all(int(line.split(',')[1]) < 500 for line in memory.splitlines()), memory
output = project / 'outputs/r1_matched_fits_20261015'
output.mkdir(exist_ok=False)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
           OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
indices = {}
for split in ['train', 'validation']:
    indices[split] = output / f'{split}_index.json'
    folders = [collection / f'collection_{split}' / f'shard{shard}' for shard in range(2)]
    command = [sys.executable, '-m', 'cadtrack_abc.merge_action_packets', '--collections',
        *map(str, folders), '--manifest', str(manifest_path), '--split', split,
        '--output', str(indices[split])]
    with (output / f'merge_{split}.log').open('w') as stream:
        subprocess.run(command, cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
jobs = []
for policy, gpu in [('future_actions', 2), ('search_actions', 3)]:
    command = [sys.executable, '-m', 'cadtrack_abc.train_future_actions', '--config', str(config_path),
        '--teacher', str(teacher), '--manifest', str(manifest_path),
        '--train-index', str(indices['train']), '--validation-index', str(indices['validation']),
        '--output', str(output / f'fit_{policy}'), '--epochs', '20', '--batch', '64',
        '--lr', '0.0001', '--workers', '4', '--device', 'cuda', '--policy', policy]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (output / f'fit_{policy}.log').open('w') as stream:
        child = subprocess.Popen(command, cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT)
    jobs.append(dict(child=child, pid=child.pid, physical_gpu=gpu, policy=policy, command=command))
state = dict(status='MATCHED_FULL_FITS_RUNNING', started_utc=datetime.now(timezone.utc).isoformat(),
    teacher=str(teacher), teacher_sha256=teacher_sha,
    jobs=[{key: value for key, value in job.items() if key != 'child'} for job in jobs],
    existing_labels_reused=True, deferred_dependency='RGBT210 download/inventory is needed for official evaluation, not these fits.',
    future_full_evaluation_required=True, overall_goal_completed=False)
(output / 'launch.json').write_text(json.dumps(state, indent=2))
codes = [job['child'].wait() for job in jobs]
(output / 'child_exits.json').write_text(json.dumps(dict(return_codes=codes,
    observed_utc=datetime.now(timezone.utc).isoformat()), indent=2))
assert codes == [0, 0], codes
queries = json.loads(indices['train'].read_text())['queries']
fits = {}
for job in jobs:
    receipt = json.loads((output / f"fit_{job['policy']}" / 'fit_complete.json').read_text())
    assert receipt['epochs'] == 20 and receipt['updates'] == ((queries + 63) // 64) * 20
    assert receipt['policy_family'] == job['policy'] and receipt['teacher_sha256'] == teacher_sha
    assert receipt['train_queries'] == queries and receipt['train_sequences'] == 881
    assert receipt['strict_head_reload'] and receipt['changed_head_tensors']
    fits[job['policy']] = receipt
state.update(status='MATCHED_FULL_FITS_COMPLETE_FULL_VIDEO_SELECTION_AND_FORMAL_EVALUATION_PENDING',
             completed_utc=datetime.now(timezone.utc).isoformat(), return_codes=codes, fits=fits)
(output / 'fits_complete.json').write_text(json.dumps(state, indent=2))
print(json.dumps(state), flush=True)
