"""Run the fixed TRAIN diagnostic on GPU3 released by the ongoing native wave."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_write_events'
BASE = Path('/data/gb/outputs/recoverability_unknown_geometry_train20_released_gpu3_20261006')
PYTHON = '/data/gb/envs/gola/bin/python'
GPU = 3


def read(path):
    return json.loads(path.read_text())


def record(stage, **details):
    value = {'stage': stage, 'pid': os.getpid(), 'gpu': GPU,
             'at_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), **details}
    (BASE / 'progress.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def wave(phase, jobs, cfg):
    output = BASE / phase / f'gpu{GPU}'
    jobs_file = BASE / f'{phase}_gpu{GPU}_jobs.json'
    jobs_file.write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
    command = [PYTHON, '-u', '-m', 'research.probe_geometry_commit']
    for key in ('root', 'cache', 'split', 'model', 'pretrained', 'c1_head', 'motion_run'):
        command += ['--' + key.replace('_', '-'), cfg[key]]
    command += ['--jobs-file', str(jobs_file), '--output', str(output), '--horizons', '3', '32',
                '--write-verification', 'action', '--seed', '42', '--geometry-reference', 'C1_components']
    env = os.environ | {'CUDA_VISIBLE_DEVICES': str(GPU), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                        'LD_LIBRARY_PATH': '/data/gb/envs/gola/lib', 'PYTHONPATH': str(ROOT),
                        'TORCH_HOME': '/data/gb/cache/torch', 'XDG_CACHE_HOME': '/data/gb/cache',
                        'TMPDIR': '/data/gb/cache/tmp', 'OMP_NUM_THREADS': '4', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    log_path = BASE / f'{phase}_gpu{GPU}.log'
    with log_path.open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record(phase.upper() + '_RUNNING', child_pid=child.pid, events=len(jobs), command=command,
           started_epoch=time.time(), output=str(output), log=str(log_path))
    code = child.wait()
    record(phase.upper() + '_CHILD_TERMINAL', child_pid=child.pid, events=len(jobs), exit_code=code)
    assert code == 0, 'Inspect preserved child log before any retry.'
    subprocess.run([PYTHON, '-u', str(FOLDER / 'validate_unknown_probe_cpu.py'), '--base', str(BASE),
                    '--phases', phase, '--gpus', str(GPU)], cwd=ROOT,
                   env=os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(ROOT)}, check=True)


def main():
    assert Path.cwd() == ROOT and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name in ('unknown_probe_source_review.json', 'unknown_released_gpu_source_review.json', 'unknown_released_gpu_rescue_source_review.json'):
        assert read(FOLDER / name)['status'] == 'PASS'
    native = read(Path('/data/gb/setup/write_events_complete_evaluation_progress_20261005.json'))
    assert native['pid'] == 2914365 and native['stage'] == 'BOTH_COMPLETE_NATIVE_CRLF_RESUME_RUNNING'
    owner = next(row for row in native['children'] if row['gpu'] == GPU)
    owner_proc = Path('/proc') / str(owner['pid'])
    assert not owner_proc.exists() or (owner_proc / 'stat').read_text().split(') ', 1)[1].split()[0] == 'Z'
    native_base = Path('/data/gb/outputs/recoverability_write_events_native_full_20261005')
    for dataset in ('lasher', 'rgbt234'):
        shard = native_base / dataset / 'shards' / f'gpu{GPU}'
        assert (shard / 'inference_completed.txt').is_file()
        assert read(shard / 'predictions' / 'inference_completion.json')['completed']
    selection = read(Path('/data/gb/setup/write_events_native_selection_20261005.json'))
    plan = read(FOLDER / 'unknown_probe_plan.json')
    assert selection['status'] == 'PASS' and selection['checkpoint'] == plan['model']
    assert len(selection['candidates']) == 16 and Path(plan['model']).is_file()
    rows = read(FOLDER / 'actual_TRAIN_unknown_probe_jobs_cpu.json')['jobs']
    first = next(row for row in rows if row['event_id'] == plan['gpu_assignment'][GPU]['events'][0])
    jobs = [first] + [row for row in rows if row['event_id'] != first['event_id']]
    assert len(jobs) == len({row['event_id'] for row in jobs}) == 20
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    row = next(line.split(', ') for line in devices.splitlines() if int(line.split(', ')[0]) == GPU)
    assert float(row[2]) > 20000 and not any(line.split(', ')[0] == row[1] for line in apps.splitlines())
    cfg = read(ROOT / 'refine-logs/runs/recoverability_state_commit/matched_velocity_train88_old4/gpu0/config.json')
    cfg['model'] = plan['model']
    assert not BASE.exists()
    BASE.mkdir()
    record('RELEASED_GPU3_TRAIN_DIAGNOSTIC_STARTED', native_pid=2914365, model=cfg['model'], events=20,
           scope='Existing TRAIN events and frozen checkpoint; concurrent native GPU0/1/2 untouched, no native result used')
    wave('sanity', jobs[:1], cfg)
    record('ONE_ACTUAL_MASKED_SANITY_EVENT_CPU_PASS')
    wave('remaining', jobs[1:], cfg)
    subprocess.run([PYTHON, '-u', str(FOLDER / 'validate_unknown_probe_cpu.py'), '--base', str(BASE),
                    '--phases', 'sanity', 'remaining', '--gpus', str(GPU)], cwd=ROOT,
                   env=os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(ROOT)}, check=True)
    record('ALL20_MASKED_TRAIN_EVENTS_COMPLETE_CPU_PASS', report=str(BASE / 'sanity_remaining_CPU.json'), optimizer_steps=0)
    (BASE / 'COMPLETE').write_text('PASS\n')


if __name__ == '__main__':
    main()
