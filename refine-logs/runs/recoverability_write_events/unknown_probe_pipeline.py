"""Four-card TRAIN GT-gap diagnostic, after both complete native reports."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_write_events'
BASE = Path('/data/gb/outputs/recoverability_unknown_geometry_train20_20261006')
PYTHON = '/data/gb/envs/gola/bin/python'


def read(path):
    return json.loads(path.read_text())


def record(stage, **details):
    value = {'stage': stage, 'pid': os.getpid(),
             'at_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), **details}
    (BASE / 'progress.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def wave(phase, jobs, cfg):
    children = []
    for gpu in range(4):
        output = BASE / phase / f'gpu{gpu}'
        jobs_file = BASE / f'{phase}_gpu{gpu}_jobs.json'
        jobs_file.write_text(json.dumps({'jobs': jobs[gpu]}, indent=2) + '\n')
        command = [PYTHON, '-u', '-m', 'research.probe_geometry_commit']
        for key in ('root', 'cache', 'split', 'model', 'pretrained', 'c1_head', 'motion_run'):
            command += ['--' + key.replace('_', '-'), cfg[key]]
        command += ['--jobs-file', str(jobs_file), '--output', str(output), '--horizons', '3', '32',
                    '--write-verification', 'action', '--seed', '42', '--geometry-reference', 'C1_components']
        env = os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                            'LD_LIBRARY_PATH': '/data/gb/envs/gola/lib', 'PYTHONPATH': str(ROOT),
                            'TORCH_HOME': '/data/gb/cache/torch', 'XDG_CACHE_HOME': '/data/gb/cache',
                            'TMPDIR': '/data/gb/cache/tmp', 'OMP_NUM_THREADS': '4', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        log_path = BASE / f'{phase}_gpu{gpu}.log'
        with log_path.open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((child, {'gpu': gpu, 'pid': child.pid, 'events': len(jobs[gpu]),
                                  'command': command, 'output': str(output), 'log': str(log_path)}))
    record(phase.upper() + '_RUNNING', started_epoch=time.time(), children=[row for _, row in children])
    codes = [child.wait() for child, _ in children]
    record(phase.upper() + '_CHILDREN_TERMINAL', children=[row | {'exit_code': code}
           for (_, row), code in zip(children, codes)])
    assert codes == [0] * 4, 'Inspect preserved child logs before any retry.'
    subprocess.run([PYTHON, '-u', str(FOLDER / 'validate_unknown_probe_cpu.py'), '--base', str(BASE),
                    '--phases', phase], cwd=ROOT,
                   env=os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(ROOT)}, check=True)


def main():
    assert Path.cwd() == ROOT and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name in ('unknown_probe_source_review.json', 'unknown_pipeline_source_review.json'):
        assert read(FOLDER / name)['status'] == 'PASS'
    native = read(Path('/data/gb/setup/write_events_complete_evaluation_progress_20261005.json'))
    assert native['stage'] == 'COMPLETE480_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY'
    assert not (Path('/proc') / str(native['pid'])).exists()
    report = read(Path(native['complete_report']))
    assert report['completed'] and len(report['acceptance']['metrics']) == 5
    assert all(row['independent_native_CPU']['status'] == 'PASS' for row in report['datasets'].values())
    plan = read(FOLDER / 'unknown_probe_plan.json')
    rows = read(FOLDER / 'actual_TRAIN_unknown_probe_jobs_cpu.json')['jobs']
    lookup = {row['event_id']: row for row in rows}
    assert len(lookup) == len(rows) == 20
    assignments = [[lookup[event] for event in row['events']] for row in plan['gpu_assignment']]
    assert [row['gpu'] for row in plan['gpu_assignment']] == list(range(4))
    assert sorted(row['event_id'] for group in assignments for row in group) == sorted(lookup)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    for line in devices.splitlines():
        index, uuid, free = line.split(', ')
        if int(index) in range(4):
            assert float(free) > 20000 and not any(row.split(', ')[0] == uuid for row in apps.splitlines())
    cfg = read(ROOT / 'refine-logs/runs/recoverability_state_commit/matched_velocity_train88_old4/gpu0/config.json')
    cfg['model'] = plan['model']
    assert cfg['model'] == report['selected_checkpoint']['checkpoint'] and Path(cfg['model']).is_file()
    assert not BASE.exists()
    BASE.mkdir()
    record('NATIVE_REPORTS_COMPLETE_DIAGNOSTIC_STARTED', model=cfg['model'], events=20,
           optimizer_steps=0, scope='TRAIN matched prestate geometry references; no native score or learned module')
    wave('sanity', [[group[0]] for group in assignments], cfg)
    record('FOUR_SANITY_EVENTS_CPU_PASS')
    wave('remaining', [group[1:] for group in assignments], cfg)
    subprocess.run([PYTHON, '-u', str(FOLDER / 'validate_unknown_probe_cpu.py'), '--base', str(BASE),
                   '--phases', 'sanity', 'remaining'], cwd=ROOT,
                   env=os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(ROOT)}, check=True)
    record('ALL20_MASKED_TRAIN_EVENTS_COMPLETE_CPU_PASS', report=str(BASE / 'sanity_remaining_CPU.json'), optimizer_steps=0)
    (BASE / 'COMPLETE').write_text('PASS\n')


if __name__ == '__main__':
    main()
