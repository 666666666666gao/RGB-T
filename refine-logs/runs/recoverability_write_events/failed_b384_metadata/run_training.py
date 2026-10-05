"""Run actual four-card M0, CPU reload, then complete 420-update ABC fits."""
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_write_events'
SETUP = Path('/data/gb/setup')
PYTHON = '/data/gb/envs/gola/bin/python'
STATE = SETUP / 'write_events_fit_progress_20261005.json'


def read(path):
    return json.loads(Path(path).read_text())


def record(stage, **details):
    value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'pid': os.getpid(), 'automatic_restarts': 0, 'official_metrics_completed': False} | details
    STATE.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def main():
    assert Path.cwd() == ROOT and os.environ['CUDA_VISIBLE_DEVICES'] == '' and not STATE.exists()
    review = read(FOLDER / 'fit_source_review.json')
    assert review['status'] == 'PASS' and review['review_independence'] == 'same-family'
    assert review['acceptance_status'] == 'provisional' and review['runtime_attested'] is False
    plan = read(FOLDER / 'fit_plan.json')
    collection = read(plan['collection_acceptance'])
    assert collection['status'] == 'PASS' and collection['new_queries'] == 98
    assert collection['all_queries_TRAIN_only'] and collection['combined_train_queries'] == 2576
    assert collection['prior2478_and_validation196_unchanged']
    private = Path(plan['private_package'])
    assert not private.exists()
    shutil.copytree(plan['source_package'], private / 'research', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / plan['private_trainer'], private / 'research/train_recoverability.py')
    env = os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(private) + ':' + str(ROOT),
                        'LD_LIBRARY_PATH': '/data/gb/envs/gola/lib', 'OMP_NUM_THREADS': '4',
                        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'TORCH_HOME': '/data/gb/cache/torch',
                        'XDG_CACHE_HOME': '/data/gb/cache', 'TMPDIR': '/data/gb/cache/tmp',
                        'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    for stage in ('m0', 'full'):
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
        rows = [line.split(', ') for line in devices.splitlines()]
        assert {int(row[0]) for row in rows} == {0, 1, 2, 3}
        assert all(float(row[2]) > 20000 and not any(line.split(', ')[0] == row[1] and 'python' in line.lower()
                   for line in apps.splitlines()) for row in rows), (devices, apps)
        assert shutil.disk_usage('/data/gb').free > 10 * 2**30
        children = []
        for arm in plan['arms']:
            output = arm['output_' + stage]
            assert not Path(output).exists()
            epochs = arm[stage + '_epochs']
            retained = [3] if stage == 'm0' else arm['retain_epochs']
            command = [PYTHON, '-u', '-m', 'research.train_recoverability', '--train', *arm['train'],
                       '--validation', arm['validation'], '--batch-size', str(arm['batch_size']), '--epochs', str(epochs),
                       '--retain-epochs', *map(str, retained), '--lr', str(arm['lr']), '--seed', str(arm['seed']),
                       '--search-supervision', arm['search_supervision'], '--action-ranking', arm['action_ranking'],
                       '--write-verification', arm['write_verification'], '--init-checkpoint', arm['init_checkpoint'],
                       '--output', output]
            log = SETUP / f"write_events_{stage}_gpu{arm['gpu']}_20261005.log"
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=private, env=env | {'CUDA_VISIBLE_DEVICES': str(arm['gpu'])},
                                         stdout=stream, stderr=subprocess.STDOUT)
            children.append((child, arm, log))
        record(stage.upper() + '_FOUR_GPU_ABC_TRAINING_RUNNING', children=[
            {'pid': child.pid, 'gpu': arm['gpu'], 'arm': arm['arm'], 'log': str(log), 'output': arm['output_' + stage],
             'expected_epochs': arm[stage + '_epochs'], 'expected_optimizer_steps': arm[stage + '_optimizer_steps']}
            for child, arm, log in children])
        for child, arm, log in children:
            code = child.wait()
            assert code == 0, (arm['arm'], code, str(log))
        record(stage.upper() + '_TRAINING_COMPLETE_CPU_RELOAD_AUDIT_RUNNING')
        with (SETUP / f'write_events_{stage}_fit_cpu_audit_20261005.log').open('x') as stream:
            subprocess.run([PYTHON, '-u', str(FOLDER / 'audit_fit_cpu.py'), '--stage', stage],
                           cwd=private, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        accepted = read(SETUP / f'write_events_{stage}_fit_cpu_acceptance_20261005.json')
        assert accepted['status'] == 'PASS' and accepted['all_four_ABC_fits_passed']
        record(stage.upper() + '_ALL_FOUR_ACTUAL_TRAINING_AND_CPU_RELOAD_AUDIT_PASS', acceptance=accepted)
    record('FOUR_COMPLETE_MATCHED420UPDATE_ABC_FITS_READY_FOR_FULL_VIDEO_SELECTION',
           next_stage='Full98 continuous developer selection, same selected trained checkpoint on both native datasets; official metrics pending.')


if __name__ == '__main__':
    main()
