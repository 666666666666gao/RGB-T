"""Run the previously unstarted complete fits after actual collection and M0 acceptance."""
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_train_events'
SETUP = Path('/data/gb/setup')
PYTHON = '/data/gb/envs/gola/bin/python'
STATE = SETUP / 'train_events_complete_fit_progress_20261005.json'


def read(path):
    return json.loads(Path(path).read_text())


def record(stage, **details):
    value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'pid': os.getpid(), 'automatic_restarts': 0, 'official_metrics_completed': False,
             'collection_repeated': False, 'capacity_training_repeated': False} | details
    STATE.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def main():
    assert Path.cwd() == ROOT and os.environ['CUDA_VISIBLE_DEVICES'] == '' and not STATE.exists()
    review = read(FOLDER / 'complete_fit_recovery_source_review.json')
    assert review['status'] == 'PASS' and review['review_independence'] == 'same-family'
    assert review['acceptance_status'] == 'provisional' and review['runtime_attested'] is False
    collection = read(SETUP / 'train_events_full_cpu_acceptance_20261005.json')
    assert collection['status'] == 'PASS' and collection['new_clips'] == 716
    assert collection['all_queries_TRAIN_only'] and collection['base1762_and_validation196_unchanged']
    capacity = read(SETUP / 'train_events_m0_fit_cpu_acceptance_20261005.json')
    assert capacity['status'] == 'PASS' and capacity['all_four_ABC_fits_passed']
    failure = read(FOLDER / 'actual_after_M0_controller_failure_intake.json')
    assert failure['original_owner_ps_returncode'] == 1
    assert all(not row['output_exists'] for row in failure['full_launch_artifacts'])
    assert 'FileExistsError' in '\n'.join(failure['tracebacks']['train_events_fit_controller_20261005.log'])
    plan = read(FOLDER / 'fit_plan.json')
    private = Path(plan['private_package'])
    assert private.is_dir() and not (SETUP / 'train_events_full_fit_cpu_acceptance_20261005.json').exists()
    assert all(not Path(arm['output_full']).exists() for arm in plan['arms'])
    env = os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(private) + ':' + str(ROOT),
                        'LD_LIBRARY_PATH': '/data/gb/envs/gola/lib', 'OMP_NUM_THREADS': '4',
                        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'TORCH_HOME': '/data/gb/cache/torch',
                        'XDG_CACHE_HOME': '/data/gb/cache', 'TMPDIR': '/data/gb/cache/tmp',
                        'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    used_output = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                                 check=True, capture_output=True, text=True).stdout
    used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in used_output.splitlines()}
    assert set(used) == {0, 1, 2, 3} and all(value < 500 for value in used.values()), used
    children = []
    for arm in plan['arms']:
        command = [PYTHON, '-u', '-m', 'research.train_recoverability', '--train', *arm['train'],
                   '--validation', arm['validation'], '--batch-size', str(arm['batch_size']),
                   '--epochs', str(arm['full_epochs']), '--retain-epochs', *map(str, arm['retain_epochs']),
                   '--lr', str(arm['lr']), '--seed', str(arm['seed']), '--search-supervision', arm['search_supervision'],
                   '--action-ranking', arm['action_ranking'], '--write-verification', arm['write_verification'],
                   '--init-checkpoint', arm['init_checkpoint'], '--output', arm['output_full']]
        log = SETUP / ('train_events_fit_full_gpu' + str(arm['gpu']) + '_20261005.log')
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=private, env=env | {'CUDA_VISIBLE_DEVICES': str(arm['gpu'])},
                                       stdout=stream, stderr=subprocess.STDOUT)
        children.append((process, arm, log))
    record('FULL_FOUR_GPU_ABC_TRAINING_RUNNING',
           children=[{'pid': process.pid, 'gpu': arm['gpu'], 'arm': arm['arm'], 'log': str(log),
                      'output': arm['output_full'], 'expected_epochs': arm['full_epochs'],
                      'expected_optimizer_steps': arm['full_optimizer_steps']}
                     for process, arm, log in children])
    for process, arm, log in children:
        result = process.wait()
        assert result == 0, (arm['arm'], result, str(log))
    record('FULL_TRAINING_COMPLETE_CPU_RELOAD_AUDIT_RUNNING')
    with (SETUP / 'train_events_complete_fit_cpu_audit_20261005.log').open('x') as stream:
        subprocess.run([PYTHON, '-u', str(FOLDER / 'audit_fit_cpu.py'), '--stage', 'full'],
                       cwd=private, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    accepted = read(SETUP / 'train_events_full_fit_cpu_acceptance_20261005.json')
    assert accepted['status'] == 'PASS' and accepted['all_four_ABC_fits_passed']
    record('FOUR_COMPLETE_MATCHED420UPDATE_ABC_FITS_READY_FOR_FULL_VIDEO_SELECTION', acceptance=accepted)


if __name__ == '__main__':
    main()
