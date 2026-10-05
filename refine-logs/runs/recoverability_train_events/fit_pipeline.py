"""Wait for actual TRAIN GT acceptance, run capacity tests, then four complete ABC fits."""
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_train_events'
SETUP = Path('/data/gb/setup')
PYTHON = '/data/gb/envs/gola/bin/python'
STATE = SETUP / 'train_events_fit_progress_20261005.json'


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
    collection = read(FOLDER / 'actual_collection_launch.json')
    upstream_state = SETUP / 'train_events_collection_progress_20261005.json'
    record('WAITING_FOR_ACTUAL_NEW716_TRAIN_COLLECTION_AND_GT_AUDIT', upstream_pid=collection['coordinator_pid'])
    while read(upstream_state)['stage'] != 'NEW716_TRAIN_STATES_READY_FOR_CAPACITY_AND_COMPLETE_FIT':
        result = subprocess.run(['ps', '-o', 'stat=', '-p', str(collection['coordinator_pid'])],
                                capture_output=True, text=True)
        assert result.returncode == 0 and result.stdout.strip() and not result.stdout.strip().startswith('Z'), result
        time.sleep(180)
    acceptance = read(SETUP / 'train_events_full_cpu_acceptance_20261005.json')
    assert acceptance['status'] == 'PASS' and acceptance['new_clips'] == 716
    assert acceptance['all_queries_TRAIN_only'] and acceptance['base1762_and_validation196_unchanged']
    private = Path(plan['private_package'])
    assert not private.exists()
    shutil.copytree('/data/gb/experiments/recoverability_best_policy_fit_20261004/research', private / 'research',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / plan['private_trainer'], private / 'research/train_recoverability.py')
    env = os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(private) + ':' + str(ROOT),
                        'LD_LIBRARY_PATH': '/data/gb/envs/gola/lib', 'OMP_NUM_THREADS': '4',
                        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'TORCH_HOME': '/data/gb/cache/torch',
                        'XDG_CACHE_HOME': '/data/gb/cache', 'TMPDIR': '/data/gb/cache/tmp',
                        'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    for stage in ('m0', 'full'):
        gpu_output = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, check=True).stdout
        used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in gpu_output.splitlines()}
        assert set(used) == {0, 1, 2, 3} and all(value < 500 for value in used.values()), used
        children = []
        for arm in plan['arms']:
            output = arm['output_' + stage]
            assert not Path(output).exists()
            epochs = arm['m0_epochs'] if stage == 'm0' else arm['full_epochs']
            retained = [3] if stage == 'm0' else arm['retain_epochs']
            command = [PYTHON, '-u', '-m', 'research.train_recoverability', '--train', *arm['train'],
                       '--validation', arm['validation'], '--batch-size', str(arm['batch_size']), '--epochs', str(epochs),
                       '--retain-epochs', *map(str, retained), '--lr', str(arm['lr']), '--seed', str(arm['seed']),
                       '--search-supervision', arm['search_supervision'], '--action-ranking', arm['action_ranking'],
                       '--write-verification', arm['write_verification'], '--init-checkpoint', arm['init_checkpoint'],
                       '--output', output]
            log = SETUP / ('train_events_' + stage + '_gpu' + str(arm['gpu']) + '_20261005.log')
            with log.open('x') as stream:
                process = subprocess.Popen(command, cwd=private, env=env | {'CUDA_VISIBLE_DEVICES': str(arm['gpu'])},
                                           stdout=stream, stderr=subprocess.STDOUT)
            children.append((process, arm, log))
        record(stage.upper() + '_FOUR_GPU_ABC_TRAINING_RUNNING',
               children=[{'pid': process.pid, 'gpu': arm['gpu'], 'arm': arm['arm'], 'log': str(log),
                          'output': arm['output_' + stage], 'expected_epochs': arm[stage + '_epochs'],
                          'expected_optimizer_steps': arm[stage + '_optimizer_steps']}
                         for process, arm, log in children])
        for process, arm, log in children:
            returncode = process.wait()
            assert returncode == 0, (arm['arm'], returncode, str(log))
        record(stage.upper() + '_TRAINING_COMPLETE_CPU_RELOAD_AUDIT_RUNNING')
        with (SETUP / ('train_events_' + stage + '_fit_cpu_audit_20261005.log')).open('x') as stream:
            subprocess.run([PYTHON, '-u', str(FOLDER / 'audit_fit_cpu.py'), '--stage', stage],
                           cwd=private, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        accepted = read(SETUP / ('train_events_' + stage + '_fit_cpu_acceptance_20261005.json'))
        assert accepted['status'] == 'PASS' and accepted['all_four_ABC_fits_passed']
        record(stage.upper() + '_ALL_FOUR_ACTUAL_TRAINING_AND_CPU_RELOAD_AUDIT_PASS', acceptance=accepted)
    record('FOUR_COMPLETE_MATCHED420UPDATE_ABC_FITS_READY_FOR_FULL_VIDEO_SELECTION',
           next_stage='Full98 continuous checkpoint selection, then the same best trained checkpoint on all245 LasHeR and234 RGBT234 sequences. Native metrics remain pending.')


if __name__ == '__main__':
    main()
