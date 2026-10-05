"""Continue actual completed ABC fits through full-video selection and both full native reports."""
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_train_events'
SETUP = Path('/data/gb/setup')
PYTHON = '/data/gb/envs/gola/bin/python'
STATE = SETUP / 'train_events_complete_evaluation_progress_20261005.json'


def read(path):
    return json.loads(Path(path).read_text())


def record(stage, **details):
    value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'pid': os.getpid(), 'automatic_restarts': 0, 'system_goal_completed': False} | details
    STATE.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def wave(commands, label, env):
    observed = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                              check=True, capture_output=True, text=True).stdout
    used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in observed.splitlines()}
    assert set(used) == {0, 1, 2, 3} and all(value < 500 for value in used.values()), used
    children = []
    for gpu, command in commands:
        log = SETUP / ('train_events_' + label + '_gpu' + str(gpu) + '_20261005.log')
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env | {'CUDA_VISIBLE_DEVICES': str(gpu)},
                                       stdout=stream, stderr=subprocess.STDOUT)
        children.append((process, gpu, log))
    record(label + '_RUNNING', children=[{'pid': p.pid, 'gpu': gpu, 'log': str(log)} for p, gpu, log in children])
    for process, gpu, log in children:
        result = process.wait()
        assert result == 0, (gpu, result, str(log))


def retire(paths, label):
    deleted = []
    for path in paths:
        assert path.parent.resolve().parent == Path('/data/gb/outputs')
        assert path.parent.name.startswith('recoverability_train_events_') and path.suffix == '.pth'
        deleted.append({'path': str(path), 'bytes': path.stat().st_size})
        path.unlink()
    receipt = {'completed': True, 'deleted': deleted, 'teacher_or_pretrained_weights_deleted': False,
               'completed_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
    (SETUP / ('train_events_' + label + '_weight_cleanup_20261005.json')).write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    assert Path.cwd() == ROOT and os.environ['CUDA_VISIBLE_DEVICES'] == '' and not STATE.exists()
    review = read(FOLDER / 'complete_fit_recovery_source_review.json')
    assert review['status'] == 'PASS' and review['review_independence'] == 'same-family'
    assert review['acceptance_status'] == 'provisional' and review['runtime_attested'] is False
    plan = read(FOLDER / 'fit_plan.json')
    launch = read(FOLDER / 'actual_complete_fit_after_M0_launch.json')
    upstream = SETUP / 'train_events_complete_fit_progress_20261005.json'
    record('WAITING_FOR_ACTUAL_FOUR_COMPLETE420UPDATE_ABC_FITS', upstream_pid=launch['coordinator_pid'])
    while read(upstream)['stage'] != 'FOUR_COMPLETE_MATCHED420UPDATE_ABC_FITS_READY_FOR_FULL_VIDEO_SELECTION':
        result = subprocess.run(['ps', '-o', 'stat=', '-p', str(launch['coordinator_pid'])], capture_output=True, text=True)
        assert result.returncode == 0 and result.stdout.strip() and not result.stdout.strip().startswith('Z'), result
        time.sleep(180)
    accepted = read(SETUP / 'train_events_full_fit_cpu_acceptance_20261005.json')
    assert accepted['status'] == 'PASS' and accepted['all_four_ABC_fits_passed']
    assert all(row['optimizer_steps'] == 420 for row in accepted['arms'].values())
    m0 = read(SETUP / 'train_events_m0_fit_cpu_acceptance_20261005.json')
    assert m0['status'] == 'PASS' and m0['all_four_ABC_fits_passed']
    retire([Path(row['output']) / name for row in m0['arms'].values() for name in row['checkpoints']], 'm0')
    env = os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(ROOT), 'LD_LIBRARY_PATH': '/data/gb/envs/gola/lib',
                        'OMP_NUM_THREADS': '4', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'TORCH_HOME': '/data/gb/cache/torch',
                        'XDG_CACHE_HOME': '/data/gb/cache', 'TMPDIR': '/data/gb/cache/tmp', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}

    def cpu(name, *arguments):
        subprocess.run([PYTHON, '-u', str(FOLDER / name), *map(str, arguments)], cwd=ROOT, env=env, check=True)

    evaluated = []
    for slot in range(4):
        commands, pending = [], []
        for arm in plan['arms']:
            output = Path(arm['output_full'])
            name = f"epoch_{arm['retain_epochs'][slot]:03d}.pth" if slot < 3 else 'best.pth'
            info = accepted['arms'][arm['arm']]['checkpoints'][name]
            if info['epoch'] == 0 or any(row['arm'] == arm['arm'] and row['epoch'] == info['epoch'] for row in evaluated):
                continue
            model = output / name
            predictions = output / ('continuous_' + name[:-4]) / 'predictions'
            assert not predictions.parent.exists()
            command = [PYTHON, '-u', '-m', 'research.evaluate_recoverability', '--dataset', 'lasher',
                       '--root', '/data/wangwj/dataset/LasHeR/traingset', '--model', str(model),
                       '--validation-split', '/data/gb/outputs/c1_initial_seed42/split.json',
                       '--write-verification', 'action', '--seed', '42', '--max-frames', '0', '--output', str(predictions)]
            commands.append((arm['gpu'], command))
            pending.append({'arm': arm['arm'], 'model': str(model), 'predictions': str(predictions), 'epoch': info['epoch']})
        if commands:
            wave(commands, 'FULL98_SLOT' + str(slot), env)
            for row in pending:
                cpu('score_full98_cpu.py', '--predictions', row['predictions'], '--model', row['model'])
                score = read(Path(row['predictions']).parent / 'full98_selection_score.json')
                evaluated.append(row | {'sequence_mean_iou': score['sequence_mean_iou']})
    assert evaluated
    selected = max(evaluated, key=lambda row: row['sequence_mean_iou'])
    arm = next(row for row in plan['arms'] if row['arm'] == selected['arm'])
    selection = {'status': 'PASS', 'checkpoint': selected['model'], 'selected_best_epoch': selected['epoch'],
                 'all_four_full_fits_CPU_passed': True, 'completed_epochs': arm['full_epochs'], 'optimizer_steps': 420,
                 'train_queries': arm['train_queries'], 'selection_rule': 'Highest full98 continuous developer sequence mean IoU among trained positive epochs; declared order on ties; no native-test selection.',
                 'candidates': evaluated, 'both_datasets_same_fixed_checkpoint': True,
                 'parent_remains_historical_reference': arm['init_checkpoint'],
                 'selected_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
    (SETUP / 'train_events_native_selection_20261005.json').write_text(json.dumps(selection, indent=2) + '\n')
    retire([Path(row['output']) / name for row in accepted['arms'].values() for name in row['checkpoints']
            if str(Path(row['output']) / name) != selected['model']], 'unselected_full')
    record('FULL98_CONTINUOUS_SELECTION_COMPLETE', selection=selection)
    wave([(gpu, ['bash', str(FOLDER / 'run_native_complete.sh'), str(gpu)]) for gpu in range(4)],
         'BOTH_COMPLETE_NATIVE_BENCHMARKS', env)
    record('BOTH_NATIVE_INFERENCE_COMPLETE_ALL_METRICS_SCORING')
    for dataset in ('lasher', 'rgbt234'):
        cpu('merge_native_complete.py', '--dataset', dataset)
        subprocess.run(['bash', str(FOLDER / 'report_native_complete.sh'), dataset], cwd=ROOT, env=env, check=True)
        cpu('audit_native_complete_cpu.py', '--dataset', dataset)
    cpu('merge_complete_goal_report.py')
    report = Path('/data/gb/outputs/recoverability_train_events_native_full_20261005/complete_report/complete_core_report.json')
    result = read(report)
    assert result['completed'] and len(result['acceptance']['metrics']) == 5
    record('COMPLETE420_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY', complete_report=str(report),
           goal_result=result['goal_result'], acceptance=result['acceptance'])


if __name__ == '__main__':
    main()
