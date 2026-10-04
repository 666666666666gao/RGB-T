"""Finish current collection, full ABC training, continuous selection and both complete benchmarks."""
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

setup = Path('/data/gb/setup')
main = Path('/data/gb/GOLA')
outputs = Path('/data/gb/outputs')
python = '/data/gb/envs/gola/bin/python'
private = Path('/data/gb/experiments/recoverability_best_native_policy_fit_20261005')
cache = outputs / 'recoverability_best_native_policy_merged_20261005/own'
teacher = outputs / 'recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
read = lambda p: json.loads(Path(p).read_text())
review = read(setup / 'best_native_policy_pipeline_source_review_20261005.json')
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
assert review['deployment_approved']
env = os.environ | {'CUDA_VISIBLE_DEVICES': '', 'LD_LIBRARY_PATH': '/data/gb/envs/gola/lib',
                    'PYTHONPATH': str(main), 'OMP_NUM_THREADS': '4', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                    'TORCH_HOME': '/data/gb/cache/torch', 'XDG_CACHE_HOME': '/data/gb/cache',
                    'TMPDIR': '/data/gb/cache/tmp', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
state = setup / 'best_native_policy_pipeline_progress_20261005.json'
assert not state.exists()
arms = [('pairwise_lr4', 'pairwise', 1e-4), ('pairwise_lr5', 'pairwise', 1e-5),
        ('budgeted_lr4', 'budgeted', 1e-4), ('budgeted_lr5', 'budgeted', 1e-5)]


def progress(stage, **fields):
    row = {'stage': stage, 'observed_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
           'automatic_restarts': 0, 'poll_interval_seconds': 180, 'official_goal_completed': False} | fields
    state.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row), flush=True)


def cpu(script, *args, cwd=main):
    subprocess.run([python, '-u', str(setup / script), *map(str, args)], cwd=cwd,
                   env=env | {'PYTHONPATH': str(cwd) + ':' + str(main)}, check=True)


def wave(commands, label):
    observed = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True).stdout
    used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in observed.splitlines()}
    assert set(used) == {0, 1, 2, 3} and all(v < 500 for v in used.values()), used
    live, rows = [], []
    for gpu, command, cwd in commands:
        log = setup / f'best_native_policy_{label}_gpu{gpu}_20261005.log'
        with log.open('wb') as stream:
            p = subprocess.Popen(command, cwd=cwd, env=env | {'CUDA_VISIBLE_DEVICES': str(gpu),
                                  'PYTHONPATH': str(cwd) + ':' + str(main)},
                                  stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        live.append(p)
        rows.append({'gpu': gpu, 'pid': p.pid, 'log': str(log), 'command': list(map(str, command))})
    progress(label, runs=rows, GPU_memory_before_launch_mib=used)
    while any(p.poll() is None for p in live):
        assert all(p.poll() in (None, 0) for p in live), [(p.pid, p.poll()) for p in live]
        time.sleep(180)
    assert all(p.returncode == 0 for p in live), [(p.pid, p.returncode) for p in live]


progress('WAIT_CURRENT_FOUR_COLLECTION_QUEUES', first_completion_check_cst='2026-10-05T03:00:00+08:00')
deadline = datetime(2026, 10, 5, 3, 0, tzinfo=timezone(timedelta(hours=8))).timestamp()
time.sleep(max(0, deadline - time.time()))
collection = read(setup / 'best_native_policy_full_launch_20261005.json')
groups = [[outputs / f'recoverability_best_native_policy_collect_{part}_gpu{gpu}_20261005/job_completed.txt'
           for part in ('train', 'validation')] for gpu in range(4)]
while not all(p.is_file() for group in groups for p in group):
    for row, markers in zip(collection['runs'], groups):
        if all(p.is_file() for p in markers):
            continue
        stat = subprocess.run(['ps', '-o', 'stat=', '-p', str(row['runner_pid'])],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert stat and not stat.startswith('Z'), row
    time.sleep(180)
progress('COLLECTION_COMPLETE_ACTUAL_GT_CPU_MERGE')
for part in ('train', 'validation'):
    cpu('best_native_policy_merge_partition_cpu_20261005.py', '--partition', part)
cpu('best_native_policy_cache_gate_cpu_20261005.py')

# Reuse the accepted isolated training package; production and live collection sources stay intact.
assert not private.exists()
shutil.copytree('/data/gb/experiments/recoverability_best_policy_fit_20261004/research', private / 'research',
                ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(setup / 'best_native_policy_train_fixed_epochs_20261005.py', private / 'research/train_recoverability.py')


def fits(stage):
    epochs, retained = (3, [3]) if stage == 'm0' else (60, [20, 40, 60])
    commands = []
    for gpu, (arm, ranking, lr) in enumerate(arms):
        out = outputs / f'recoverability_best_native_policy_{arm}_{stage}_20261005'
        assert not out.exists()
        command = [python, '-u', '-m', 'research.train_recoverability', '--train', str(cache / 'train'),
                   '--validation', str(cache / 'validation'), '--batch-size', '384', '--epochs', str(epochs),
                   '--retain-epochs', *map(str, retained), '--lr', str(lr), '--seed', '42',
                   '--search-supervision', 'selector', '--action-ranking', ranking, '--write-verification', 'action',
                   '--init-checkpoint', str(teacher), '--output', str(out)]
        commands.append((gpu, command, private))
    wave(commands, stage.upper() + '_ABC_TRAINING')
    cpu('best_native_policy_audit_fit_cpu_20261005.py', '--stage', stage, cwd=private)


fits('m0')
assert read(setup / 'best_native_policy_m0_fit_cpu_acceptance_20261005.json')['status'] == 'PASS'
fits('full')
accepted = read(setup / 'best_native_policy_full_fit_cpu_acceptance_20261005.json')
assert accepted['status'] == 'PASS' and accepted['all_four_full_fits_CPU_passed']
progress('FULL60_EPOCH_TRAINING_COMPLETE_ALL4_300_UPDATES')

evaluated = []
for epoch_file in ['epoch_020.pth', 'epoch_040.pth', 'epoch_060.pth', 'best.pth']:
    commands, pending = [], []
    for gpu, (arm, _, _) in enumerate(arms):
        root = Path(accepted['arms'][arm]['root'])
        info = accepted['arms'][arm]['checkpoints'][epoch_file]
        # Cached epoch0 is the unchanged parent; an already evaluated fixed epoch needs no repeated visual pass.
        if info['epoch'] == 0 or any(row['arm'] == arm and row['epoch'] == info['epoch'] for row in evaluated):
            continue
        model = root / epoch_file
        pred = root / ('continuous_' + epoch_file[:-4]) / 'predictions'
        assert not pred.parent.exists()
        command = [python, '-u', '-m', 'research.evaluate_recoverability', '--dataset', 'lasher',
                   '--root', '/data/wangwj/dataset/LasHeR/traingset', '--model', str(model),
                   '--validation-split', '/data/gb/outputs/c1_initial_seed42/split.json',
                   '--write-verification', 'action', '--seed', '42', '--max-frames', '0', '--output', str(pred)]
        commands.append((gpu, command, main))
        pending.append({'arm': arm, 'model': str(model), 'predictions': str(pred), 'epoch': info['epoch']})
    if commands:
        wave(commands, 'FULL98_' + epoch_file[:-4])
        for row in pending:
            cpu('best_native_policy_score_full98_cpu_20261005.py', '--predictions', row['predictions'], '--model', row['model'])
            score = read(Path(row['predictions']).parent / 'full98_selection_score.json')
            evaluated.append(row | {'sequence_mean_iou': score['sequence_mean_iou']})
assert evaluated
selected = max(evaluated, key=lambda r: r['sequence_mean_iou'])
selection = {'status': 'PASS', 'checkpoint': selected['model'], 'selected_best_epoch': selected['epoch'],
             'all_four_full_fits_CPU_passed': True, 'completed_epochs': 60, 'optimizer_steps': 300,
             'selection_rule': 'Highest full98 continuous TRAIN-held-out sequence mean IoU among trained positive epochs; declared order on ties; no native-test selection.',
             'candidates': evaluated, 'both_datasets_same_fixed_checkpoint': True,
             'parent_remains_historical_reference': str(teacher), 'selected_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
(setup / 'best_native_policy_native_selection_20261005.json').write_text(json.dumps(selection, indent=2) + '\n')
progress('FULL98_CONTINUOUS_SELECTION_COMPLETE', selection=selection)

# Keep all four GPUs useful by partitioning both complete benchmark datasets, preserving every frame.
commands = []
for gpu in range(4):
    commands.append((gpu, ['bash', str(setup / 'run_best_native_policy_native_complete_20261005.sh'), str(gpu)], main))
wave(commands, 'BOTH_COMPLETE_NATIVE_BENCHMARKS')
progress('BOTH_NATIVE_INFERENCE_COMPLETE_SCORING_ALL_METRICS')
for dataset in ('lasher', 'rgbt234'):
    cpu('best_native_policy_merge_native_complete_20261005.py', '--dataset', dataset)
    subprocess.run(['bash', str(setup / 'report_best_native_policy_complete_20261005.sh'), dataset], cwd=main, env=env, check=True)
    cpu('best_native_policy_audit_native_complete_cpu_20261005.py', '--dataset', dataset)
reports = {d: read(outputs / f'recoverability_best_native_policy_native_full_20261005/{d}/core_report/full_report.json')
           for d in ('lasher', 'rgbt234')}
metrics = []
for dataset, keys in [('lasher', ('PR', 'NPR', 'SR')), ('rgbt234', ('MPR', 'MSR'))]:
    report = reports[dataset]
    for key in keys:
        baseline = report['variants']['baseline']['overall_metrics_percent'][key]
        ours = report['variants']['best_native_policy_complete']['overall_metrics_percent'][key]
        metrics.append({'dataset': dataset, 'metric': key, 'baseline_percent': baseline,
                        'ABC_percent': ours, 'delta_percentage_points': ours - baseline,
                        'target_percent': baseline + 2, 'meets_plus_two': ours >= baseline + 2})
combined = {'completed': True, 'selected_checkpoint': selection, 'training': accepted,
            'datasets': reports, 'acceptance': metrics, 'all_five_meet_plus_two': all(r['meets_plus_two'] for r in metrics),
            'scope': 'Complete frozen-base ABC module training plus full continuous validation and both native benchmarks; not end-to-end visual-backbone training.'}
complete = outputs / 'recoverability_best_native_policy_native_full_20261005/complete_core_report.json'
complete.write_text(json.dumps(combined, indent=2) + '\n')
progress('COMPLETE_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY', selection=selection,
         complete_core_report=str(complete), full_metrics=metrics,
         full_metrics_reports={d: str(outputs / f'recoverability_best_native_policy_native_full_20261005/{d}/core_report/full_report.json') for d in ('lasher', 'rgbt234')})
