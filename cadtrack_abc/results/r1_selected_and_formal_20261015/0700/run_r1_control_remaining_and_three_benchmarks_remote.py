"""Finish the matched control's existing full98 and formal evaluation protocol."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

project = Path('/home/gaob/CADTrack-ABC')
python = '/data/wangwj/envs/CADTrack/bin/python'
fit = project / 'outputs/r1_matched_fits_20261015/fit_future_actions'
development = project / 'outputs/r1_control_full98_20261015'
manifest = project / 'outputs/cad_abc_20261013/manifest.json'
config_path = project / 'cadtrack_abc/config.json'
config = json.loads(config_path.read_text())
assert config['physical_gpus'] == [2, 3]
fitted = json.loads((fit / 'fit_complete.json').read_text())
assert fitted['status'] == 'FUTURE_HEAD_FIT_COMPLETE' and fitted['epochs'] == 20
assert fitted['updates'] == 1600 and fitted['strict_head_reload']
assert [row['epoch'] for row in fitted['candidates']] == [0, 1, 2, 5, 10, 20]
names = sorted(json.loads(manifest.read_text())['validation_sequences'])
assert len(names) == 98
output = project / 'outputs/r1_control_reports_20261015'
output.mkdir(exist_ok=False)
state_path = output / 'benchmark_state.json'
state = dict(status='WAITING_FOR_ORIGINAL_EPOCH00', controller_pid=os.getpid(),
    started_utc=datetime.now(timezone.utc).isoformat(), physical_gpu=3, max_workers_per_gpu=2,
    first_parent_observation_bjt='2026-10-15 07:12:00', subsequent_wait_seconds=240,
    policy='future_actions', phases={}, reports={}, new_training_started=False,
    optimizer_updates=0, baseline_lasher_rerun=False, weights_cleaned=False,
    goal_completion_not_claimed=True)

def save():
    state_path.write_text(json.dumps(state, indent=2))

def run(phase, commands, gpu=None):
    record = dict(status='RUNNING', commands=commands,
        started_utc=datetime.now(timezone.utc).isoformat(), jobs=[])
    state['phases'][phase] = record
    streams, jobs = [], []
    for index, command in enumerate(commands):
        stream = (output / f'{phase}_shard{index}.log').open('w')
        streams.append(stream)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
        child = subprocess.Popen([python, '-m', *command], cwd=project, env=environment,
            stdout=stream, stderr=subprocess.STDOUT)
        jobs.append(child)
        record['jobs'].append(dict(pid=child.pid, physical_gpu=gpu, command=command))
    save()
    print(json.dumps(dict(event='PHASE_STARTED', phase=phase, jobs=record['jobs'])), flush=True)
    codes = [child.wait() for child in jobs]
    for stream in streams:
        stream.close()
    record.update(return_codes=codes, completed_utc=datetime.now(timezone.utc).isoformat(),
        status='COMMANDS_EXITED_ZERO' if all(code == 0 for code in codes) else 'FAILED')
    save()
    assert all(code == 0 for code in codes), (phase, codes)

def verify_development(epoch):
    root = development / f'candidate_epoch{epoch:02d}'
    records = []
    for shard in (0, 1):
        receipt = json.loads((root / f'development_shard{shard}.json').read_text())
        assert receipt['completed'] and receipt['full_videos'] and not receipt['sanity']
        assert receipt['epoch'] == epoch and receipt['policy_family'] == 'future_actions'
        assert receipt['checkpoint'] == str(fit / f'candidate_epoch{epoch:02d}.pth')
        assert receipt['shard'] == shard and receipt['shards'] == 2
        assert [row['sequence'] for row in receipt['records']] == names[shard::2]
        records.extend(receipt['records'])
    assert len(records) == 98 and sum(row['frames'] for row in records) == 52200
    return dict(sequences=98, frames=52200,
        sequence_equal_mean_iou=sum(row['mean_iou'] for row in records) / 98,
        extra_forwards=sum(row['extra_forwards'] for row in records),
        future_interventions=sum(row['future_interventions'] for row in records))

save()
deadline = datetime(2026, 10, 14, 23, 12, tzinfo=timezone.utc)
time.sleep(max(0, deadline.timestamp() - time.time()))
initial = development / 'candidate_epoch00'
while True:
    observed = []
    for shard in (0, 1):
        admission = json.loads((initial / f'launch_shard{shard}.json').read_text())
        pid = admission['pid']
        process = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True)
        live = process.returncode == 0
        if live:
            assert 'cadtrack_abc.validate_future_actions' in process.stdout
            assert str(fit / 'candidate_epoch00.pth') in process.stdout
            assert f'--shard {shard}' in process.stdout
        exit_path = initial / f'exit_shard{shard}.json'
        observed.append(dict(pid=pid, shard=shard, actual_pid_live=live,
            exit_receipt_present=exit_path.exists()))
    state['parent_epoch00_observation'] = dict(observed_utc=datetime.now(timezone.utc).isoformat(), jobs=observed)
    save()
    if all(row['exit_receipt_present'] for row in observed):
        break
    assert any(row['actual_pid_live'] for row in observed), observed
    time.sleep(240)
assert all(json.loads((initial / f'exit_shard{shard}.json').read_text())['child_exit_code'] == 0 for shard in (0, 1))
state['phases']['develop_epoch00'] = dict(status='REUSED_VERIFIED_COMPLETE',
    original_native_sessions=[46511, 25976], original_child_exit_codes=[0, 0],
    evidence=verify_development(0))
memory = subprocess.run(['nvidia-smi', '-i', '3', '--query-gpu=memory.used',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
assert int(memory.strip()) < 500, memory
state['status'] = 'RUNNING'
save()
for candidate in fitted['candidates'][1:]:
    epoch = candidate['epoch']
    checkpoint = Path(candidate['checkpoint'])
    destination = development / checkpoint.stem
    assert not destination.exists()
    phase = f'develop_epoch{epoch:02d}'
    run(phase, [['cadtrack_abc.validate_future_actions', '--config', str(config_path),
        '--checkpoint', str(checkpoint), '--manifest', str(manifest), '--output', str(destination),
        '--policy', 'future_actions', '--shard', str(shard), '--shards', '2'] for shard in (0, 1)], gpu=3)
    state['phases'][phase].update(status='VERIFIED_COMPLETE', evidence=verify_development(epoch))
    save()
run('select', [['cadtrack_abc.select_future_actions', '--fit', str(fit), '--policy', 'future_actions',
    '--manifest', str(manifest), '--development', str(development)]])
selection = json.loads((fit / 'selection.json').read_text())
assert selection['full98_development_selection_complete'] and selection['strict_future_head_reload']
checkpoints = {'selected': (selection['best_checkpoint'], selection['selected']['epoch'])}
if selection['trained_best_checkpoint'] != selection['best_checkpoint']:
    checkpoints['trained'] = (selection['trained_best_checkpoint'], selection['trained_selected']['epoch'])
for role, (checkpoint, epoch) in checkpoints.items():
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    state['reports'][role] = {}
    for dataset in ('lasher', 'rgbt234', 'rgbt210'):
        assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == digest
        if dataset == 'lasher':
            root, expected = Path(config['test_root']), dict(sequences=245, frames=220703)
        elif dataset == 'rgbt234':
            root, expected = project / 'dataset/rgbt234', dict(sequences=234, frames=116649)
        else:
            inventory = json.loads((project / 'dataset/rgbt210_complete.json').read_text())
            assert inventory['completed'] and inventory['sequences'] == 210
            root, expected = Path(inventory['root']), dict(sequences=210, frames=inventory['frames'])
        predictions = output / role / f'{dataset}_predictions'
        phase = f'infer_{role}_{dataset}'
        run(phase, [['cadtrack_abc.infer', '--config', str(config_path), '--checkpoint', checkpoint,
            '--policy', 'future_actions', '--dataset', dataset, '--root', str(root),
            '--output', str(predictions), '--shard', str(shard), '--shards', '2'] for shard in (0, 1)], gpu=3)
        receipts = [json.loads((predictions / f'complete_shard{shard}.json').read_text()) for shard in (0, 1)]
        assert all(row['completed'] and row['dataset'] == dataset and row['checkpoint'] == checkpoint
            and row['epoch'] == epoch and row['policy_family'] == 'future_actions' for row in receipts)
        assert sum(row['sequences'] for row in receipts) == expected['sequences']
        assert sum(row['frames'] for row in receipts) == expected['frames']
        state['phases'][phase].update(status='VERIFIED_COMPLETE', inventory=expected)
        report_directory = output / role / f'{dataset}_report'
        run(f'score_{role}_{dataset}', [['cadtrack_abc.score', '--dataset', dataset, '--root', str(root),
            '--predictions', str(predictions), '--output', str(report_directory), '--lasher-target-gain', '6']])
        report = json.loads((report_directory / 'full_report.json').read_text())
        assert report['completed'] and report['dataset'] == dataset and report['checkpoint'] == checkpoint
        assert report['epoch'] == epoch and report['sequences'] == expected['sequences']
        assert report['frames'] == expected['frames']
        assert set(report['overall_metrics_percent']) == {
            'lasher': {'PR', 'NPR', 'SR'}, 'rgbt234': {'MPR', 'MSR'}, 'rgbt210': {'PR', 'SR'}}[dataset]
        state['reports'][role][dataset] = dict(report=str(report_directory / 'full_report.json'),
            checkpoint=checkpoint, epoch=epoch, metrics=report['overall_metrics_percent'])
        save()
assert all(set(reports) == {'lasher', 'rgbt234', 'rgbt210'} for reports in state['reports'].values())
selection['formal_metrics_complete'] = True
selection['formal_reports'] = state['reports']
(fit / 'selection.json').write_text(json.dumps(selection, indent=2))
state.update(status='CONTROL_R1_THREE_BENCHMARKS_COMPLETE',
    completed_utc=datetime.now(timezone.utc).isoformat(), trained_checkpoint_formal_metrics_complete=True)
save()
print(json.dumps(state), flush=True)
