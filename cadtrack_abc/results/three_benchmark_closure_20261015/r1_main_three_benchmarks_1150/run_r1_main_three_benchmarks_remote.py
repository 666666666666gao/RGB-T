"""Run the fixed selected R1 policy through existing inference and scoring CLIs."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess

project = Path('/home/gaob/CADTrack-ABC')
python = '/data/wangwj/envs/CADTrack/bin/python'
fit = project / 'outputs/r1_matched_fits_20261015/fit_search_actions'
selection = json.loads((fit / 'selection.json').read_text())
assert selection['full98_development_selection_complete'] and selection['strict_future_head_reload']
assert len(selection['candidates']) == 6
assert selection['selected']['epoch'] == 10
assert selection['best_checkpoint'] == selection['trained_best_checkpoint']
checkpoint = Path(selection['best_checkpoint'])
checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
config_path = project / 'cadtrack_abc/config.json'
config = json.loads(config_path.read_text())
assert config['physical_gpus'] == [2, 3]
assert config['pretrained'] == '/home/gaob/CADTrack_ep0014.pth.tar'
output = project / 'outputs/r1_main_reports_20261015'
output.mkdir(exist_ok=False)
state_path = output / 'benchmark_state.json'
source_names = ['baseline.py', 'data.py', 'model.py', 'collect_actions.py', 'future_actions.py',
    'pure_baseline_tracker.py', 'infer_baseline.py', 'baseline_pipeline.py', 'future_pipeline.py',
    'train_future_actions.py', 'r1_pair_pipeline.py', 'select_future_actions.py', 'infer.py', 'score.py']
source_digests = {name: hashlib.sha256((project / 'cadtrack_abc' / name).read_bytes()).hexdigest()
    for name in source_names}
state = dict(status='RUNNING', controller_pid=os.getpid(),
    started_utc=datetime.now(timezone.utc).isoformat(), physical_gpu=2,
    max_workers_per_gpu=2, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_digest,
    policy='search_actions', selected_epoch=10, configuration=str(config_path),
    configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
    source_sha256=source_digests, phases={}, reports={}, baseline_lasher_rerun=False,
    source_changed=False, optimizer_updates=0, goal_completion_not_claimed=True)

def save():
    state_path.write_text(json.dumps(state, indent=2))

def run(phase, commands, gpu=None):
    record = dict(status='RUNNING', commands=commands,
        started_utc=datetime.now(timezone.utc).isoformat(), jobs=[])
    state['phases'][phase] = record
    streams, jobs = [], []
    for index, command in enumerate(commands):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
        stream = (output / f'{phase}_shard{index}.log').open('w')
        streams.append(stream)
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

gpu_apps = subprocess.run(['nvidia-smi', '-i', '2', '--query-gpu=memory.used',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
assert int(gpu_apps.strip()) < 500, gpu_apps
save()
for dataset in ('lasher', 'rgbt234', 'rgbt210'):
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_digest
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == state['configuration_sha256']
    assert {name: hashlib.sha256((project / 'cadtrack_abc' / name).read_bytes()).hexdigest()
        for name in source_names} == source_digests
    if dataset == 'lasher':
        root = Path(config['test_root'])
        expected = dict(sequences=245, frames=220703)
    elif dataset == 'rgbt234':
        root = project / 'dataset/rgbt234'
        expected = dict(sequences=234, frames=116649)
    else:
        inventory = json.loads((project / 'dataset/rgbt210_complete.json').read_text())
        assert inventory['completed'] and inventory['sequences'] == 210
        root = Path(inventory['root'])
        expected = dict(sequences=210, frames=inventory['frames'])
    role = output / 'selected'
    predictions = role / f'{dataset}_predictions'
    phase = 'infer_' + dataset
    commands = [['cadtrack_abc.infer', '--config', str(config_path), '--checkpoint', str(checkpoint),
        '--policy', 'search_actions', '--dataset', dataset, '--root', str(root),
        '--output', str(predictions), '--shard', str(shard), '--shards', '2'] for shard in (0, 1)]
    run(phase, commands, gpu=2)
    receipts = [json.loads((predictions / f'complete_shard{shard}.json').read_text()) for shard in (0, 1)]
    assert all(row['completed'] and row['dataset'] == dataset and row['checkpoint'] == str(checkpoint)
        and row['epoch'] == 10 and row['policy_family'] == 'search_actions' for row in receipts)
    assert sum(row['sequences'] for row in receipts) == expected['sequences']
    assert sum(row['frames'] for row in receipts) == expected['frames']
    state['phases'][phase]['status'] = 'VERIFIED_COMPLETE'
    state['phases'][phase]['inventory'] = expected
    save()
    report_directory = role / f'{dataset}_report'
    run('score_' + dataset, [['cadtrack_abc.score', '--dataset', dataset, '--root', str(root),
        '--predictions', str(predictions), '--output', str(report_directory), '--lasher-target-gain', '6']])
    report = json.loads((report_directory / 'full_report.json').read_text())
    assert report['completed'] and report['checkpoint'] == str(checkpoint) and report['epoch'] == 10
    assert report['dataset'] == dataset and report['sequences'] == expected['sequences']
    assert report['frames'] == expected['frames']
    assert set(report['overall_metrics_percent']) == {
        'lasher': {'PR', 'NPR', 'SR'}, 'rgbt234': {'MPR', 'MSR'}, 'rgbt210': {'PR', 'SR'}}[dataset]
    state['reports'][dataset] = dict(report=str(report_directory / 'full_report.json'),
        checkpoint=str(checkpoint), epoch=10, metrics=report['overall_metrics_percent'])
    state['phases']['score_' + dataset]['status'] = 'VERIFIED_COMPLETE'
    save()
    print(json.dumps(dict(event='FORMAL_DATASET_COMPLETE', dataset=dataset,
        metrics=report['overall_metrics_percent'])), flush=True)
assert set(state['reports']) == {'lasher', 'rgbt234', 'rgbt210'}
selection['formal_metrics_complete'] = True
selection['formal_reports'] = {'selected': state['reports']}
(fit / 'selection.json').write_text(json.dumps(selection, indent=2))
state.update(status='MAIN_R1_THREE_BENCHMARKS_COMPLETE',
    completed_utc=datetime.now(timezone.utc).isoformat(),
    trained_checkpoint_formal_metrics_complete=True, weights_cleaned=False)
save()
print(json.dumps(state), flush=True)
