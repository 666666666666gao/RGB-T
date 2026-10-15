"""Close a completed R3 fit with full98 selection and all three benchmarks.

This separate controller never edits or starts the currently running R2 jobs.
Use only after its actual training process and R3 GPU qualification succeed.
Each inference shard runs on one of the authorized physical GPUs2/3.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', required=True)
    parser.add_argument('--split-manifest', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline-234', required=True)
    parser.add_argument('--baseline-210', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    fit = Path(args.fit).resolve()
    terminal = json.loads((fit / 'fit_complete.json').read_text())
    assert terminal['status'] in ('DENSE_MOTION_TRAINING_COMPLETE', 'DENSE_CONSEQUENCE_TRAINING_COMPLETE')
    assert terminal['epochs'] == 20
    assert [candidate['epoch'] for candidate in terminal['candidates']] == [0, 1, 2, 5, 10, 20]
    split_manifest = Path(args.split_manifest).resolve()
    split = json.loads(split_manifest.read_text())
    assert len(split['train_sequences']) == 881 and len(split['validation_sequences']) == 98
    assert not set(split['train_sequences']) & set(split['validation_sequences'])
    config = json.loads(Path(args.config).read_text())
    project = Path(__file__).resolve().parent.parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='DENSE_R3_FULL_VIDEO_EVALUATION_RUNNING', controller_pid=os.getpid(),
        fit=str(fit), started_utc=datetime.now(timezone.utc).isoformat(), phases={}, reports={},
        formal_checkpoint_selection='full98 trained positive epoch, same checkpoint on all3',
        goal_achieved=False)

    def save():
        (output / 'queue_state.json').write_text(json.dumps(state, indent=2))

    def run(name, commands, devices):
        phase = dict(status='RUNNING', started_utc=datetime.now(timezone.utc).isoformat(),
            commands=commands, physical_devices=devices)
        state['phases'][name] = phase
        children = []
        for index, (command, device) in enumerate(zip(commands, devices)):
            with (output / f'{name}_{index}.log').open('w') as stream:
                child = subprocess.Popen([sys.executable, '-m', *command], cwd=project,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='4',
                        OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'),
                    stdout=stream, stderr=subprocess.STDOUT)
            children.append(child)
        phase['pids'] = [child.pid for child in children]
        save()
        codes = [child.wait() for child in children]
        phase.update(status='COMPLETE' if all(code == 0 for code in codes) else 'FAILED',
            actual_exit_codes=codes, ended_utc=datetime.now(timezone.utc).isoformat())
        save()
        assert all(code == 0 for code in codes), (name, codes)

    save()
    for candidate in terminal['candidates']:
        checkpoint = Path(candidate['checkpoint']).resolve()
        development = output / 'development' / checkpoint.stem
        commands = [['cadtrack_abc.validate_dense_main', '--checkpoint', str(checkpoint),
            '--manifest', str(split_manifest), '--output', str(development), '--shard', str(shard)]
            for shard in (0, 1)]
        run('full98_' + checkpoint.stem, commands, [2, 3])
    run('select_trained', [['cadtrack_abc.select_dense_main', '--fit', str(fit),
        '--split-manifest', str(split_manifest), '--development', str(output / 'development')]], [''])
    selection = json.loads((fit / 'selection.json').read_text())
    assert selection['full98_development_selection_complete'] and selection['strict_dense_abc_reload']
    selected = Path(selection['trained_best_checkpoint'])
    digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    assert digest == selection['trained_selected']['checkpoint_sha256']
    state['selected_checkpoint'] = str(selected)
    state['selected_checkpoint_sha256'] = digest
    state['selected_epoch'] = selection['trained_selected']['epoch']
    save()
    roots = dict(lasher=Path(config['test_root']), rgbt234=project / 'dataset/rgbt234',
        rgbt210=project / 'dataset/rgbt210/RGBT210')
    baselines = dict(rgbt234=Path(args.baseline_234).resolve(), rgbt210=Path(args.baseline_210).resolve())
    for dataset, count, frames in [('lasher', 245, 220703), ('rgbt234', 234, 116649), ('rgbt210', 210, 104706)]:
        predictions = output / 'formal' / (dataset + '_predictions')
        commands = [['cadtrack_abc.infer_dense_main', '--checkpoint', str(selected), '--dataset', dataset,
            '--root', str(roots[dataset]), '--output', str(predictions), '--shard', str(shard)]
            for shard in (0, 1)]
        run('infer_' + dataset, commands, [2, 3])
        receipts = [json.loads((predictions / f'complete_shard{shard}.json').read_text()) for shard in (0, 1)]
        assert all(receipt['completed'] and receipt['checkpoint_sha256'] == digest for receipt in receipts)
        assert sum(receipt['sequences'] for receipt in receipts) == count
        assert sum(receipt['frames'] for receipt in receipts) == frames
        command = ['cadtrack_abc.score_dense_main', '--dataset', dataset, '--root', str(roots[dataset]),
            '--predictions', str(predictions), '--output', str(output / 'formal' / (dataset + '_report'))]
        if dataset != 'lasher':
            command += ['--baseline-report', str(baselines[dataset])]
        run('score_' + dataset, [command], [''])
        report_path = output / 'formal' / (dataset + '_report') / 'qualified_full_report.json'
        report = json.loads(report_path.read_text())
        assert report['completed'] and report['checkpoint_sha256'] == digest
        assert report['sequences'] == count and report['frames'] == frames
        state['reports'][dataset] = dict(path=str(report_path), metrics=report['overall_metrics_percent'],
            baseline=report['baseline_metrics_percent'], delta=report['delta_percentage_points'],
            plus_two_reached=report['plus_two_reached'], stage=report['stage'],
            future_actions_trained=report['future_actions_trained'], search_actions_trained=report['search_actions_trained'])
        save()
    assert hashlib.sha256(selected.read_bytes()).hexdigest() == digest
    state.update(status='DENSE_R3_FULL_VIDEO_EVALUATION_COMPLETE',
        ended_utc=datetime.now(timezone.utc).isoformat(), full98_selection_complete=True,
        seven_metric_plus_two=all(all(report['plus_two_reached'].values()) for report in state['reports'].values()),
        goal_achieved=False, module_effectiveness_not_established=True,
        cleanup='Retain selected dependencies and candidates until complete report audit; retire unused generated weights afterwards.')
    save()
    (output / 'evaluation_complete.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
