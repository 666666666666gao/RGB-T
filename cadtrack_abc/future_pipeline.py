"""One deferred future-C round after the native20ep/fullLas result is closed.

Uses the project's direct subprocess pipeline style. No retry, GPU reassignment,
new baseline evaluation or additional dataset. Each dependency needs verified
success, rather than merely a process that has exited.
"""
import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def validate_parent(parent):
    assert (parent / 'pipeline_complete.json').exists(), 'Current native training/fullLas pipeline has not completed; no future jobs may start.'
    trained = json.loads((parent / 'complete.json').read_text())
    completed = json.loads((parent / 'pipeline_complete.json').read_text())
    report = json.loads((parent / 'lasher_report/full_report.json').read_text())
    assert trained['status'] == 'TRAINING_COMPLETE' and trained['epochs'] == 20 and trained['updates'] == 3140
    assert trained['strict_reload_passed'] and completed['training_complete']
    assert report['completed'] and report['dataset'] == 'lasher'
    assert report['sequences'] == 245 and report['frames'] == 220703
    assert set(report['overall_metrics_percent']) == {'PR', 'NPR', 'SR'}
    checkpoint = Path(completed['best_checkpoint']).resolve()
    assert checkpoint == Path(trained['best_checkpoint']).resolve() == Path(report['checkpoint']).resolve()
    assert checkpoint.is_file()
    return checkpoint, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, help='New future-round directory; current native output is never reused.')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    assert config['physical_gpus'] == [2, 3] and config['evaluation_focus'] == 'lasher'
    parent = Path(config['output']).resolve()
    parent_checkpoint, parent_report = validate_parent(parent)
    output = Path(args.output).resolve()
    assert output != parent
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parent.parent
    manifest = parent / 'manifest.json'
    inventory = json.loads(manifest.read_text())
    assert len(inventory['train_sequences']) == 881 and len(inventory['validation_sequences']) == 98
    teacher = output / 'teacher.pth'
    shutil.copyfile(parent_checkpoint, teacher)
    assert filecmp.cmp(parent_checkpoint, teacher, shallow=False)
    teacher_sha = hashlib.sha256(teacher.read_bytes()).hexdigest()
    state = {'status': 'RUNNING', 'parent_checkpoint': str(parent_checkpoint),
             'parent_metrics_percent': parent_report['overall_metrics_percent'],
             'teacher': str(teacher), 'teacher_sha256': teacher_sha,
             'physical_gpus': [2, 3], 'baseline_rerun': False, 'evaluation_focus': 'lasher',
             'started_unix': time.time(), 'phases': {}}

    def store():
        (output / 'queue_state.json').write_text(json.dumps(state, indent=2))

    def run(name, commands, devices):
        phase = {'status': 'RUNNING', 'commands': commands, 'devices': devices, 'started_unix': time.time()}
        state['phases'][name] = phase
        store()
        streams, jobs = [], []
        for index, (command, device) in enumerate(zip(commands, devices)):
            stream = (output / (name + '_' + str(index) + '.log')).open('w')
            streams.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            jobs.append(subprocess.Popen([sys.executable, '-m', *command], cwd=project, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT))
        codes = [job.wait() for job in jobs]
        for stream in streams:
            stream.close()
        phase.update(return_codes=codes, exited_unix=time.time(),
                     status='COMMANDS_EXITED_SUCCESSFULLY' if all(code == 0 for code in codes) else 'FAILED')
        state['status'] = 'RUNNING' if all(code == 0 for code in codes) else 'FAILED'
        store()
        assert all(code == 0 for code in codes), (name, codes)

    def verified(name, evidence):
        state['phases'][name].update(status='VERIFIED_COMPLETE', evidence=evidence)
        store()

    indices = {}
    for split in ['train', 'validation']:
        sampling = output / ('queries_' + split)
        sample_commands = [['cadtrack_abc.sample_action_queries', '--config', str(config_path), '--teacher', str(teacher),
                            '--manifest', str(manifest), '--split', split, '--output', str(sampling), '--device', 'cuda',
                            '--shard', str(shard)] for shard in range(2)]
        phase = 'sample_' + split
        run(phase, sample_commands, [2, 3])
        sampled = [json.loads((sampling / ('sampling_shard' + str(shard) + '.json')).read_text()) for shard in range(2)]
        expected = set(inventory['train_sequences' if split == 'train' else 'validation_sequences'])
        assert all(receipt['program_complete'] and receipt['full_shard_queries_complete'] and receipt['full_sequence_rollouts']
                   and not receipt['sanity'] and receipt['teacher_sha256'] == teacher_sha for receipt in sampled)
        assert sum(receipt['sequences'] for receipt in sampled) == len(expected)
        assert {record['sequence'] for receipt in sampled for record in receipt['records']} == expected
        verified(phase, {'sequences': len(expected), 'queries': sum(receipt['queries'] for receipt in sampled)})
        folders = [output / ('collection_' + split) / ('shard' + str(shard)) for shard in range(2)]
        collect_commands = [['cadtrack_abc.collect_actions', '--config', str(config_path), '--teacher', str(teacher),
                             '--manifest', str(manifest), '--split', split, '--queries', str(sampling / ('queries_shard' + str(shard) + '.json')),
                             '--output', str(folders[shard]), '--device', 'cuda', '--horizon', '3'] for shard in range(2)]
        for command in collect_commands:
            command.extend(['--action-batch', '64'])
        phase = 'collect_' + split
        run(phase, collect_commands, [2, 3])
        receipts = [json.loads((folder / 'collection_complete.json').read_text()) for folder in folders]
        assert all(receipt['completed'] and receipt['teacher_sha256'] == teacher_sha
                   and receipt['model_family'] == 'CADTrackABC-native-current-quality' for receipt in receipts)
        assert all(receipts[shard]['queries'] == sampled[shard]['queries'] for shard in range(2))
        verified(phase, {'queries': sum(receipt['queries'] for receipt in receipts), 'horizon': 3})
        index = output / (split + '_index.json')
        phase = 'merge_' + split
        run(phase, [['cadtrack_abc.merge_action_packets', '--collections', *map(str, folders), '--manifest', str(manifest),
                     '--split', split, '--output', str(index)]], [2])
        merged = json.loads(index.read_text())
        assert merged['completed'] and merged['teacher_sha256'] == teacher_sha and merged['sequences'] == len(expected)
        indices[split] = index
        verified(phase, {'index': str(index), 'queries': merged['queries'], 'sequences': merged['sequences']})
    fit = output / 'fit'
    run('fit', [['cadtrack_abc.train_future_actions', '--config', str(config_path), '--teacher', str(teacher),
                 '--manifest', str(manifest), '--train-index', str(indices['train']), '--validation-index', str(indices['validation']),
                 '--output', str(fit), '--epochs', '20', '--batch', '64', '--workers', '4', '--device', 'cuda']], [2])
    fitted = json.loads((fit / 'fit_complete.json').read_text())
    query_count = json.loads(indices['train'].read_text())['queries']
    assert fitted['status'] == 'FUTURE_HEAD_FIT_COMPLETE' and fitted['epochs'] == 20
    assert fitted['updates'] == ((query_count + 63) // 64) * 20 and fitted['train_sequences'] == 881
    assert fitted['strict_head_reload'] and fitted['teacher_sha256'] == teacher_sha
    verified('fit', {'epochs': fitted['epochs'], 'updates': fitted['updates'], 'candidates': fitted['candidates']})
    development = output / 'development'
    for candidate in fitted['candidates']:
        checkpoint = Path(candidate['checkpoint'])
        destination = development / checkpoint.stem
        phase = 'develop_' + checkpoint.stem
        commands = [['cadtrack_abc.validate_future_actions', '--config', str(config_path), '--checkpoint', str(checkpoint),
                     '--manifest', str(manifest), '--output', str(destination), '--shard', str(shard)] for shard in range(2)]
        run(phase, commands, [2, 3])
        receipts = [json.loads((destination / ('development_shard' + str(shard) + '.json')).read_text()) for shard in range(2)]
        assert all(receipt['completed'] and receipt['full_videos'] and not receipt['sanity'] for receipt in receipts)
        assert sum(receipt['sequences'] for receipt in receipts) == 98
        verified(phase, {'sequences': 98, 'checkpoint': str(checkpoint)})
    run('select', [['cadtrack_abc.select_future_actions', '--fit', str(fit), '--manifest', str(manifest),
                    '--development', str(development)]], [2])
    selected = json.loads((fit / 'selection.json').read_text())
    assert selected['full98_development_selection_complete'] and selected['strict_future_head_reload']
    best = selected['best_checkpoint']
    verified('select', {'checkpoint': best, 'selected': selected['selected']})
    predictions = output / 'lasher_predictions'
    commands = [['cadtrack_abc.infer', '--config', str(config_path), '--checkpoint', best, '--policy', 'future_actions',
                 '--dataset', 'lasher', '--root', config['test_root'], '--output', str(predictions), '--shard', str(shard)] for shard in range(2)]
    run('formal_inference', commands, [2, 3])
    receipts = [json.loads((predictions / ('complete_shard' + str(shard) + '.json')).read_text()) for shard in range(2)]
    assert all(receipt['completed'] and receipt['policy_family'] == 'future_actions' and receipt['checkpoint'] == best for receipt in receipts)
    assert sum(receipt['sequences'] for receipt in receipts) == 245 and sum(receipt['frames'] for receipt in receipts) == 220703
    verified('formal_inference', {'checkpoint': best, 'sequences': 245, 'frames': 220703})
    report_directory = output / 'lasher_report'
    run('score', [['cadtrack_abc.score', '--dataset', 'lasher', '--root', config['test_root'],
                   '--predictions', str(predictions), '--output', str(report_directory)]], [2])
    report = json.loads((report_directory / 'full_report.json').read_text())
    assert report['completed'] and report['sequences'] == 245 and report['frames'] == 220703
    assert set(report['overall_metrics_percent']) == {'PR', 'NPR', 'SR'} and report['checkpoint'] == best
    verified('score', {'report': str(report_directory / 'full_report.json'), 'metrics': report['overall_metrics_percent']})
    assert filecmp.cmp(parent_checkpoint, teacher, shallow=False)
    teacher.unlink()
    state.update(status='FUTURE_ROUND_COMPLETE', best_checkpoint=best, formal_report=str(report_directory / 'full_report.json'),
                 lasher_stage_goal=report['lasher_stage_goal'], overall_goal_status_not_changed=True,
                 completed_unix=time.time(), temporary_teacher_cleaned=True)
    store()
    (output / 'pipeline_complete.json').write_text(json.dumps(state, indent=2))
    print(json.dumps({'status': state['status'], 'metrics': report['overall_metrics_percent'],
                      'lasher_stage_goal': report['lasher_stage_goal']}), flush=True)


if __name__ == '__main__':
    main()
