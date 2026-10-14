"""Train, select on full98, then fully test selected and trained models on three benchmarks."""
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

import numpy as np


def validate_benchmark_inventory(dataset, root):
    sequences = sorted(path for path in root.iterdir() if path.is_dir())
    assert len(sequences) == {'lasher': 245, 'rgbt210': 210, 'rgbt234': 234}[dataset]
    frames = 0
    for sequence in sequences:
        annotation = sequence / ('visible.txt' if dataset == 'rgbt234' else 'init.txt')
        labels = np.loadtxt(annotation, delimiter=',', ndmin=2)
        rgb = list((sequence / 'visible').iterdir())
        tir = list((sequence / 'infrared').iterdir())
        assert labels.shape == (len(rgb), 4) and len(rgb) == len(tir), sequence.name
        if dataset == 'rgbt234':
            infrared = np.loadtxt(sequence / 'infrared.txt', delimiter=',', ndmin=2)
            assert infrared.shape == labels.shape, sequence.name
        frames += len(rgb)
    if dataset == 'lasher':
        assert frames == 220703
    if dataset == 'rgbt234':
        assert frames == 116649
    return {'sequences': len(sequences), 'frames': frames}


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


def validate_sampling(sampling, expected, teacher_sha):
    receipts = [json.loads((sampling / ('sampling_shard' + str(shard) + '.json')).read_text())
                for shard in range(2)]
    assert all(receipt['program_complete'] and receipt['full_shard_queries_complete']
               and receipt['full_sequence_rollouts'] and not receipt['sanity']
               and receipt['teacher_sha256'] == teacher_sha
               and receipt['model_family'] == 'CADTrackABC-native-current-quality'
               and receipt['horizon'] == 3
               for receipt in receipts)
    assert sum(receipt['sequences'] for receipt in receipts) == len(expected)
    assert {record['sequence'] for receipt in receipts for record in receipt['records']} == expected
    for shard, receipt in enumerate(receipts):
        queries = json.loads((sampling / ('queries_shard' + str(shard) + '.json')).read_text())
        assert receipt['shard'] == shard and receipt['shards'] == 2
        assert receipt['queries'] == len(queries)
        assert len({(query['sequence'], query['frame']) for query in queries}) == len(queries)
        assert {query['sequence'] for query in queries} == {record['sequence'] for record in receipt['records']}
    return receipts


def validate_collection(folders, sampling, teacher_sha):
    receipts = [json.loads((folder / 'collection_complete.json').read_text()) for folder in folders]
    for shard, (folder, receipt) in enumerate(zip(folders, receipts)):
        queries = json.loads((sampling / ('queries_shard' + str(shard) + '.json')).read_text())
        expected = {(query['sequence'], query['frame']) for query in queries}
        assert receipt['completed'] and receipt['teacher_sha256'] == teacher_sha
        assert receipt['model_family'] == 'CADTrackABC-native-current-quality' and receipt['action_batch'] == 64
        records = receipt['records']
        assert receipt['queries'] == len(records) == len(expected)
        assert {(record['sequence'], record['frame']) for record in records} == expected
        assert all(record['horizon'] == 3 and record['action_labels'] == 210
                   and record['native_state_reconstruction_equal'] and record['full_predicted_prefix_from_frame0']
                   and (folder / record['file']).is_file() for record in records)
    return receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, help='New future-round directory; current native output is never reused.')
    parser.add_argument('--sampling-root', help='Completed native TRAIN/DEV query root; reuse all four verified shards instead of repeating trajectories.')
    parser.add_argument('--collection-root', help='Completed native TRAIN/DEV action root; requires matching --sampling-root and reuses all four label shards.')
    parser.add_argument('--policy', choices=['future_actions', 'search_actions'], default='future_actions')
    parser.add_argument('--fit-root', help='Reuse a completed matching20-epoch fit; still run full98 selection and all formal reports.')
    parser.add_argument('--rgbt210-root', required=True)
    parser.add_argument('--rgbt234-root', required=True)
    parser.add_argument('--lasher-target-gain', type=float, default=6.)
    args = parser.parse_args()
    assert not args.collection_root or args.sampling_root
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    assert config['physical_gpus'] == [2, 3]
    parent = Path(config['output']).resolve()
    parent_checkpoint, parent_report = validate_parent(parent)
    benchmark_roots = {'lasher': Path(config['test_root']).resolve(),
                       'rgbt210': Path(args.rgbt210_root).resolve(), 'rgbt234': Path(args.rgbt234_root).resolve()}
    benchmark_inventory = {name: validate_benchmark_inventory(name, root) for name, root in benchmark_roots.items()}
    inventory = json.loads((parent / 'manifest.json').read_text())
    parent_teacher_sha = hashlib.sha256(parent_checkpoint.read_bytes()).hexdigest()
    sampled_existing = {}
    collected_existing = {}
    if args.sampling_root:
        sampling_root = Path(args.sampling_root).resolve()
        # Require both splits before any new process: each existing card runs
        # TRAIN then DEV, so TRAIN-only completion does not free the cards.
        for split in ['train', 'validation']:
            expected = set(inventory['train_sequences' if split == 'train' else 'validation_sequences'])
            sampled_existing[split] = validate_sampling(sampling_root / ('queries_' + split), expected, parent_teacher_sha)
            if args.collection_root:
                folders = [Path(args.collection_root).resolve() / ('collection_' + split) / ('shard' + str(shard)) for shard in range(2)]
                collected_existing[split] = validate_collection(folders, sampling_root / ('queries_' + split), parent_teacher_sha)
    output = Path(args.output).resolve()
    assert output != parent
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parent.parent
    manifest = parent / 'manifest.json'
    assert len(inventory['train_sequences']) == 881 and len(inventory['validation_sequences']) == 98
    teacher = output / 'teacher.pth'
    shutil.copyfile(parent_checkpoint, teacher)
    assert filecmp.cmp(parent_checkpoint, teacher, shallow=False)
    teacher_sha = hashlib.sha256(teacher.read_bytes()).hexdigest()
    state = {'status': 'RUNNING', 'parent_checkpoint': str(parent_checkpoint),
             'parent_metrics_percent': parent_report['overall_metrics_percent'],
             'teacher': str(teacher), 'teacher_sha256': teacher_sha,
             'physical_gpus': [2, 3], 'baseline_rerun': False, 'evaluation_focus': list(benchmark_roots),
             'benchmark_inventory': benchmark_inventory, 'lasher_target_gain': args.lasher_target_gain,
             'policy_family': args.policy,
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
        sampling = (sampling_root if args.sampling_root else output) / ('queries_' + split)
        sample_commands = [['cadtrack_abc.sample_action_queries', '--config', str(config_path), '--teacher', str(teacher),
                            '--manifest', str(manifest), '--split', split, '--output', str(sampling), '--device', 'cuda',
                            '--shard', str(shard)] for shard in range(2)]
        phase = 'sample_' + split
        expected = set(inventory['train_sequences' if split == 'train' else 'validation_sequences'])
        if args.sampling_root:
            sampled = sampled_existing[split]
            state['phases'][phase] = {'status': 'REUSED_COMPLETED_NATIVE_SAMPLING', 'sampling': str(sampling), 'commands': []}
        else:
            run(phase, sample_commands, [2, 3])
            sampled = validate_sampling(sampling, expected, teacher_sha)
        verified(phase, {'sequences': len(expected), 'queries': sum(receipt['queries'] for receipt in sampled)})
        collection_root = Path(args.collection_root).resolve() if args.collection_root else output
        folders = [collection_root / ('collection_' + split) / ('shard' + str(shard)) for shard in range(2)]
        collect_commands = [['cadtrack_abc.collect_actions', '--config', str(config_path), '--teacher', str(teacher),
                             '--manifest', str(manifest), '--split', split, '--queries', str(sampling / ('queries_shard' + str(shard) + '.json')),
                             '--output', str(folders[shard]), '--device', 'cuda', '--horizon', '3'] for shard in range(2)]
        for command in collect_commands:
            command.extend(['--action-batch', '64'])
        phase = 'collect_' + split
        if args.collection_root:
            receipts = collected_existing[split]
            state['phases'][phase] = {'status': 'REUSED_COMPLETED_NATIVE_COLLECTION', 'collection': str(collection_root), 'commands': []}
        else:
            run(phase, collect_commands, [2, 3])
            receipts = validate_collection(folders, sampling, teacher_sha)
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
    fit = Path(args.fit_root).resolve() if args.fit_root else output / 'fit'
    if args.fit_root:
        state['phases']['fit'] = {'status': 'REUSED_COMPLETED_MATCHED_FIT', 'fit_root': str(fit), 'commands': []}
        store()
    else:
        run('fit', [['cadtrack_abc.train_future_actions', '--config', str(config_path), '--teacher', str(teacher),
                     '--manifest', str(manifest), '--train-index', str(indices['train']), '--validation-index', str(indices['validation']),
                     '--output', str(fit), '--epochs', '20', '--batch', '64', '--workers', '4', '--device', 'cuda', '--policy', args.policy]], [2])
    fitted = json.loads((fit / 'fit_complete.json').read_text())
    query_count = json.loads(indices['train'].read_text())['queries']
    assert fitted['status'] == ('SEARCH_ACTION_FIT_COMPLETE' if args.policy == 'search_actions' else 'FUTURE_HEAD_FIT_COMPLETE') and fitted['epochs'] == 20
    assert fitted['updates'] == ((query_count + 63) // 64) * 20 and fitted['train_sequences'] == 881
    assert fitted['strict_head_reload'] and fitted['teacher_sha256'] == teacher_sha
    if args.fit_root:
        assert fitted['policy_family'] == args.policy and fitted['train_queries'] == query_count
        assert fitted['training_hyperparameters'] == {'batch': 64, 'lr': 1e-4,
            'seed': config['seed'], 'teacher_policy': 'native'}
        for split in ['train', 'validation']:
            source_index = Path(fitted[split + '_index'])
            reused = json.loads(source_index.read_text())
            merged = json.loads(indices[split].read_text())
            def packet_records(index, source):
                return {(row['sequence'], row['frame'], str((index.parent / row['file']).resolve()))
                        for row in source['records']}
            assert packet_records(source_index, reused) == packet_records(indices[split], merged)
    verified('fit', {'epochs': fitted['epochs'], 'updates': fitted['updates'], 'candidates': fitted['candidates']})
    development = output / 'development'
    for candidate in fitted['candidates']:
        checkpoint = Path(candidate['checkpoint'])
        destination = development / checkpoint.stem
        phase = 'develop_' + checkpoint.stem
        commands = [['cadtrack_abc.validate_future_actions', '--config', str(config_path), '--checkpoint', str(checkpoint),
                     '--manifest', str(manifest), '--output', str(destination), '--shard', str(shard), '--policy', args.policy] for shard in range(2)]
        run(phase, commands, [2, 3])
        receipts = [json.loads((destination / ('development_shard' + str(shard) + '.json')).read_text()) for shard in range(2)]
        assert all(receipt['completed'] and receipt['full_videos'] and not receipt['sanity'] for receipt in receipts)
        assert sum(receipt['sequences'] for receipt in receipts) == 98
        verified(phase, {'sequences': 98, 'checkpoint': str(checkpoint)})
    run('select', [['cadtrack_abc.select_future_actions', '--fit', str(fit), '--manifest', str(manifest),
                    '--development', str(development), '--policy', args.policy]], [2])
    selected = json.loads((fit / 'selection.json').read_text())
    assert selected['full98_development_selection_complete'] and selected['strict_future_head_reload']
    best = selected['best_checkpoint']
    verified('select', {'checkpoint': best, 'selected': selected['selected']})
    checkpoints = {'selected': best}
    if selected['trained_best_checkpoint'] != best:
        checkpoints['trained'] = selected['trained_best_checkpoint']
    formal_reports = {}
    for role, checkpoint in checkpoints.items():
        formal_reports[role] = {}
        for dataset, root in benchmark_roots.items():
            predictions = output / role / (dataset + '_predictions')
            phase = 'formal_' + role + '_' + dataset
            commands = [['cadtrack_abc.infer', '--config', str(config_path), '--checkpoint', checkpoint, '--policy', args.policy,
                         '--dataset', dataset, '--root', str(root), '--output', str(predictions), '--shard', str(shard)] for shard in range(2)]
            run(phase, commands, [2, 3])
            receipts = [json.loads((predictions / ('complete_shard' + str(shard) + '.json')).read_text()) for shard in range(2)]
            expected = benchmark_inventory[dataset]
            assert all(receipt['completed'] and receipt['dataset'] == dataset and receipt['policy_family'] == args.policy
                       and receipt['checkpoint'] == checkpoint for receipt in receipts)
            assert sum(receipt['sequences'] for receipt in receipts) == expected['sequences']
            assert sum(receipt['frames'] for receipt in receipts) == expected['frames']
            verified(phase, {'checkpoint': checkpoint, **expected})
            report_directory = output / role / (dataset + '_report')
            phase = 'score_' + role + '_' + dataset
            run(phase, [['cadtrack_abc.score', '--dataset', dataset, '--root', str(root),
                         '--predictions', str(predictions), '--output', str(report_directory),
                         '--lasher-target-gain', str(args.lasher_target_gain)]], [2])
            report = json.loads((report_directory / 'full_report.json').read_text())
            assert report['completed'] and report['dataset'] == dataset and report['checkpoint'] == checkpoint
            assert report['sequences'] == expected['sequences'] and report['frames'] == expected['frames']
            assert set(report['overall_metrics_percent']) == {'lasher': {'PR', 'NPR', 'SR'},
                                                             'rgbt210': {'PR', 'SR'}, 'rgbt234': {'MPR', 'MSR'}}[dataset]
            formal_reports[role][dataset] = {'report': str(report_directory / 'full_report.json'),
                                             'checkpoint': checkpoint, 'epoch': report['epoch'],
                                             'metrics': report['overall_metrics_percent']}
            if dataset == 'lasher':
                formal_reports[role][dataset]['lasher_stage_goal'] = report['lasher_stage_goal']
            verified(phase, formal_reports[role][dataset])
    selected['formal_metrics_complete'] = True
    selected['formal_reports'] = formal_reports
    (fit / 'selection.json').write_text(json.dumps(selected, indent=2))
    for candidate in fitted['candidates']:
        path = Path(candidate['checkpoint']).resolve()
        assert path.parent == fit.resolve() and path.name == 'candidate_epoch' + str(candidate['epoch']).zfill(2) + '.pth'
        path.unlink()
    assert filecmp.cmp(parent_checkpoint, teacher, shallow=False)
    teacher.unlink()
    state.update(status='FUTURE_ROUND_COMPLETE', best_checkpoint=best, formal_reports=formal_reports,
                 full_three_benchmarks_complete=True, trained_checkpoint_formal_metrics_complete=True,
                 trained_checkpoint_report_role='trained' if 'trained' in formal_reports else 'selected',
                 trained_best_retained_for_result_audit=selected['trained_best_checkpoint'] != best,
                 lasher_stage_goal=formal_reports['selected']['lasher']['lasher_stage_goal'], overall_goal_status_not_changed=True,
                 completed_unix=time.time(), temporary_teacher_cleaned=True)
    store()
    (output / 'pipeline_complete.json').write_text(json.dumps(state, indent=2))
    print(json.dumps({'status': state['status'], 'formal_reports': formal_reports,
                      'lasher_stage_goal': state['lasher_stage_goal']}), flush=True)


if __name__ == '__main__':
    main()
