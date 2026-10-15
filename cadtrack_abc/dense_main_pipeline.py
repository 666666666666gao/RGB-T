"""R3 qualification, matched raw continuation, calibration, policy and full tests.

Start only after the existing R2 pair is terminal and GPUs2/3 are available.
This controller has no retry, background polling, environment install or weight
deletion. A failed child leaves its real logs/exit code and stops the chain.
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
    parser.add_argument('--config', required=True)
    parser.add_argument('--r2-pair-root', required=True)
    parser.add_argument('--train-queries', required=True)
    parser.add_argument('--validation-queries', required=True)
    parser.add_argument('--baseline-234', required=True)
    parser.add_argument('--baseline-210', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, choices=[32, 64, 128], default=128)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    config_path, pair_root = Path(args.config).resolve(), Path(args.r2_pair_root).resolve()
    config = json.loads(config_path.read_text())
    pair = json.loads((pair_root / 'pair_complete.json').read_text())
    assert pair['status'] == 'DENSE_MATCHED_PAIR_FULL_TRAINING_AND_EVALUATION_COMPLETE'
    assert pair['all20_actual_clip_orders_equal']
    selection = json.loads((pair_root / 'HA/fit/selection.json').read_text())
    assert selection['full98_development_selection_complete'] and selection['strict_dense_reload']
    parent = Path(selection['best_checkpoint']).resolve()
    parent_digest = hashlib.sha256(parent.read_bytes()).hexdigest()
    assert parent_digest == selection['selected']['checkpoint_sha256']
    split_path = pair_root / 'HA/fit/manifest.json'
    split = json.loads(split_path.read_text())
    assert len(split['train_sequences']) == 881 and len(split['validation_sequences']) == 98
    assert not set(split['train_sequences']) & set(split['validation_sequences'])
    queries = {name: json.loads(Path(path).read_text()) for name, path in
        [('train', args.train_queries), ('validation', args.validation_queries)]}
    for name, items in queries.items():
        assert {item['sequence'] for item in items} == set(split[name + '_sequences'])
        assert len({(item['sequence'], item['frame']) for item in items}) == len(items)
    capacity = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=index,memory.used,memory.total',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    cards = [[int(value.strip()) for value in line.split(',')] for line in capacity.splitlines()]
    assert {row[0] for row in cards} == {2, 3} and all(row[1] < 1024 for row in cards), cards
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='DENSE_R3_MAIN_PIPELINE_RUNNING', controller_pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(), phases={},
        parent_checkpoint=str(parent), parent_checkpoint_sha256=parent_digest,
        physical_gpus=[2, 3], initial_capacity_csv=capacity,
        query_timestamps_only_reused=True, old_feature_or_action_labels_reused=False,
        full_ABC_training_complete=False, goal_achieved=False)

    def save():
        (output / 'queue_state.json').write_text(json.dumps(state, indent=2))

    def run(name, commands, devices):
        assert len(commands) == len(devices) and len(commands) <= 2
        assert all(device in (2, 3, '') for device in devices)
        phase = dict(status='RUNNING', commands=commands, physical_devices=devices,
            started_utc=datetime.now(timezone.utc).isoformat())
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
        if any(code != 0 for code in codes):
            state['status'] = 'DENSE_R3_MAIN_PIPELINE_FAILED'
        save()
        assert all(code == 0 for code in codes), (name, codes)

    def training_commands(probe):
        destinations = [output / ('raw_probe' if probe else 'raw_fit'),
            output / ('control_probe' if probe else 'control_fit')]
        commands = [
            ['cadtrack_abc.train_dense_motion', '--r2-pair-root', str(pair_root)],
            ['cadtrack_abc.train_dense_continuation', '--parent', str(parent)]]
        for command, destination in zip(commands, destinations):
            command += ['--config', str(config_path), '--output', str(destination),
                '--batch', str(args.batch), '--epochs', '20', '--samples', '20000',
                '--steps', '8', '--workers', '4']
            if probe:
                command.append('--probe')
        commands[0] += ['--localizer-lr', '0.00001', '--temporal-lr', '0.0001']
        commands[1] += ['--head-lr', '0.00001', '--prompt-lr', '0.00001']
        return commands

    save()
    run('native_qualification', [['cadtrack_abc.qualify_dense_main', '--config', str(config_path),
        '--r2-pair-root', str(pair_root), '--output', str(output / 'native_qualification')]], [2])
    native = json.loads((output / 'native_qualification/qualification_complete.json').read_text())
    assert native['status'] == 'DENSE_R3_NATIVE_QUALIFICATION_PASS'
    assert native['parent_checkpoint_sha256'] == parent_digest
    assert native['actual_causal_query_frames'] == 24 and native['actual_matched_actions'] == 210
    assert native['actual_native_replay_equal'] and native['parent_tracker_parameters_unchanged']
    run('raw_and_control_probes', training_commands(True), [2, 3])
    for variant, status in [('raw', 'DENSE_MOTION_QUALIFICATION_COMPLETE'), ('control', 'DENSE_QUALIFICATION_COMPLETE')]:
        result = json.loads((output / (variant + '_probe') / 'fit_complete.json').read_text())
        assert result['status'] == status and result['probe'] and result['updates'] == 2
        assert result['raw_clips'] == args.batch * 2 and all(result['changed_parameter_tensors'].values())
    state['GPU_qualification_complete'] = True
    save()
    run('matched_full20_continuations', training_commands(False), [2, 3])
    raw_fit, control_fit = output / 'raw_fit', output / 'control_fit'
    raw = json.loads((raw_fit / 'fit_complete.json').read_text())
    control = json.loads((control_fit / 'fit_complete.json').read_text())
    for result, status in [(raw, 'DENSE_MOTION_TRAINING_COMPLETE'), (control, 'DENSE_TRAINING_COMPLETE')]:
        assert result['status'] == status and not result['probe']
        assert result['epochs'] == 20 and result['raw_clips'] == 400000
        assert result['updates'] == ((20000 + args.batch - 1) // args.batch) * 20
        assert result['query_frame_observations'] == 3200000
        assert all(result['changed_parameter_tensors'].values())
    raw_history = json.loads((raw_fit / 'history.json').read_text())
    control_history = json.loads((control_fit / 'history.json').read_text())
    assert len(raw_history) == len(control_history) == 20
    assert all(a['clip_order_sha256'] == b['clip_order_sha256'] for a, b in zip(raw_history, control_history))
    assert json.loads((control_fit / 'manifest.json').read_text())['config']['parent_checkpoint_sha256'] == parent_digest
    assert raw['parent_checkpoint_sha256'] == parent_digest
    state['matched20_actual_clip_orders_equal'] = True
    save()
    baselines = ['--baseline-234', str(Path(args.baseline_234).resolve()),
        '--baseline-210', str(Path(args.baseline_210).resolve())]
    run('raw_full98_and_formal', [['cadtrack_abc.evaluate_dense_main', '--fit', str(raw_fit),
        '--split-manifest', str(split_path), '--config', str(config_path),
        '--output', str(output / 'raw_evaluation'), *baselines]], [''])
    raw_evaluation = json.loads((output / 'raw_evaluation/evaluation_complete.json').read_text())
    assert raw_evaluation['status'] == 'DENSE_R3_FULL_VIDEO_EVALUATION_COMPLETE'
    raw_checkpoint = Path(json.loads((raw_fit / 'selection.json').read_text())['trained_best_checkpoint'])

    # The control keeps the original R2 family and evaluator, not R3 labels.
    control_output = output / 'control_evaluation'
    for candidate in control['candidates']:
        checkpoint = Path(candidate['checkpoint'])
        commands = [['cadtrack_abc.validate_dense', '--checkpoint', str(checkpoint),
            '--manifest', str(control_fit / 'manifest.json'), '--output',
            str(control_output / 'development' / checkpoint.stem), '--shard', str(shard)] for shard in (0, 1)]
        run('control_full98_' + checkpoint.stem, commands, [2, 3])
    run('select_control', [['cadtrack_abc.select_dense', '--fit', str(control_fit),
        '--development', str(control_output / 'development')]], [''])
    control_selection = json.loads((control_fit / 'selection.json').read_text())
    assert control_selection['full98_development_selection_complete'] and control_selection['strict_dense_reload']
    selected_control = Path(control_selection['trained_best_checkpoint'])
    control_digest = hashlib.sha256(selected_control.read_bytes()).hexdigest()
    assert control_digest == control_selection['trained_selected']['checkpoint_sha256']
    roots = dict(lasher=Path(config['test_root']), rgbt234=project / 'dataset/rgbt234',
        rgbt210=project / 'dataset/rgbt210/RGBT210')
    reference_reports = dict(rgbt234=Path(args.baseline_234).resolve(), rgbt210=Path(args.baseline_210).resolve())
    control_reports = {}
    for dataset, count, frames in [('lasher', 245, 220703), ('rgbt234', 234, 116649), ('rgbt210', 210, 104706)]:
        predictions = control_output / (dataset + '_predictions')
        run('control_infer_' + dataset, [['cadtrack_abc.infer_dense', '--checkpoint', str(selected_control),
            '--dataset', dataset, '--root', str(roots[dataset]), '--output', str(predictions),
            '--shard', str(shard)] for shard in (0, 1)], [2, 3])
        receipts = [json.loads((predictions / f'complete_shard{shard}.json').read_text()) for shard in (0, 1)]
        assert all(receipt['completed'] and receipt['checkpoint_sha256'] == control_digest for receipt in receipts)
        assert sum(receipt['sequences'] for receipt in receipts) == count
        assert sum(receipt['frames'] for receipt in receipts) == frames
        report_output = control_output / (dataset + '_report')
        command = ['cadtrack_abc.score_dense', '--dataset', dataset, '--root', str(roots[dataset]),
            '--predictions', str(predictions), '--output', str(report_output)]
        if dataset != 'lasher':
            command += ['--baseline-report', str(reference_reports[dataset])]
        run('control_score_' + dataset, [command], [''])
        report_path = report_output / 'qualified_full_report.json'
        report = json.loads(report_path.read_text())
        assert report['completed'] and report['checkpoint_sha256'] == control_digest
        control_reports[dataset] = dict(path=str(report_path), metrics=report['overall_metrics_percent'])
    state['control_reports'] = control_reports
    save()

    # Both authorized cards share collection work. Each shard owns complete
    # sequences, so no two collectors write a query file or tracker prefix.
    def collect(entry, name, teacher):
        teacher_digest = hashlib.sha256(teacher.read_bytes()).hexdigest()
        collection = output / name
        for split_name, items in queries.items():
            root = collection / split_name
            root.mkdir(parents=True, exist_ok=False)
            names = sorted({item['sequence'] for item in items})
            commands = []
            for shard in (0, 1):
                selected_names = set(names[shard::2])
                subset = [item for item in items if item['sequence'] in selected_names]
                query_path = root / f'queries_shard{shard}.json'
                query_path.write_text(json.dumps(subset, indent=2))
                commands.append([entry, '--checkpoint', str(teacher), '--manifest', str(split_path),
                    '--queries', str(query_path), '--split', split_name, '--output', str(root / f'shard{shard}')])
            run(name + '_' + split_name, commands, [2, 3])
            receipts = [json.loads((root / f'shard{shard}/collection_complete.json').read_text()) for shard in (0, 1)]
            assert all(receipt['completed'] and receipt['split'] == split_name and
                receipt['checkpoint_sha256'] == teacher_digest for receipt in receipts)
            records = [dict(record, file=f'shard{shard}/' + record['file'])
                for shard, receipt in enumerate(receipts) for record in receipt['records']]
            assert {(record['sequence'], record['frame']) for record in records} == {(item['sequence'], item['frame']) for item in items}
            assert len(records) == len(items)
            merged = dict(receipts[0], records=records, queries=len(records), physical_gpus=[2, 3],
                physical_gpu='two_disjoint_sequence_shards', merged_shards=2)
            (root / 'collection_complete.json').write_text(json.dumps(merged, indent=2))
        return collection

    evidence = collect('cadtrack_abc.collect_dense_evidence', 'evidence', raw_checkpoint)
    calibration = output / 'calibration_fit'
    run('calibrate_candidate_evidence', [['cadtrack_abc.train_dense_evidence', '--parent', str(raw_checkpoint),
        '--train', str(evidence / 'train'), '--validation', str(evidence / 'validation'),
        '--output', str(calibration), '--epochs', '20', '--batch', '256', '--lr', '0.0001']], [2])
    calibrated = json.loads((calibration / 'fit_complete.json').read_text())
    assert calibrated['status'] == 'DENSE_EVIDENCE_CALIBRATION_COMPLETE' and calibrated['epochs'] == 20
    assert calibrated['strict_candidate_reload']
    policy_parent = Path(calibrated['selected_checkpoint'])
    assert hashlib.sha256(policy_parent.read_bytes()).hexdigest() == calibrated['selected_checkpoint_sha256']
    consequences = collect('cadtrack_abc.collect_dense_consequences', 'consequences', policy_parent)
    policy_fit = output / 'policy_fit'
    run('train_complete_policy', [['cadtrack_abc.train_dense_consequences', '--parent', str(policy_parent),
        '--train', str(consequences / 'train'), '--validation', str(consequences / 'validation'),
        '--output', str(policy_fit), '--epochs', '20', '--batch', '256', '--lr', '0.0001']], [2])
    policy = json.loads((policy_fit / 'fit_complete.json').read_text())
    assert policy['status'] == 'DENSE_CONSEQUENCE_TRAINING_COMPLETE' and policy['epochs'] == 20
    assert policy['future_actions_trained'] and policy['search_actions_trained'] and policy['strict_decision_reload']
    run('main_full98_and_formal', [['cadtrack_abc.evaluate_dense_main', '--fit', str(policy_fit),
        '--split-manifest', str(split_path), '--config', str(config_path),
        '--output', str(output / 'main_evaluation'), *baselines]], [''])
    final = json.loads((output / 'main_evaluation/evaluation_complete.json').read_text())
    assert final['status'] == 'DENSE_R3_FULL_VIDEO_EVALUATION_COMPLETE'
    assert set(final['reports']) == {'lasher', 'rgbt234', 'rgbt210'}
    assert all(report['future_actions_trained'] and report['search_actions_trained'] for report in final['reports'].values())
    state.update(status='DENSE_R3_COMPLETE_TRAINING_AND_EVALUATION',
        ended_utc=datetime.now(timezone.utc).isoformat(), full_ABC_training_complete=True,
        formal_reports=final['reports'], selected_checkpoint=final['selected_checkpoint'],
        selected_checkpoint_sha256=final['selected_checkpoint_sha256'],
        seven_metric_plus_two=final['seven_metric_plus_two'], goal_achieved=False,
        component_effectiveness='Requires matched ablations and result audit; completion of this chain alone is insufficient.',
        weight_cleanup='Defer until real result audit; retain pretrain, parent and selected dependencies.')
    save()
    (output / 'pipeline_complete.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
