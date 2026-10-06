"""Matched one-way/post-search relation training, full videos and native metrics."""
import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.run_candidate_relation_training import environment
from scripts.run_current_policy_aggregation import read, ORIGINAL, PARENT, SPLIT, TRAIN_ROOT, PYTHON, REPO

PREVIOUS = Path('/data/gb/outputs/current_policy_aggregation_20261006')
ARMS = [('one_way_lr5', False, 1e-5), ('post_search_lr5', True, 1e-5),
        ('one_way_lr3', False, 3e-5), ('post_search_lr3', True, 3e-5)]


def merge_initial(root, model):
    """Merge the four disjoint epoch0 development shards without new inference."""
    output = root / 'full98/post_search_initial/predictions'; output.mkdir(parents=True)
    expected = read(ORIGINAL / 'full98/relations_lr5_best/predictions/inference_completion.json')['records']
    records, latencies, completions = [], [], []
    for gpu, offset, count in ((0, 0, 25), (1, 25, 25), (2, 50, 24), (3, 74, 24)):
        folder = root / 'initial_shards' / f'gpu{gpu}'
        done, config = read(folder / 'inference_completion.json'), read(folder / 'inference_config.json')
        assert done['completed'] and done['sequences'] == count
        assert config['model'] == str(model) and config['validation_split'] == SPLIT and config['max_frames'] == 0
        assert config['sequence_offset'] == offset and config['limit_sequences'] == count
        assert [r['sequence'] for r in done['records']] == [r['sequence'] for r in expected[offset:offset + count]]
        for row, original in zip(done['records'], expected[offset:offset + count]):
            assert row['frames'] == original['frames']
            name = row['sequence']
            prediction = np.loadtxt(folder / (name + '.txt'), ndmin=2)
            latency = np.load(folder / (name + '_latency.npy'))
            assert prediction.shape == (row['frames'], 4) and np.isfinite(prediction).all()
            assert latency.shape == (row['frames'] - 1,) and np.isfinite(latency).all() and (latency > 0).all()
            latencies.append(latency)
            for suffix in ('.txt', '_latency.npy', '_recoverability_decisions.npz'):
                os.link(folder / (name + suffix), output / (name + suffix))
        records.extend(done['records']); completions.append(done)
    assert len(records) == 98 and sum(r['frames'] for r in records) == 49418
    for i, row in enumerate(records, 1): row.update(sequence_index=i, sequences=98)
    config.update(output=str(output), sequence_offset=0, limit_sequences=0, smoke_only=True, full_shard_union_verified=True)
    (output / 'inference_config.json').write_text(json.dumps(config, indent=2))
    latency = np.concatenate(latencies)
    receipt = dict(completed=True, sequences=98, frames=49418, smoke_only=True, records=records,
        full_shard_union_verified=True, gt_scoring_completed=False,
        fps_including_decode_crop_update=len(latency) / float(latency.sum()),
        latency_p50_ms=float(np.percentile(latency, 50) * 1000), latency_p95_ms=float(np.percentile(latency, 95) * 1000),
        peak_cuda_allocated_mib=max(c['peak_cuda_allocated_mib'] for c in completions),
        peak_cuda_reserved_mib=max(c['peak_cuda_reserved_mib'] for c in completions),
        wall_seconds_including_initialization_and_diagnostic_serialization=max(c['wall_seconds_including_initialization_and_diagnostic_serialization'] for c in completions),
        timing_note='Four disjoint full-sequence shards; concurrent-load timing is descriptive')
    (output / 'inference_completion.json').write_text(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    review = read(args.review)
    assert review['status'] == 'PASS' and review['scope'] == 'COMPLETE_CANDIDATE_RELATION_PIPELINE_SOURCE'
    assert review['run_scope'] == 'DEPLOYED_POST_SEARCH_BIDIRECTIONAL_CONTROL'
    assert read(PREVIOUS / 'progress.json')['stage'] == 'COMPLETE_CURRENT_POLICY_AGGREGATION_FULL_TRAINING_AND_NATIVE'
    source = read(PREVIOUS / 'plan.json')
    roots = source['old_train_roots'] + source['current_train_roots']
    validation = source['validation']
    assert source['aggregate_queries'] == 2832 and source['aggregate_unique_sequence_query_pairs'] == 2576
    assert Path(PARENT).is_file()
    free = shutil.disk_usage('/data/gb').free
    assert free > 20 * 1024**3
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    values = [tuple(int(x.strip()) for x in line.split(',')) for line in cards.splitlines()]
    assert [v[0] for v in values] == [0, 1, 2, 3] and all(v[1] < 500 and v[2] == 0 for v in values)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def record(stage, **fields):
        value = dict(stage=stage, at_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                     elapsed_seconds=time.perf_counter() - started, **fields)
        (root / 'progress.json').write_text(json.dumps(value, indent=2))
        with (root / 'events.jsonl').open('a') as stream: stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    plan = dict(arms=ARMS, init=PARENT, seed=42, batch_size=336, full_epochs=64,
                optimizer_updates_per_full_arm=576, train_policy_states=2832,
                train_unique_queries=2576, train_sequences=881, validation_queries=196,
                validation_sequences=98, train_roots=roots, validation=validation,
                observed_pair_training=True, disk_free_before_bytes=free,
                search_budget='local plus at most one extra visual forward, 5+5 candidates',
                independent_cases='local-only plus six separate local+one cases; no extra-extra context',
                no_new_parameters=True, epoch0_behavior_change_reported_separately=True,
                selection='one best of eight trained best/last checkpoints plus one explicitly reported bidirectional epoch0; locked before both native datasets')
    (root / 'plan.json').write_text(json.dumps(plan, indent=2))
    record('CAUSAL_PAIR_CHECK')
    with (root / 'causal_pair_check.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'research.check_post_search_relations', '--model', PARENT,
                        '--validation', validation, '--output', str(root / 'causal_pair_check.json')],
                       cwd=REPO, env=environment(0), stdout=log, stderr=subprocess.STDOUT, check=True)
    assert read(root / 'causal_pair_check.json')['completed']

    def wave(stage, commands):
        children = []
        for gpu, command in enumerate(commands):
            log = (root / f'{stage}_gpu{gpu}.log').open('w')
            child = subprocess.Popen(command, cwd=REPO, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log, gpu))
        record(stage.upper(), children=[dict(gpu=gpu, pid=child.pid) for child, _, gpu in children])
        exits = []
        for child, log, gpu in children:
            code = child.wait(); log.close(); exits.append((gpu, code))
        assert all(code == 0 for _, code in exits), (stage, exits, 'Read original log; no unchanged retry')

    for stage, epochs in (('fit_sanity', 2), ('fit_full', 64)):
        commands = []
        for name, bidirectional, lr in ARMS:
            command = [PYTHON, '-u', '-m', 'research.train_recoverability', '--train', *roots,
                '--validation', validation, '--init-checkpoint', PARENT, '--aggregate-policy-states',
                '--epochs', str(epochs), '--batch-size', '336', '--lr', str(lr), '--seed', '42',
                '--threshold', '.03', '--search-supervision', 'oracle', '--action-ranking', 'budgeted',
                '--write-verification', 'action', '--search-value', 'gross', '--observed-pair-training',
                '--retain-last', '--output', str(root / stage / name)]
            if bidirectional: command.append('--post-search-bidirectional')
            commands.append(command)
        wave(stage, commands)
        for name, bidirectional, _ in ARMS:
            done = read(root / stage / name / 'completion.json')
            config = read(root / stage / name / 'config.json')
            assert done['completed'] and done['optimizer_steps'] == epochs * 9
            assert config['train_clips'] == 2832 and config['validation_clips'] == 196
            assert config['initial_checkpoint_epoch'] == 5 and config['initial_checkpoint_weights_exact']
            assert config['post_search_bidirectional'] == bidirectional and config['observed_pair_training']
            assert done['modules_changed'] == {'A': True, 'B': True, 'C': True}
            assert all(v > 0 for v in done['max_module_gradient_norms'].values())
            assert done['relation_parameters_changed'] and done['max_relation_gradient_norm'] > 0
            assert done['strict_reload_metrics_equal'] and done['last_strict_reload_metrics_equal']
            assert done['frozen_c1_gradients_absent']
        record(stage.upper() + '_ABC_RELATIONS_UPDATES_RELOAD_PASS')

    candidates = []
    for checkpoint in ('best', 'last'):
        commands = []
        for name, _, _ in ARMS:
            label = name + '_' + checkpoint
            model = root / 'fit_full' / name / (checkpoint + '.pth')
            candidates.append((label, name, checkpoint, model))
            commands.append([PYTHON, '-u', '-m', 'research.evaluate_recoverability', '--dataset', 'lasher',
                '--root', TRAIN_ROOT, '--validation-split', SPLIT, '--model', str(model),
                '--search-value', 'gross', '--write-verification', 'action',
                '--output', str(root / 'full98' / label / 'predictions')])
        wave('full98_' + checkpoint, commands)
    for label, _, _, _ in candidates:
        done = read(root / 'full98' / label / 'predictions/inference_completion.json')
        assert done['completed'] and (done['sequences'], done['frames']) == (98, 49418)
    initial_model = root / 'fit_full/post_search_lr5/initial.pth'
    initial_commands = []
    for gpu, offset, count in ((0, 0, 25), (1, 25, 25), (2, 50, 24), (3, 74, 24)):
        initial_commands.append([PYTHON, '-u', '-m', 'research.evaluate_recoverability', '--dataset', 'lasher',
            '--root', TRAIN_ROOT, '--validation-split', SPLIT, '--model', str(initial_model),
            '--sequence-offset', str(offset), '--limit-sequences', str(count),
            '--search-value', 'gross', '--write-verification', 'action', '--output', str(root / 'initial_shards' / f'gpu{gpu}')])
    wave('full98_post_search_initial', initial_commands)
    merge_initial(root, initial_model)
    candidates.append(('post_search_initial', 'post_search_lr5', 'initial', initial_model))
    references = ['/data/gb/outputs/abc_internal_validation_v1/baseline/predictions',
        '/data/gb/outputs/abc_internal_validation_v1/c1/predictions',
        '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/predictions',
        '/data/gb/outputs/recoverability_search_gross_control_20261006/predictions',
        str(ORIGINAL / 'full98/relations_lr5_best/predictions')]
    with (root / 'full98_CPU_report.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'research.collect_recoverability_metrics', '--dataset', 'lasher',
            '--root', TRAIN_ROOT, '--split', SPLIT, '--labels', *[c[0] for c in candidates],
            '--runs', *[str(root / 'full98' / c[0] / 'predictions') for c in candidates],
            '--reference-labels', 'baseline', 'c1', 'old4', 'gross_parent', 'current_parent', '--references', *references,
            '--output', str(root / 'full98_report')], cwd=REPO, env=environment(''), stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(root / 'full98_report/full_recoverability_report.json')
    label, arm, checkpoint, model = max(candidates, key=lambda c: report['variants'][c[0]]['sequence_mean_iou'])
    selection = dict(selected_candidate=label, selected_arm=arm, selected_checkpoint=checkpoint,
        selected_epoch=0 if checkpoint == 'initial' else 64 if checkpoint == 'last' else read(model.parent / 'completion.json')['best_epoch'],
        parent_model=str(model), state_commit_model=None, search_value='gross',
        same_checkpoint_both_native_datasets=True, native_metrics_completed=False,
        sequence_mean_iou={k: v['sequence_mean_iou'] for k, v in report['variants'].items()},
        selection_scope='Eight matched trained checkpoints plus explicit bidirectional epoch0 on reused developer98; locked before TEST; epoch0 is not new learning gain')
    (root / 'selected_model.json').write_text(json.dumps(selection, indent=2))
    record('ONE_MODEL_LOCKED_BEFORE_NATIVE', **selection)
    with (root / 'native_controller.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'scripts.run_selective_state_native', '--selection', str(root / 'selected_model.json'),
            '--review', args.review, '--output', str(root / 'native')], cwd=REPO, env=environment(''), stdout=log, stderr=subprocess.STDOUT, check=True)
    native = read(root / 'native/complete_metrics.json')
    assert native['completed'] and native['same_ABC_model_both_datasets'] == str(model)
    removed = []
    for stage in ('fit_sanity', 'fit_full'):
        for name, bidirectional, _ in ARMS:
            folder = root / stage / name
            assert read(folder / 'completion.json')['completed']
            for filename in ('best.pth', 'last.pth') + (('initial.pth',) if bidirectional else ()):
                weight = folder / filename
                if weight == model: continue
                assert weight.resolve().is_relative_to(root.resolve())
                removed.append(dict(path=str(weight), bytes=weight.stat().st_size)); weight.unlink()
    (root / 'unused_own_weight_cleanup.json').write_text(json.dumps(dict(removed=removed, kept=str(model),
        protected_current_parent_old4_GOLA_C1_motion_untouched=True), indent=2))
    record('COMPLETE_POST_SEARCH_RELATION_CONTROL_AND_NATIVE', native=native, selected_model=selection,
           unused_own_weights_removed=len(removed))


if __name__ == '__main__':
    main()
