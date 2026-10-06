"""Matched raw ABC fits, actual parent parity, full-video selection, one native model."""
import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.run_search_gain_native import partitions

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/data/gb/envs/gola/bin/python'
PARENT = '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
SPLIT = '/data/gb/outputs/c1_initial_seed42/split.json'
TRAIN_ROOT = '/data/wangwj/dataset/LasHeR/traingset'
GROSS = Path('/data/gb/outputs/recoverability_search_gross_control_20261006/predictions')
SOURCE = 'refine-logs/runs/recoverability_write_events/native_complete/training/budgeted_lr4/config.json'
ARMS = [('ABC_lr5', False, 1e-5), ('relations_lr5', True, 1e-5),
        ('ABC_lr4', False, 1e-4), ('relations_lr4', True, 1e-4)]


def read(path):
    return json.loads(Path(path).read_text())


def environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                LD_LIBRARY_PATH='/data/gb/envs/gola/lib', PYTHONPATH=str(REPO),
                TORCH_HOME='/data/gb/cache/torch', XDG_CACHE_HOME='/data/gb/cache',
                TMPDIR='/data/gb/cache/tmp', OMP_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    review = read(args.review)
    assert review['status'] == 'PASS' and review['scope'] == 'COMPLETE_CANDIDATE_RELATION_PIPELINE_SOURCE'
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    source = read(REPO / SOURCE)
    assert len(source['train']) == 9 and source['init_checkpoint'] == PARENT
    assert not source['frozen_modules'] and source['batch_size'] == 336
    started = time.perf_counter()

    def record(stage, **fields):
        value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 'elapsed_seconds': time.perf_counter() - started, **fields}
        (root / 'progress.json').write_text(json.dumps(value, indent=2))
        with (root / 'events.jsonl').open('a') as stream: stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    def wave(stage, commands):
        children = []
        for gpu, command in enumerate(commands):
            log = (root / f'{stage}_gpu{gpu}.log').open('w')
            child = subprocess.Popen(command, cwd=REPO, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log, gpu))
        record(stage.upper(), children=[{'gpu': gpu, 'pid': child.pid} for child, _, gpu in children])
        exits = []
        for child, log, gpu in children:
            code = child.wait(); log.close(); exits.append((gpu, code))
            record(stage.upper() + '_WORKER_ENDED', gpu=gpu, exit_code=code)
        assert all(code == 0 for _, code in exits), (stage, exits, 'Primary logs retained; no unchanged retry')

    def fit(stage, epochs):
        commands = []
        for arm, relations, lr in ARMS:
            command = [PYTHON, '-u', '-m', 'research.train_recoverability', '--train', *source['train'],
                       '--validation', source['validation'], '--c1-head', source['c1_head'],
                       '--init-checkpoint', PARENT, '--epochs', str(epochs), '--batch-size', '336',
                       '--lr', str(lr), '--seed', '42', '--threshold', '.03',
                       '--search-supervision', 'oracle', '--action-ranking', 'budgeted',
                       '--write-verification', 'action', '--search-value', 'gross', '--retain-last',
                       '--output', str(root / stage / arm)]
            if relations: command.append('--candidate-relations')
            commands.append(command)
        wave(stage, commands)
        for arm, relations, _ in ARMS:
            folder = root / stage / arm
            done, config, history = read(folder / 'completion.json'), read(folder / 'config.json'), read(folder / 'metrics.json')
            assert done['completed'] and done['epochs'] == epochs
            assert config['train_clips'] == 2576 and config['validation_clips'] == 196
            assert len(set(job[0] for job in config['train_jobs'])) == 881
            assert len(set(job[0] for job in config['validation_jobs'])) == 98
            assert done['optimizer_steps'] == epochs * math.ceil(2576 / 336)
            assert [row['epoch'] for row in history] == list(range(epochs + 1))
            assert done['modules_changed'] == {'A': True, 'B': True, 'C': True}
            assert all(value > 0 for value in done['max_module_gradient_norms'].values())
            assert done['frozen_c1_gradients_absent'] and done['strict_reload_metrics_equal']
            assert done['last_strict_reload_metrics_equal'] and not done['frozen_modules']
            if relations:
                assert done['initial_parent_all_output_tensors_exact']
                assert done['relation_parameters_changed'] and done['max_relation_gradient_norm'] > 0
                assert (folder / 'initial.pth').is_file()
        record(stage.upper() + '_ALL_ABC_GRADIENTS_UPDATES_RELOAD_PASS', epochs=epochs, updates_per_arm=epochs * 8)

    fit('fit_sanity', 2)
    names = sorted(read(SPLIT)['validation'])
    lengths = [sum(path.is_file() for path in (Path(TRAIN_ROOT) / name / 'visible').iterdir()) for name in names]
    assert len(names) == 98 and sum(lengths) == 49418
    groups = partitions(lengths)
    initial = root / 'fit_sanity/relations_lr5/initial.pth'
    parity_commands = []
    for gpu, group in enumerate(groups):
        parity_commands.append([PYTHON, '-u', '-m', 'research.evaluate_recoverability',
                               '--dataset', 'lasher', '--root', TRAIN_ROOT, '--validation-split', SPLIT,
                               '--model', str(initial), '--search-value', 'gross', '--write-verification', 'action',
                               '--sequence-offset', str(group['offset']), '--limit-sequences', str(group['count']),
                               '--output', str(root / 'parent_parity' / f'gpu{gpu}' / 'predictions')])
    wave('parent_parity', parity_commands)
    parity_names = []
    fields = ('choice', 'original_choice', 'pause', 'searched_region', 'extra_executed',
              'template_updated', 'template_source_frame', 'forecast_reference', 'memory_target_commit')
    for gpu, group in enumerate(groups):
        folder = root / 'parent_parity' / f'gpu{gpu}' / 'predictions'
        done, config = read(folder / 'inference_completion.json'), read(folder / 'inference_config.json')
        expected = names[group['offset']:group['offset'] + group['count']]
        assert done['completed'] and done['sequences'] == len(expected) and done['frames'] == group['frames']
        assert config['model'] == str(initial) and not config['zero_init'] and not config['parity_check']
        assert [row['sequence'] for row in done['records']] == expected
        for name in expected:
            assert (folder / (name + '.txt')).read_bytes() == (GROSS / (name + '.txt')).read_bytes(), name
            with np.load(folder / (name + '_recoverability_decisions.npz')) as actual, np.load(GROSS / (name + '_recoverability_decisions.npz')) as reference:
                assert all(np.array_equal(actual[field], reference[field]) for field in fields), name
        parity_names.extend(expected)
    assert parity_names == names
    (root / 'actual_parent_parity.json').write_text(json.dumps({'passed': True, 'sequences': 98, 'frames': 49418,
        'actual_saved_initial_model': str(initial), 'parent': PARENT, 'parent_predictions': str(GROSS),
        'predictions_byte_exact': True, 'decision_fields_exact': fields,
        'legacy_C1_parity_or_zero_init_used': False}, indent=2))
    record('ACTUAL_PRETRAINED_INITIAL_FULL98_PARENT_PARITY_PASS')
    fit('fit_full', 60)
    candidates = []
    for checkpoint in ('best', 'last'):
        commands = []
        for arm, _, _ in ARMS:
            label = arm + '_' + checkpoint
            model = root / 'fit_full' / arm / (checkpoint + '.pth')
            candidates.append((label, arm, checkpoint, model))
            commands.append([PYTHON, '-u', '-m', 'research.evaluate_recoverability', '--dataset', 'lasher',
                             '--root', TRAIN_ROOT, '--validation-split', SPLIT, '--model', str(model),
                             '--search-value', 'gross', '--write-verification', 'action',
                             '--output', str(root / 'full98' / label / 'predictions')])
        wave('full98_' + checkpoint, commands)
    for label, _, _, _ in candidates:
        done = read(root / 'full98' / label / 'predictions/inference_completion.json')
        assert done['completed'] and done['sequences'] == 98 and done['frames'] == 49418
    references = ['/data/gb/outputs/abc_internal_validation_v1/baseline/predictions',
                  '/data/gb/outputs/abc_internal_validation_v1/c1/predictions',
                  '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/predictions', str(GROSS)]
    with (root / 'full98_CPU_report.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'research.collect_recoverability_metrics', '--dataset', 'lasher',
                        '--root', TRAIN_ROOT, '--split', SPLIT, '--labels', *[row[0] for row in candidates],
                        '--runs', *[str(root / 'full98' / row[0] / 'predictions') for row in candidates],
                        '--reference-labels', 'baseline', 'c1', 'old4', 'gross_parent', '--references', *references,
                        '--output', str(root / 'full98_report')], cwd=REPO, env=environment(''),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(root / 'full98_report/full_recoverability_report.json')
    assert report['completed'] and report['sequences'] == 98
    label, arm, checkpoint, model = max(candidates, key=lambda row: report['variants'][row[0]]['sequence_mean_iou'])
    done = read(model.parent / 'completion.json')
    improves_parent = report['variants'][label]['sequence_mean_iou'] > report['variants']['gross_parent']['sequence_mean_iou']
    selection = {'selected_candidate': label, 'selected_arm': arm, 'selected_checkpoint': checkpoint,
                 'selected_epoch': 60 if checkpoint == 'last' else done['best_epoch'], 'parent_model': str(model),
                 'state_commit_model': None, 'search_value': 'gross', 'same_checkpoint_both_native_datasets': True,
                 'native_metrics_completed': False, 'improves_protected_parent_on_developer_full98': improves_parent,
                 'sequence_mean_iou': {key: value['sequence_mean_iou'] for key, value in report['variants'].items()},
                 'selection_scope': 'One best of eight full98 models before both native tests; reused developer98, not seed stability or untouched confirmation'}
    (root / 'selected_model.json').write_text(json.dumps(selection, indent=2))
    record('FOUR_FULL_ABC_FITS_EIGHT_FULL98_ONE_MODEL_LOCKED', **selection)
    if improves_parent:
        with (root / 'native_controller.log').open('w') as log:
            subprocess.run([PYTHON, '-u', '-m', 'scripts.run_selective_state_native', '--selection', str(root / 'selected_model.json'),
                            '--review', args.review, '--output', str(root / 'native')], cwd=REPO, env=environment(''),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        native = read(root / 'native/complete_metrics.json')
        assert native['completed'] and native['same_ABC_model_both_datasets'] == str(model)
    else:
        parent_root = Path('/data/gb/outputs/state_commit_current_future_20261006')
        parent_selection = read(parent_root / 'selected_model.json')
        parent_native = read(parent_root / 'native/complete_metrics.json')
        assert read(parent_root / 'progress.json')['stage'] == 'COMPLETE_FULL_TRAINING_FULL98_BOTH_NATIVE_FIVE_METRICS'
        assert parent_selection['parent_model'] == PARENT and parent_selection['search_value'] == 'gross'
        assert parent_selection['selected_epoch'] == 0 and parent_native['completed']
        assert len(parent_native['five_metrics']) == 5
        for dataset, expected in [('lasher', (245, 220703)), ('rgbt234', (234, 116649))]:
            core = read(parent_root / 'native' / dataset / 'core_report/full_report.json')
            assert core['all_actual_ground_truth_verified'] and (core['sequences'], core['frames']) == expected
            assert core['variants']['state_commit']['overall_metrics_percent'] == core['variants']['gross_parent']['overall_metrics_percent']
            for metric in [row for row in parent_native['five_metrics'] if row['dataset'] == dataset]:
                assert metric['percent'] == core['variants']['gross_parent']['overall_metrics_percent'][metric['metric']]
        native = {'new_native_inference_executed': False, 'reason': 'No complete-video improvement over protected parent; no unchanged formal replay.',
                  'retained_incumbent': PARENT, 'retained_incumbent_search_value': 'gross',
                  'completed_parent_formal_reference': str(parent_root / 'native/complete_metrics.json'),
                  'verified_completed_parent_five_metrics': parent_native['five_metrics'],
                  'all_five_plus_two': parent_native['all_five_plus_two']}
        (root / 'native_not_replayed.json').write_text(json.dumps(native, indent=2))
    kept = str(model) if improves_parent else None
    removed = []
    for stage in ('fit_sanity', 'fit_full'):
        for name, relations, _ in ARMS:
            folder = root / stage / name
            assert read(folder / 'completion.json')['completed']
            for filename in (['best.pth', 'last.pth', 'initial.pth'] if relations else ['best.pth', 'last.pth']):
                weight = folder / filename
                if str(weight) == kept: continue
                assert weight.resolve().is_relative_to(root.resolve())
                removed.append({'path': str(weight), 'bytes': weight.stat().st_size}); weight.unlink()
    (root / 'unused_own_weight_cleanup.json').write_text(json.dumps({'removed': removed, 'kept': kept,
        'protected_parent_and_all_dependencies_preserved': True}, indent=2))
    record('COMPLETE_RAW_ABC_TRAINING_FULL98_AND_NATIVE_DECISION', native=native, selected_model=selection,
           developer_full98_parent_point_improvement=improves_parent, unused_own_weights_removed=len(removed))


if __name__ == '__main__':
    main()
