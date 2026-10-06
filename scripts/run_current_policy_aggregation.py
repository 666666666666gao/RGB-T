"""Four complete ABC/relation fits after current-policy collection, then full metrics."""
import argparse
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.run_candidate_relation_training import environment

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/data/gb/envs/gola/bin/python'
ORIGINAL = Path('/data/gb/outputs/candidate_relation_raw_ABC_20261006_v2')
PARENT = '/data/gb/outputs/candidate_relation_native_recovery_20261006/reconstructed_epoch5/best.pth'
SPLIT = '/data/gb/outputs/c1_initial_seed42/split.json'
TRAIN_ROOT = '/data/wangwj/dataset/LasHeR/traingset'
ARMS = [('old_lr5', False, 1e-5, 72), ('aggregate_lr5', True, 1e-5, 64),
        ('old_lr3', False, 3e-5, 72), ('aggregate_lr3', True, 3e-5, 64)]


def read(path):
    return json.loads(Path(path).read_text())


def merge_validation(collection, root):
    arrays, configs, receipts, jobs = [], [], [], []
    for gpu in range(4):
        folder = collection / 'validation/full' / f'gpu{gpu}'
        config, done = read(folder / 'config.json'), read(folder / 'completion.json')
        assert done['completed'] and done['clips'] == 49 and not done['decision_input_contains_future']
        assert config['partition'] == 'validation' and config['prefix_model'] == PARENT
        assert config['prefix_search_value'] == 'gross' and config['future_policy_mode'] == 'own'
        with np.load(folder / 'samples.npz') as archive:
            data = {key: archive[key].copy() for key in archive.files}
        assert all(len(value) == 49 and np.isfinite(value).all() for value in data.values())
        if arrays:
            assert data.keys() == arrays[0].keys()
            assert all(config[key] == configs[0][key] for key in
                       ('split', 'root', 'cache', 'head', 'pretrained', 'motion_run', 'max_prefix',
                        'regions', 'future_policy', 'future_horizon', 'prefix_search_value', 'prefix_model'))
        arrays.append(data); configs.append(config); receipts.append(done); jobs.extend(config['jobs'])
    assert len({(j['sequence'], j['query_frame']) for j in jobs}) == 196
    expected = read(ORIGINAL / 'fit_full/relations_lr5/config.json')['source_configs'][-1]['jobs']
    indices = {(j['sequence'], j['query_frame']): i for i, j in enumerate(jobs)}
    order = [indices[(j['sequence'], j['query_frame'])] for j in expected]
    data = {key: np.concatenate([part[key] for part in arrays])[order] for key in arrays[0]}
    out = root / 'validation'; out.mkdir()
    np.savez_compressed(out / 'samples.npz', **data)
    with np.load(out / 'samples.npz') as archive:
        assert all(np.array_equal(archive[key], value) for key, value in data.items())
    config = configs[0] | {'output': str(out), 'clips': 196, 'jobs': expected,
                           'source_configs': configs, 'cpu_merge_only': True}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'completion.json').write_text(json.dumps({'completed': True, 'partition': 'validation',
        'clips': 196, 'source_receipts': receipts, 'valid_actions': int(data['action_valid'].sum()),
        'decision_input_contains_future': False, 'official_tracking_accuracy': False,
        'cpu_merge_only': True, 'strict_npz_reload_equal': True}, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--collection-pid', required=True, type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    review = read(args.review)
    assert review['status'] == 'PASS' and review['scope'] == 'COMPLETE_CANDIDATE_RELATION_PIPELINE_SOURCE'
    assert review['run_scope'] == 'DEPLOYED_CURRENT_POLICY_AGGREGATION'
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def record(stage, **fields):
        value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 'elapsed_seconds': time.perf_counter() - started, **fields}
        (root / 'progress.json').write_text(json.dumps(value, indent=2))
        with (root / 'events.jsonl').open('a') as stream: stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    record('WAIT_FOR_CURRENT_POLICY_COLLECTION', collection_owner=args.collection_pid)
    while Path('/proc/' + str(args.collection_pid)).exists():
        time.sleep(240)
    collection = Path(args.collection)
    assert read(collection / 'progress.json')['stage'] == 'COMPLETE_CURRENT_RELATION_POLICY_COLLECTION'
    collected = read(collection / 'plan.json')
    assert collected['model'] == PARENT and collected['train_queries'] == 256 and collected['validation_queries'] == 196
    assert collected['prefix_search_value'] == 'gross' and collected['future_policy'] == 'own'
    disk_free = shutil.disk_usage('/data/gb').free
    assert disk_free > 20 * 1024**3 and Path(PARENT).is_file()
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True)
    values = [tuple(int(x.strip()) for x in line.split(',')) for line in cards.splitlines()]
    assert [v[0] for v in values] == [0, 1, 2, 3] and all(v[1] < 500 and v[2] == 0 for v in values)
    source = read(ORIGINAL / 'fit_full/relations_lr5/config.json')
    assert source['train_clips'] == 2576 and len(source['train']) == 9
    current = [str(collection / 'train/full' / f'gpu{gpu}') for gpu in range(4)]
    for folder in current:
        config, done = read(Path(folder) / 'config.json'), read(Path(folder) / 'completion.json')
        assert config['prefix_model'] == PARENT and config['prefix_model_family'] == 'ABC_candidate_relations'
        assert config['prefix_search_value'] == 'gross' and done['completed'] and done['clips'] == 64
    validation = merge_validation(collection, root)
    (root / 'plan.json').write_text(json.dumps({'arms': ARMS, 'batch_size': 336,
        'optimizer_updates_per_full_arm': 576, 'seed': 42, 'parent': PARENT,
        'old_train_roots': source['train'], 'current_train_roots': current, 'validation': str(validation),
        'aggregate_queries': 2832, 'aggregate_unique_sequence_query_pairs': 2576,
        'disk_free_before_bytes': disk_free, 'output_budget_bytes': 20 * 1024**3,
        'scope': 'Same architecture and pretrained epoch5; data aggregation and learning rate controls; no three-seed gate',
        'selection': 'One best of eight full98 checkpoints, fixed before both full native datasets'}, indent=2))

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
        assert all(code == 0 for _, code in exits), (stage, exits, 'Read original logs; no unchanged retry')

    def fit(stage, sanity):
        commands = []
        for name, aggregate, lr, full_epochs in ARMS:
            epochs = 2 if sanity else full_epochs
            roots = source['train'] + (current if aggregate else [])
            commands.append([PYTHON, '-u', '-m', 'research.train_recoverability', '--train', *roots,
                '--validation', str(validation), '--init-checkpoint', PARENT, '--c1-head', source['c1_head'],
                '--aggregate-policy-states', '--epochs', str(epochs), '--batch-size', '336', '--lr', str(lr),
                '--threshold', '.03', '--seed', '42', '--search-supervision', 'oracle', '--action-ranking', 'budgeted',
                '--write-verification', 'action', '--search-value', 'gross', '--retain-last',
                '--output', str(root / stage / name)])
        wave(stage, commands)
        for name, aggregate, _, full_epochs in ARMS:
            folder = root / stage / name
            done, config, history = read(folder / 'completion.json'), read(folder / 'config.json'), read(folder / 'metrics.json')
            clips, epochs = (2832 if aggregate else 2576), (2 if sanity else full_epochs)
            assert done['completed'] and done['epochs'] == epochs and done['optimizer_steps'] == epochs * math.ceil(clips / 336)
            assert config['module'] == 'ABC_candidate_relations' and config['train_clips'] == clips and config['validation_clips'] == 196
            assert config['initial_checkpoint_epoch'] == 5 and config['initial_checkpoint_weights_exact']
            assert len({j[0] for j in config['train_jobs']}) == 881 and len({j[0] for j in config['validation_jobs']}) == 98
            assert len({(j[0], j[1]) for j in config['train_jobs']}) == 2576
            assert len({j[2] for j in config['train_jobs']}) == (2 if aggregate else 1)
            assert [r['epoch'] for r in history] == list(range(epochs + 1))
            assert done['modules_changed'] == {'A': True, 'B': True, 'C': True} and not done['frozen_modules']
            assert all(v > 0 for v in done['max_module_gradient_norms'].values())
            assert done['relation_parameters_changed'] and done['max_relation_gradient_norm'] > 0
            assert done['frozen_c1_gradients_absent'] and done['strict_reload_metrics_equal'] and done['last_strict_reload_metrics_equal']
        record(stage.upper() + '_ALL_ABC_RELATION_UPDATES_RELOAD_PASS')

    fit('fit_sanity', True)
    fit('fit_full', False)
    candidates = []
    for checkpoint in ('best', 'last'):
        commands = []
        for name, _, _, epochs in ARMS:
            label = name + '_' + checkpoint
            model = root / 'fit_full' / name / (checkpoint + '.pth')
            candidates.append((label, name, checkpoint, model, epochs))
            commands.append([PYTHON, '-u', '-m', 'research.evaluate_recoverability', '--dataset', 'lasher',
                '--root', TRAIN_ROOT, '--validation-split', SPLIT, '--model', str(model),
                '--search-value', 'gross', '--write-verification', 'action',
                '--output', str(root / 'full98' / label / 'predictions')])
        wave('full98_' + checkpoint, commands)
    for label, _, _, _, _ in candidates:
        done = read(root / 'full98' / label / 'predictions/inference_completion.json')
        assert done['completed'] and done['sequences'] == 98 and done['frames'] == 49418
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
    assert report['completed'] and report['sequences'] == 98
    label, name, checkpoint, model, epochs = max(candidates, key=lambda c: report['variants'][c[0]]['sequence_mean_iou'])
    selected_epoch = epochs if checkpoint == 'last' else read(model.parent / 'completion.json')['best_epoch']
    selection = {'selected_candidate': label, 'selected_arm': name, 'selected_checkpoint': checkpoint,
        'selected_epoch': selected_epoch, 'parent_model': str(model), 'state_commit_model': None, 'search_value': 'gross',
        'same_checkpoint_both_native_datasets': True, 'native_metrics_completed': False,
        'sequence_mean_iou': {k: v['sequence_mean_iou'] for k, v in report['variants'].items()},
        'selection_scope': 'Eight completed checkpoints on reused developer98; selection precedes TEST; not seed-stability evidence'}
    (root / 'selected_model.json').write_text(json.dumps(selection, indent=2))
    record('FOUR_COMPLETE_ABC_FITS_EIGHT_FULL98_ONE_MODEL_LOCKED', **selection)
    with (root / 'native_controller.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'scripts.run_selective_state_native', '--selection', str(root / 'selected_model.json'),
            '--review', args.review, '--output', str(root / 'native')], cwd=REPO, env=environment(''), stdout=log, stderr=subprocess.STDOUT, check=True)
    native = read(root / 'native/complete_metrics.json')
    assert native['completed'] and native['same_ABC_model_both_datasets'] == str(model)
    removed = []
    for stage in ('fit_sanity', 'fit_full'):
        for name, _, _, _ in ARMS:
            folder = root / stage / name
            assert read(folder / 'completion.json')['completed']
            for filename in ('best.pth', 'last.pth'):
                weight = folder / filename
                if weight == model: continue
                assert weight.resolve().is_relative_to(root.resolve())
                removed.append({'path': str(weight), 'bytes': weight.stat().st_size}); weight.unlink()
    (root / 'unused_own_weight_cleanup.json').write_text(json.dumps({'removed': removed, 'kept': str(model),
        'protected_current_parent_old4_GOLA_C1_motion_untouched': True}, indent=2))
    record('COMPLETE_CURRENT_POLICY_AGGREGATION_FULL_TRAINING_AND_NATIVE', native=native,
           selected_model=selection, unused_own_weights_removed=len(removed))


if __name__ == '__main__':
    main()
