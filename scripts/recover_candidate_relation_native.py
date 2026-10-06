"""Reconstruct the prematurely removed selected epoch, verify it, finish native metrics.

The original four full60 runs and eight full98 results remain immutable. This is
recovery of their preselected epoch5, not a new hyperparameter search or full60 fit.
"""
import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.run_search_gain_native import partitions
from scripts.run_candidate_relation_training import environment

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/data/gb/envs/gola/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = Path(args.original_root)
    review = read(args.review)
    assert review['status'] == 'PASS' and review['scope'] == 'COMPLETE_CANDIDATE_RELATION_PIPELINE_SOURCE'
    assert read(original / 'progress.json')['stage'] == 'COMPLETE_RAW_ABC_TRAINING_FULL98_AND_NATIVE_DECISION'
    selected = read(original / 'selected_model.json')
    assert selected['selected_arm'] == 'relations_lr5' and selected['selected_checkpoint'] == 'best'
    assert selected['selected_epoch'] == 5 and not selected['native_metrics_completed']
    old_weight = Path(selected['parent_model'])
    assert not old_weight.exists()
    config = read(old_weight.parent / 'config.json')
    old_done = read(old_weight.parent / 'completion.json')
    assert old_done['completed'] and old_done['epochs'] == 60 and old_done['optimizer_steps'] == 480
    assert old_done['best_epoch'] == 5 and config['candidate_relations']
    assert not config['frozen_modules'] and not config['write_pair_calibration'] and not config['prefer_last_prefix']
    assert config['batch_size'] == 336 and config['seed'] == 42 and config['lr'] == 1e-5
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()

    def record(stage, **fields):
        value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(), **fields}
        (root / 'progress.json').write_text(json.dumps(value, indent=2))
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    rebuilt = root / 'reconstructed_epoch5'
    command = [PYTHON, '-u', '-m', 'research.train_recoverability', '--train', *config['train'],
               '--validation', config['validation'], '--c1-head', config['c1_head'],
               '--init-checkpoint', config['init_checkpoint'], '--epochs', '5',
               '--batch-size', str(config['batch_size']), '--lr', str(config['lr']),
               '--weight-decay', str(config['weight_decay']), '--seed', str(config['seed']),
               '--threshold', str(config['threshold']), '--search-supervision', config['search_supervision'],
               '--action-ranking', config['action_ranking'], '--write-verification', config['write_verification'],
               '--search-value', config['search_value'], '--candidate-relations', '--retain-last', '--output', str(rebuilt)]
    (root / 'reconstruction_plan.json').write_text(json.dumps({'original_selection': selected,
        'original_full_training': old_done, 'command': command, 'started_cst': started,
        'scope': 'Reconstruct selected epoch5 with its original constant-LR prefix40 updates; not a new full60 training run'}, indent=2))
    with (root / 'reconstruction.log').open('w') as log:
        child = subprocess.Popen(command, cwd=REPO, env=environment(1), stdout=log, stderr=subprocess.STDOUT)
        record('RECONSTRUCT_EPOCH5', children=[{'gpu': 1, 'pid': child.pid}])
        assert child.wait() == 0, 'Read reconstruction.log; no unchanged retry'
    done = read(rebuilt / 'completion.json')
    assert done['completed'] and done['epochs'] == done['best_epoch'] == 5 and done['optimizer_steps'] == 40
    assert done['strict_reload_metrics_equal'] and done['last_strict_reload_metrics_equal']
    assert all(done['modules_changed'][k] and done['max_module_gradient_norms'][k] > 0 for k in ('A', 'B', 'C'))
    assert done['relation_parameters_changed'] and done['frozen_c1_gradients_absent']
    prior_records, rebuilt_records = read(old_weight.parent / 'metrics.json')[:6], read(rebuilt / 'metrics.json')
    assert len(prior_records) == len(rebuilt_records) == 6
    for old, current in zip(prior_records, rebuilt_records):
        assert old.keys() == current.keys()
        for key in old:
            assert abs(old[key] - current[key]) <= 1e-6, (old['epoch'], key, old[key], current[key])
    with np.load(old_weight.parent / 'best_validation.npz') as old, np.load(rebuilt / 'best_validation.npz') as new:
        assert old.files == new.files
        assert all(np.array_equal(old[key], new[key]) for key in old.files)
    record('RECONSTRUCTION_PREFIX40_AND_SAVED_QUERY_RESULTS_PASS')

    original_inference = read(original / 'full98' / selected['selected_candidate'] / 'predictions' / 'inference_config.json')
    split = read(original_inference['validation_split'])
    data_root = Path(original_inference['root'])
    names = sorted(split['validation'])
    lengths = [sum(p.is_file() for p in (data_root / name / 'visible').iterdir()) for name in names]
    assert len(names) == 98 and sum(lengths) == 49418
    groups = partitions(lengths)
    children = []
    for gpu, group in enumerate(groups):
        output = root / 'parity_full98' / f'gpu{gpu}' / 'predictions'
        command = [PYTHON, '-u', '-m', 'research.evaluate_recoverability', '--dataset', 'lasher', '--root', str(data_root),
                   '--model', str(rebuilt / 'best.pth'), '--pretrained', config['source_configs'][0]['pretrained'],
                   '--c1-head', config['c1_head'], '--motion-run', config['source_configs'][0]['motion_run'],
                   '--policy', 'learned', '--write-verification', 'action',
                   '--search-value', 'gross', '--validation-split', config['source_configs'][0]['split'],
                   '--sequence-offset', str(group['offset']), '--limit-sequences', str(group['count']), '--output', str(output)]
        log = (root / f'parity_full98_gpu{gpu}.log').open('w')
        child = subprocess.Popen(command, cwd=REPO, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
        children.append((child, log, gpu, group, output))
    record('VERIFY_RECONSTRUCTED_FULL98', children=[{'gpu': g, 'pid': c.pid} for c, _, g, _, _ in children])
    for child, log, gpu, _, _ in children:
        code = child.wait()
        log.close()
        assert code == 0, (gpu, 'Read original parity log; no retry')
    old_predictions = original / 'full98' / selected['selected_candidate'] / 'predictions'
    fields = ('choice', 'original_choice', 'pause', 'searched_region', 'extra_executed',
              'template_updated', 'template_source_frame', 'forecast_reference', 'memory_target_commit')
    checked = []
    for _, _, _, group, folder in children:
        receipt = read(folder / 'inference_completion.json')
        inference_config = read(folder / 'inference_config.json')
        subset = names[group['offset']:group['offset'] + group['count']]
        assert receipt['completed'] and receipt['sequences'] == group['count'] and receipt['frames'] == group['frames']
        assert inference_config['model'] == str(rebuilt / 'best.pth') and inference_config['head_epoch'] == 5
        assert inference_config['max_frames'] == 0 and not inference_config['zero_init'] and not inference_config['parity_check']
        assert {p.stem for p in folder.glob('*.txt')} == set(subset)
        for name in subset:
            assert (folder / (name + '.txt')).read_bytes() == (old_predictions / (name + '.txt')).read_bytes(), name
            with np.load(folder / (name + '_recoverability_decisions.npz')) as new, np.load(old_predictions / (name + '_recoverability_decisions.npz')) as old:
                assert all(np.array_equal(old[key], new[key]) for key in fields), name
            checked.append(name)
    assert sorted(checked) == names
    recovery = {'passed': True, 'original_four_full60_runs_replayed': False, 'prefix_epochs': 5, 'prefix_optimizer_steps': 40,
                'original_full_training_epochs': 60, 'original_full_training_optimizer_steps': 480,
                'epoch0_to5_validation_records_match_atol': 1e-6, 'saved_validation_arrays_exact': True,
                'full98_predictions_byte_exact': True, 'full98_decision_fields_exact': fields,
                'sequences': 98, 'frames': 49418, 'scope': 'Reconstructed epoch5 verified on original developer98; no claim of archived original weight-byte equality'}
    (root / 'reconstruction_verification.json').write_text(json.dumps(recovery, indent=2))
    record('RECONSTRUCTED_SELECTED_MODEL_FULL98_PARITY_PASS', **recovery)
    selection = dict(selected, parent_model=str(rebuilt / 'best.pth'), recovered_original_model=str(old_weight),
                     reconstruction_verification=str(root / 'reconstruction_verification.json'),
                     native_metrics_completed=False, same_checkpoint_both_native_datasets=True)
    (root / 'selected_model.json').write_text(json.dumps(selection, indent=2))
    # The selected trained model is protected regardless of its native result.
    with (root / 'native_controller.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'scripts.run_selective_state_native', '--selection', str(root / 'selected_model.json'),
                        '--review', args.review, '--output', str(root / 'native')], cwd=REPO,
                       env=environment(''), stdout=log, stderr=subprocess.STDOUT, check=True)
    native = read(root / 'native/complete_metrics.json')
    assert native['completed'] and native['same_ABC_model_both_datasets'] == str(rebuilt / 'best.pth')
    removed = []
    for filename in ('initial.pth', 'last.pth'):
        path = rebuilt / filename
        assert path.resolve().is_relative_to(root.resolve())
        removed.append({'path': str(path), 'bytes': path.stat().st_size})
        path.unlink()
    (root / 'unused_reconstruction_weight_cleanup.json').write_text(json.dumps({'removed': removed,
        'kept': str(rebuilt / 'best.pth'), 'protected_original_parent_and_dependencies_preserved': True}, indent=2))
    record('COMPLETE_RECONSTRUCTED_SELECTED_RAW_ABC_BOTH_NATIVE', native=native, reconstructed_model=selection,
           unused_own_weights_removed=len(removed))


if __name__ == '__main__':
    main()
