"""Four full C fits, cached-best and full-endpoint full98 selection, one native model."""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path('/data/gb/GOLA')
PYTHON = '/data/gb/envs/gola/bin/python'
PARENT = '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
SPLIT = '/data/gb/outputs/c1_initial_seed42/split.json'
ARMS = [('H3_lost0', 3, 0.), ('H3_lost01', 3, .1), ('H32_lost0', 32, 0.), ('H32_lost01', 32, .1)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', required=True)
    p.add_argument('--collection-launch', required=True)
    p.add_argument('--review', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reward', choices=('mean_all', 'current_future'), default='mean_all')
    args = p.parse_args()
    arms = ([(arm, horizon, penalty, 'mlp') for arm, horizon, penalty in ARMS]
            if args.reward == 'mean_all' else
            [('H3_current_future_mlp', 3, 0., 'mlp'), ('H32_current_future_mlp', 32, 0., 'mlp'),
             ('H3_current_future_linear', 3, 0., 'linear'), ('H32_current_future_linear', 32, 0., 'linear')])
    assert json.loads(Path(args.review).read_text())['scope'] == 'COMPLETE_STATE_COMMIT_PIPELINE_SOURCE'
    assert json.loads(Path(args.review).read_text())['status'] == 'PASS'
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def record(stage, **extra):
        value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 'elapsed_seconds': time.perf_counter() - started, **extra}
        (root / 'progress.json').write_text(json.dumps(value, indent=2))
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    collection = Path(args.collection)
    pid = json.loads(Path(args.collection_launch).read_text())['pid']
    record('WAITING_ORIGINAL_COLLECTION', collector_pid=pid, interval_seconds=240)
    while not (collection / 'collection_complete.json').is_file():
        proc = Path('/proc') / str(pid) / 'stat'
        assert proc.exists() and proc.read_text().rsplit(')', 1)[1].split()[0] != 'Z', 'Collector ended without complete receipt; no retry'
        time.sleep(240)
    done = json.loads((collection / 'collection_complete.json').read_text())
    assert done['complete'] and done['queries'] == 1789
    record('COLLECTION_ACCEPTED', partitions=done['partitions'])

    def wave(stage, commands):
        children = []
        for gpu, command in enumerate(commands):
            log = (root / f'{stage}_gpu{gpu}.log').open('w')
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log, gpu))
        record(stage.upper(), children=[{'gpu': g, 'pid': c.pid} for c, _, g in children])
        exits = []
        for child, log, gpu in children:
            code = child.wait()
            log.close()
            exits.append((gpu, code))
            record(stage.upper() + '_WORKER_ENDED', gpu=gpu, exit_code=code)
        assert all(code == 0 for _, code in exits), (stage, exits, 'Logs retained; no unchanged retry')

    for stage, epochs in [('fit_sanity', 2), ('fit_full', 60)]:
        commands = []
        for gpu, (arm, horizon, penalty, architecture) in enumerate(arms):
            command = ['bash', 'scripts/run_temporal.sh', str(gpu), 'train_selective_state_commit',
                       '--collection', str(collection), '--output', str(root / stage / arm),
                       '--horizon', str(horizon), '--lost-penalty', str(penalty), '--epochs', str(epochs),
                       '--batch', '128', '--lr', '.001', '--seed', '42',
                       '--reward', args.reward, '--head-architecture', architecture]
            if stage == 'fit_sanity':
                command.append('--sanity')
            commands.append(command)
        wave(stage, commands)
        for arm, _, _, _ in arms:
            result = json.loads((root / stage / arm / 'training_complete.json').read_text())
            config = json.loads((root / stage / arm / 'config.json').read_text())
            assert result['epochs'] == epochs and result['optimizer_updates'] == config['required_optimizer_updates']
            assert result['M0_parent_exact'] and result['strict_reload_pass'] and result['actual_changed_parameters']
            if stage == 'fit_full':
                assert result['last_epoch'] == 60 and result['last_strict_reload_pass']
        record(stage.upper() + '_ACTUAL_TRAINING_PASS')
    candidates = []
    for checkpoint in ('best', 'last'):
        commands = []
        for gpu, (arm, _, _, _) in enumerate(arms):
            label = arm + '_' + checkpoint
            weight = root / 'fit_full' / arm / (checkpoint + '.pth')
            candidates.append((label, arm, checkpoint, weight))
            commands.append(['bash', 'scripts/run_temporal.sh', str(gpu), 'evaluate_recoverability',
                             '--dataset', 'lasher', '--root', '/data/wangwj/dataset/LasHeR/traingset',
                             '--validation-split', SPLIT, '--model', PARENT,
                             '--state-commit-model', str(weight),
                             '--search-value', 'gross', '--write-verification', 'action',
                             '--output', str(root / 'full98' / label / 'predictions')])
        wave('full98_' + checkpoint, commands)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(REPO),
               LD_LIBRARY_PATH='/data/gb/envs/gola/lib', OMP_NUM_THREADS='4')
    references = ['/data/gb/outputs/abc_internal_validation_v1/baseline/predictions',
                  '/data/gb/outputs/abc_internal_validation_v1/c1/predictions',
                  '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/predictions',
                  '/data/gb/outputs/recoverability_search_gross_control_20261006/predictions']
    for label, _, _, _ in candidates:
        receipt = json.loads((root / 'full98' / label / 'predictions/inference_completion.json').read_text())
        assert receipt['completed'] and receipt['sequences'] == 98 and receipt['frames'] == 49418
    with (root / 'full98_CPU_report.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'research.collect_recoverability_metrics',
                        '--dataset', 'lasher', '--root', '/data/wangwj/dataset/LasHeR/traingset', '--split', SPLIT,
                        '--labels', *[label for label, _, _, _ in candidates],
                        '--runs', *[str(root / 'full98' / label / 'predictions') for label, _, _, _ in candidates],
                        '--reference-labels', 'baseline', 'c1', 'old4', 'gross_parent', '--references', *references,
                        '--output', str(root / 'full98_report')], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((root / 'full98_report/full_recoverability_report.json').read_text())
    assert report['completed'] and report['sequences'] == 98
    selected, arm, checkpoint, weight = max(candidates, key=lambda row: report['variants'][row[0]]['sequence_mean_iou'])
    selected_training = json.loads((weight.parent / 'training_complete.json').read_text())
    selection = {'selected_candidate': selected, 'selected_arm': arm, 'selected_checkpoint': checkpoint,
                 'selected_epoch': 60 if checkpoint == 'last' else selected_training['best_epoch'],
                 'state_commit_model': str(weight),
                 'parent_model': PARENT, 'search_value': 'gross', 'same_checkpoint_both_native_datasets': True,
                 'selection_scope': 'One best among eight checkpoints: cached best and full60 endpoint of four configs, by full98 developer sequence IoU before native TEST; not seed stability or untouched confirmation',
                 'sequence_mean_iou': {k: v['sequence_mean_iou'] for k, v in report['variants'].items()},
                 'native_metrics_completed': False}
    (root / 'selected_model.json').write_text(json.dumps(selection, indent=2))
    record('COMPLETE_FOUR60_EPOCH_FITS_FULL98_ONE_MODEL_LOCKED', **selection)
    with (root / 'native_controller.log').open('w') as log:
        subprocess.run([PYTHON, '-u', 'scripts/run_selective_state_native.py', '--selection', str(root / 'selected_model.json'),
                        '--review', args.review, '--output', str(root / 'native')], cwd=REPO,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    native = json.loads((root / 'native/complete_metrics.json').read_text())
    assert native['completed'] and len(native['five_metrics']) == 5
    removed = []
    for stage in ('fit_sanity', 'fit_full'):
        for arm, _, _, _ in arms:
            for checkpoint in ('best',) if stage == 'fit_sanity' else ('best', 'last'):
                weight = root / stage / arm / (checkpoint + '.pth')
                if str(weight) == selection['state_commit_model']:
                    continue
                assert (weight.parent / 'COMPLETE').is_file()
                removed.append({'path': str(weight), 'bytes': weight.stat().st_size})
                weight.unlink()
    assert len(removed) == 11
    (root / 'unused_own_weight_cleanup.json').write_text(json.dumps({'removed': removed,
        'kept': selection['state_commit_model'], 'old4_and_all_dependencies_preserved': True}, indent=2))
    record('COMPLETE_FULL_TRAINING_FULL98_BOTH_NATIVE_FIVE_METRICS', metrics=native['five_metrics'],
           all_five_plus_two=native['all_five_plus_two'], selected_model=selection, cleaned_own_unused_weights=removed)


if __name__ == '__main__':
    main()
