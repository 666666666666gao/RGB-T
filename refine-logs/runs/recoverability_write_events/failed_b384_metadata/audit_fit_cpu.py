"""Verify the actual matched-update ABC capacity/full fits and retained weights."""
import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from research.recoverability_modules import RecoverabilityModules
from research.train_recoverability import evaluate, load_data


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('m0', 'full'), required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    folder = Path('/data/gb/GOLA/refine-logs/runs/recoverability_write_events')
    plan = read(folder / 'fit_plan.json')
    parent = torch.load(plan['arms'][0]['init_checkpoint'], map_location='cpu', weights_only=False)
    c1 = torch.load('/data/gb/outputs/c1_initial_seed42/best.pth', map_location='cpu', weights_only=False)
    data, _, _, validation_jobs = load_data([plan['arms'][0]['validation']], 'validation', torch.device('cpu'))
    assert len(validation_jobs) == 196
    results = {}
    for arm in plan['arms']:
        output = Path(arm['output_' + args.stage])
        config, completed, metrics = [read(output / name) for name in ('config.json', 'completion.json', 'metrics.json')]
        epochs = arm['m0_epochs'] if args.stage == 'm0' else arm['full_epochs']
        retained = [3] if args.stage == 'm0' else arm['retain_epochs']
        steps_per_epoch = math.ceil(arm['train_queries'] / arm['batch_size'])
        assert config['epochs'] == completed['epochs'] == epochs
        assert config['train'] == arm['train'] and config['validation'] == arm['validation']
        assert config['train_clips'] == len(config['train_jobs']) == arm['train_queries']
        assert config['validation_clips'] == 196 and config['validation_jobs'] == [list(job) for job in validation_jobs]
        for key in ('batch_size', 'lr', 'seed', 'search_supervision', 'action_ranking',
                    'write_verification', 'init_checkpoint', 'frozen_modules'):
            assert config[key] == arm[key], key
        assert config['retain_epochs'] == retained and config['initial_checkpoint_epoch'] == 4
        assert completed['completed'] and completed['optimizer_steps'] == epochs * steps_per_epoch == arm[args.stage + '_optimizer_steps']
        assert completed['modules_changed'] == {'A': True, 'B': True, 'C': True}
        assert completed['frozen_c1_gradients_absent'] and completed['strict_reload_metrics_equal']
        assert all(np.isfinite(value) and value > 0 for value in completed['max_module_gradient_norms'].values())
        assert [row['epoch'] for row in metrics] == list(range(epochs + 1))
        logs = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
        assert [[row['epoch'], row['step'], row['optimizer_steps']] for row in logs] == [
            [epoch, step, (epoch - 1) * steps_per_epoch + step]
            for epoch in range(1, epochs + 1) for step in (1, steps_per_epoch)]
        assert all(np.isfinite(row['loss']) and all(np.isfinite(value) and value > 0
                   for value in row['module_gradient_norms'].values()) for row in logs)
        expected = ['best.pth'] + [f'epoch_{epoch:03d}.pth' for epoch in retained]
        assert sorted(path.name for path in output.glob('*.pth')) == sorted(expected)
        assert completed['retained_weights'] == expected
        model = RecoverabilityModules(c1).cpu().eval()
        checkpoints = {}
        best_epoch = max(range(epochs + 1), key=lambda epoch: metrics[epoch]['utility'])
        for name in expected:
            checkpoint = torch.load(output / name, map_location='cpu', weights_only=False)
            assert checkpoint['module'] == 'ABC_recoverability' and set(checkpoint['model']) == set(parent['model'])
            assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
            assert all(torch.equal(checkpoint['model']['c1.' + key], value) for key, value in c1['head'].items())
            epoch = checkpoint['epoch']
            assert epoch == (best_epoch if name == 'best.pth' else int(name[6:9]))
            if name == 'best.pth':
                assert epoch == completed['best_epoch']
            changed = {label: any(not torch.equal(value, parent['model'][key])
                       for key, value in checkpoint['model'].items() if key.startswith(prefix))
                       for label, prefix in [('A', 'memory.'), ('B', 'search.'), ('C', 'action.')]}
            assert all(changed.values()) if epoch > 0 else not any(changed.values())
            model.load_state_dict(checkpoint['model'], strict=True)
            measured, values = evaluate(model, data, arm['batch_size'], .03, details=True,
                                       search_supervision=arm['search_supervision'],
                                       action_ranking=arm['action_ranking'], write_verification=arm['write_verification'])
            target = {key: value for key, value in metrics[epoch].items() if key != 'epoch'}
            assert target == checkpoint['validation'] and set(measured) == set(target)
            assert all(abs(measured[key] - target[key]) <= 2e-6 for key in target), (arm['arm'], name)
            if name == 'best.pth':
                with np.load(output / 'best_validation.npz', allow_pickle=False) as archive:
                    assert set(archive.files) == set(values)
                    for key, value in values.items():
                        assert value.shape == archive[key].shape == (196,)
                        assert np.array_equal(value, archive[key]) if value.dtype.kind in 'biu' else np.allclose(
                            value, archive[key], atol=np.finfo(np.float32).eps, rtol=0)
            checkpoints[name] = {'epoch': epoch, 'ABC_changed': changed, 'cached_CPU_replay_pass': True}
        results[arm['arm']] = {'output': str(output), 'epochs': epochs,
                              'optimizer_steps': completed['optimizer_steps'], 'train_queries': arm['train_queries'],
                              'checkpoints': checkpoints, 'peak_cuda_mib': completed['peak_cuda_mib'],
                              'elapsed_seconds': completed['elapsed_seconds']}
        print('FIT_CPU_PASS', arm['arm'], flush=True)
        del model, checkpoint, values
    assert not torch.cuda.is_initialized()
    receipt = {'status': 'PASS', 'stage': args.stage, 'all_four_ABC_fits_passed': True, 'arms': results,
               'validation_queries': 196, 'official_metrics_completed': False, 'CUDA_initialized': False,
               'completed_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
    (Path('/data/gb/setup') / ('write_events_' + args.stage + '_fit_cpu_acceptance_20261005.json')).write_text(
        json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
