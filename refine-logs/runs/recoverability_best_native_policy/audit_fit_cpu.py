"""Check actual complete module fits and all retained checkpoints on CPU."""
import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from research.recoverability_modules import RecoverabilityModules
from research.train_recoverability import load_data, evaluate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--stage', choices=('m0', 'full'), required=True)
args = p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
torch.set_num_threads(4)
torch.set_grad_enabled(False)
setup = Path('/data/gb/setup')
outputs = Path('/data/gb/outputs')
read = lambda p: json.loads(Path(p).read_text())
parent = torch.load(outputs / 'recoverability_write_pair_reference_own_b384_full_20261004/best.pth', map_location='cpu', weights_only=False)
c1 = torch.load(outputs / 'c1_initial_seed42/best.pth', map_location='cpu', weights_only=False)
epochs = 3 if args.stage == 'm0' else 60
retained = [3] if args.stage == 'm0' else [20, 40, 60]
arms = {}
for arm, ranking, lr in [('pairwise_lr4', 'pairwise', 1e-4), ('pairwise_lr5', 'pairwise', 1e-5),
                         ('budgeted_lr4', 'budgeted', 1e-4), ('budgeted_lr5', 'budgeted', 1e-5)]:
    root = outputs / f'recoverability_best_native_policy_{arm}_{args.stage}_20261005'
    cfg, done, metrics = [read(root / name) for name in ('config.json', 'completion.json', 'metrics.json')]
    assert cfg['epochs'] == done['epochs'] == epochs and cfg['batch_size'] == 384
    assert cfg['train_clips'] == 1762 and cfg['validation_clips'] == 196
    assert cfg['search_supervision'] == 'selector' and cfg['action_ranking'] == ranking and cfg['lr'] == lr
    assert cfg['seed'] == 42 and cfg['write_verification'] == 'action' and not cfg['frozen_modules']
    assert cfg['initial_checkpoint_epoch'] == 4 and cfg['retain_epochs'] == retained
    assert done['completed'] and done['optimizer_steps'] == epochs * math.ceil(1762 / 384)
    assert done['modules_changed'] == {'A': True, 'B': True, 'C': True}
    assert done['frozen_c1_gradients_absent'] and done['strict_reload_metrics_equal']
    assert all(np.isfinite(v) and v > 0 for v in done['max_module_gradient_norms'].values())
    assert [m['epoch'] for m in metrics] == list(range(epochs + 1))
    logs = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    assert [[r['epoch'], r['step'], r['optimizer_steps']] for r in logs] == [
        [e, s, (e - 1) * 5 + s] for e in range(1, epochs + 1) for s in (1, 5)]
    assert all(np.isfinite(r['loss']) and all(np.isfinite(v) and v > 0 for v in r['module_gradient_norms'].values()) for r in logs)
    expected = ['best.pth'] + [f'epoch_{e:03d}.pth' for e in retained]
    assert sorted(f.name for f in root.glob('*.pth')) == sorted(expected) and done['retained_weights'] == expected
    model = RecoverabilityModules(c1).cpu().eval()
    data, _, _, jobs = load_data([cfg['validation']], 'validation', torch.device('cpu'))
    assert len(jobs) == 196 and [list(j) for j in jobs] == cfg['validation_jobs']
    best_epoch = max(range(epochs + 1), key=lambda e: metrics[e]['utility'])
    checkpoints = {}
    for name in expected:
        ck = torch.load(root / name, map_location='cpu', weights_only=False)
        assert ck['module'] == 'ABC_recoverability' and set(ck['model']) == set(parent['model']) and len(ck['model']) == 25
        assert all(torch.isfinite(v).all() for v in ck['model'].values())
        assert all(torch.equal(ck['model']['c1.' + k], v) for k, v in c1['head'].items())
        if name == 'best.pth':
            assert ck['epoch'] == done['best_epoch'] == best_epoch
        else:
            assert ck['epoch'] == int(name[6:9])
        changed = {label: any(not torch.equal(v, parent['model'][k]) for k, v in ck['model'].items() if k.startswith(prefix))
                   for label, prefix in [('A', 'memory.'), ('B', 'search.'), ('C', 'action.')]}
        assert all(changed.values()) if ck['epoch'] > 0 else not any(changed.values())
        model.load_state_dict(ck['model'], strict=True)
        measured, values = evaluate(model, data, 384, .03, details=True, search_supervision='selector',
                                    action_ranking=ranking, write_verification='action')
        target = {k: v for k, v in metrics[ck['epoch']].items() if k != 'epoch'}
        assert target == ck['validation'] and set(measured) == set(target)
        assert all(abs(measured[k] - target[k]) <= 2e-6 for k in target), (arm, name, measured, target)
        if name == 'best.pth':
            with np.load(root / 'best_validation.npz', allow_pickle=False) as saved:
                assert set(saved.files) == set(values)
                for k, v in values.items():
                    assert v.shape == saved[k].shape == (196,)
                    assert np.array_equal(v, saved[k]) if v.dtype.kind in 'biu' else np.allclose(v, saved[k], atol=np.finfo(np.float32).eps, rtol=0)
        checkpoints[name] = {'epoch': ck['epoch'], 'ABC_changed': changed, 'all_C1_tensors_exact': True, 'cached_CPU_replay_pass': True}
    arms[arm] = {'root': str(root), 'checkpoints': checkpoints, 'peak_cuda_mib': done['peak_cuda_mib'], 'elapsed_seconds': done['elapsed_seconds']}
    print('FIT_CPU_PASS', arm, flush=True)
    del model, data, ck, values
record = {'status': 'PASS', 'stage': args.stage, 'all_four_full_fits_CPU_passed': True,
          'epochs_per_arm': epochs, 'optimizer_steps_per_arm': epochs * 5, 'batch_size': 384,
          'train_queries': 1762, 'validation_queries': 196, 'arms': arms,
          'official_metrics_completed': False, 'CUDA_initialized': torch.cuda.is_initialized(),
          'accepted_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
assert not record['CUDA_initialized']
(setup / f'best_native_policy_{args.stage}_fit_cpu_acceptance_20261005.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
