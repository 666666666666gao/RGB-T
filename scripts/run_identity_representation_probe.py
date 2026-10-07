"""Own one four-GPU paired representation probe, sanity then full A pretraining."""
import datetime
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path('/data/gb/outputs/identity_representation_probe_20261007')
PYTHON = '/data/gb/envs/gola/bin/python'
REPO = Path('/data/gb/GOLA')
ARMS = [('pre_lr4', 'pre', 1e-4), ('encoded_lr4', 'encoded', 1e-4),
        ('pre_lr5', 'pre', 1e-5), ('encoded_lr5', 'encoded', 1e-5)]


def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def wave(stage, jobs):
    handles, logs, children = [], [], []
    for gpu, args in enumerate(jobs):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TORCH_HOME='/data/gb/cache/torch',
                   OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        log = (ROOT / (stage + '_gpu' + str(gpu) + '.log')).open('w')
        command = [PYTHON, '-m', 'research.identity_representation_probe', *args]
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        handles.append(child)
        logs.append(log)
        children.append(dict(gpu=gpu, pid=child.pid, command=command))
    (ROOT / 'progress.json').write_text(json.dumps(dict(stage=stage, at_cst=stamp(), children=children), indent=2))
    return_codes = [handle.wait() for handle in handles]
    for log in logs:
        log.close()
    assert all(code == 0 for code in return_codes), return_codes


def extract_corpus(stage, views, limit):
    for partition in ('train', 'validation'):
        wave(stage + '_' + partition.upper(), [[
            '--stage', 'extract', '--output', str(ROOT / stage / ('gpu' + str(gpu)) / partition),
            '--partition', partition, '--views', str(views), '--shard', str(gpu), '--shards', '4',
            '--batch-size', '64', '--workers', '4', '--limit-sequences', str(limit)] for gpu in range(4)])
    return [str(ROOT / stage / ('gpu' + str(gpu))) for gpu in range(4)]


def fit_corpus(stage, inputs, epochs, batch):
    wave(stage, [['--stage', 'fit', '--output', str(ROOT / stage / name), '--inputs', *inputs,
                 '--source', source, '--lr', str(lr), '--epochs', str(epochs), '--batch-size', str(batch)]
                for name, source, lr in ARMS])
    receipts = {name: json.loads((ROOT / stage / name / 'completion.json').read_text()) for name, _, _ in ARMS}
    assert all(row['completed'] and row['epochs'] == epochs and row['optimizer_steps'] > 0
               and row['parameters_changed'] and row['strict_reload_metrics_equal'] for row in receipts.values())
    assert len({row['optimizer_steps'] for row in receipts.values()}) == 1
    return receipts


def main():
    assert not ROOT.exists()
    assert shutil.disk_usage(ROOT.parent).free > 3 * 1024**3
    memory = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True).stdout
    assert len(memory.splitlines()) == 4
    assert all(int(line.split(',')[1]) < 500 for line in memory.splitlines())
    ROOT.mkdir()
    started = time.perf_counter()
    plan = dict(created_cst=stamp(), seed=42, arms=ARMS, views_per_sequence=16,
                train_sequences=881, validation_sequences=98, train_queries=14096, validation_queries=1568,
                full_identity_epochs=32, fitted_component='A identity projection only',
                main_ABC_not_replaced=True, native_accuracy=False,
                feature_visual_budget='Identical query forward reused; one fixed self-context per sequence initialization',
                frozen_GOLA_and_C1=True, alpha=.1, temperature=.1,
                paired_labels='Actual sampled GT IoU: positive >=0.5, negative <0.2; not full distractor identity labels',
                corpus='Existing supervised frame-pair sampling and template perturbation; not own-policy rollouts',
                validation_reused_for_model_selection=True)
    (ROOT / 'plan.json').write_text(json.dumps(plan, indent=2))
    wave('REAL_SOURCE_CHECK', [['--stage', 'check', '--output', str(ROOT / 'source_check')]])
    assert json.loads((ROOT / 'source_check/completion.json').read_text())['completed']
    sanity_inputs = extract_corpus('sanity_features', 8, 16)
    sanity = fit_corpus('fit_sanity', sanity_inputs, 2, 32)
    inputs = extract_corpus('full_features', 16, 0)
    full = fit_corpus('fit_full', inputs, 32, 512)
    val = []
    for folder in inputs:
        with np.load(Path(folder) / 'validation/samples.npz') as archive:
            val.append({key: archive[key].copy() for key in ('sequence_index', 'valid', 'quality', 'c1_score')})
    data = {key: np.concatenate([row[key] for row in val]) for key in val[0]}
    assert len(data['valid']) == 1568 and len(np.unique(data['sequence_index'])) == 98
    c1_choice = np.where(data['valid'], data['c1_score'], -np.inf).argmax(-1)
    c1_iou = data['quality'][np.arange(len(c1_choice)), c1_choice]
    rng = np.random.default_rng(42)
    resamples = rng.integers(0, 98, (5000, 98))
    results = {}
    for name, source, lr in ARMS:
        picked = np.load(ROOT / 'fit_full' / name / 'best_selected_iou.npy')
        delta = np.array([(picked[data['sequence_index'] == i] - c1_iou[data['sequence_index'] == i]).mean()
                          for i in np.unique(data['sequence_index'])]) * 100
        results[name] = dict(full[name], source=source, lr=lr, delta_vs_C1_query_iou_pp=float(delta.mean()),
                             paired_sequence_95_interval=np.percentile(delta[resamples].mean(1), [2.5, 97.5]).tolist())
    selected = max(results, key=lambda name: results[name]['validation_metrics']['selected_query_mean_iou'])
    kept = ROOT / 'fit_full' / selected / 'best.pth'
    removed = []
    for stage in ('fit_sanity', 'fit_full'):
        for name, _, _ in ARMS:
            path = ROOT / stage / name / 'best.pth'
            if path != kept:
                assert path.resolve().is_relative_to(ROOT.resolve())
                removed.append(dict(path=str(path), bytes=path.stat().st_size))
                path.unlink()
    report = dict(completed=True, identity_only_pretraining=True, native_accuracy=False,
                  complete_main_ABC_gain_not_established=True, results=results,
                  same_checkpoint_initialization=True, matched_updates=True,
                  c1_query_mean_iou=float(c1_iou.mean()), sanity=sanity,
                  selected=selected, kept=str(kept), removed=removed,
                  elapsed_seconds=time.perf_counter() - started)
    (ROOT / 'complete_report.json').write_text(json.dumps(report, indent=2))
    (ROOT / 'progress.json').write_text(json.dumps(dict(stage='COMPLETE_IDENTITY_REPRESENTATION_PROBE',
        at_cst=stamp(), selected=selected, fitted_component='A projection; no new native results'), indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
