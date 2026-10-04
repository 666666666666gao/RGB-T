"""Independently rescore completed native predictions against actual dataset GT."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from rgbt import LasHeR, RGBT234

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', choices=('lasher', 'rgbt234'), required=True)
args = p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
dataset = args.dataset
root = Path('/data/gb/outputs/recoverability_best_native_policy_native_full_20261005') / dataset
data = Path('/data/wangwj/dataset/LasHeR/testingset' if dataset == 'lasher' else '/data/zhouy/DATASET/RGB-T234')
read = lambda path: json.loads(Path(path).read_text())
selection = read('/data/gb/setup/best_native_policy_native_selection_20261005.json')
assert selection['status'] == 'PASS' and (root / 'report_completed.txt').is_file()
assert (root / 'mechanism_report/recoverability_metrics_completed.txt').is_file()
evaluator = LasHeR() if dataset == 'lasher' else RGBT234()
names = list(evaluator.ALL)
sequence_count, frame_count = (245, 220703) if dataset == 'lasher' else (234, 116649)
assert len(names) == sequence_count and set(names) == {path.name for path in data.iterdir() if path.is_dir()}
gt = {name: {modality: np.loadtxt(data / name / file, delimiter=',', dtype=np.float32, ndmin=2)
             for modality, file in ({'target': 'init.txt'} if dataset == 'lasher' else {'visible': 'visible.txt', 'infrared': 'infrared.txt'}).items()}
      for name in names}
lengths = {name: len(next(iter(target.values()))) for name, target in gt.items()}
assert sum(lengths.values()) == frame_count
if dataset == 'lasher':
    assert all(np.array_equal(gt[name]['target'], np.asarray(evaluator.seqs_gt[name], dtype=np.float32)) for name in names)
else:
    assert all(target['visible'].shape == target['infrared'].shape for target in gt.values())
thresholds = {'NPR': np.linspace(0, .5, 51), 'PR': np.linspace(0, 50, 51), 'SR': np.linspace(0, 1, 21)} if dataset == 'lasher' else {'MPR': np.linspace(0, 50, 51), 'MSR': np.linspace(0, 1, 21)}
runs = {'baseline': Path('/data/gb/outputs/online_core') / (dataset + '_baseline'),
        'c1': Path('/data/gb/outputs/online_core_v2') / (dataset + '_c1'), 'best_native_policy_complete': root}
report = read(root / 'core_report/full_report.json')
assert report['sequences'] == sequence_count and report['frames'] == frame_count and set(report['variants']) == set(runs)
with (root / 'core_report/per_sequence.csv').open() as stream:
    rows = list(csv.DictReader(stream))
recorded = {(row['variant'], row['sequence']): row for row in rows}
assert len(rows) == len(recorded) == 3 * sequence_count
scores, overall = {}, {}


def geometry(prediction, target):
    pc = (prediction[:, 2:] - 1) / 2 + prediction[:, :2]
    gc = (target[:, 2:] - 1) / 2 + target[:, :2]
    error = ((pc - gc) ** 2).sum(1) ** .5
    normalized = ((pc / (target[:, 2:] + 1e-8) - gc / (target[:, 2:] + 1e-8)) ** 2).sum(1) ** .5
    right = np.minimum(prediction[:, :2] + prediction[:, 2:] - 1, target[:, :2] + target[:, 2:] - 1)
    left = np.maximum(prediction[:, :2], target[:, :2])
    intersection = np.maximum(right - left + 1, 0).prod(1)
    overlap = intersection / (prediction[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection)
    return error, normalized, overlap


for variant, run in runs.items():
    pred = run / 'predictions'
    done = read(pred / 'inference_completion.json')
    assert done['completed'] and not done['smoke_only'] and done['sequences'] == sequence_count and done['frames'] == frame_count
    assert {path.stem for path in pred.glob('*.txt')} == set(names)
    assert len(done['records']) == sequence_count and {row['sequence'] for row in done['records']} == set(names)
    assert all(row['frames'] == lengths[row['sequence']] for row in done['records'])
    curves = {metric: [] for metric in thresholds}
    for name in names:
        prediction = np.loadtxt(pred / (name + '.txt'), dtype=np.float32, ndmin=2).round(0)
        # Preserve all raw output rows; LasHeR's official scorer handles zero-area boxes below.
        assert prediction.shape == (lengths[name], 4) and np.isfinite(prediction).all() and (prediction[1:, 2:] >= 0).all()
        if dataset == 'lasher':
            target = gt[name]['target']
            prediction[0] = target[0]
            # Match NPR/PR/SR_LasHeR on this in-memory copy, without changing prediction files.
            for frame in range(1, len(target)):
                if (prediction[frame, 2:] <= 0).any():
                    prediction[frame] = prediction[frame - 1].copy()
            error, normalized, overlap = geometry(prediction, target)
            values = {'PR': error, 'NPR': normalized, 'SR': overlap}
            for value in values.values():
                value[(target <= 0).any(1)] = -1
        else:
            visible = geometry(prediction, gt[name]['visible'])
            infrared = geometry(prediction, gt[name]['infrared'])
            values = {'MPR': np.minimum(visible[0], infrared[0]), 'MSR': np.maximum(visible[2], infrared[2])}
        for metric, value in values.items():
            curve = (value[:, None] > thresholds[metric] if metric.endswith('SR') else value[:, None] <= thresholds[metric]).mean(0)
            curves[metric].append(curve)
            measured = float(curve.mean() if metric.endswith('SR') else curve[20]) * 100
            assert measured == float(recorded[variant, name][metric]), (variant, name, metric)
    curves = {metric: np.asarray(value) for metric, value in curves.items()}
    overall[variant] = {metric: float(curve.mean() if metric.endswith('SR') else curve.mean(0)[20]) * 100 for metric, curve in curves.items()}
    assert overall[variant] == report['variants'][variant]['overall_metrics_percent']
    for metric, curve in curves.items():
        assert np.array_equal(curve.mean(0), report['variants'][variant]['mean_curves'][metric]['values'])
        assert np.array_equal(thresholds[metric], report['variants'][variant]['mean_curves'][metric]['thresholds'])
    for attribute in evaluator.get_attr_list():
        indices = [names.index(name) for name in getattr(evaluator, attribute)]
        values = {metric: float(curve[indices].mean() if metric.endswith('SR') else curve[indices].mean(0)[20]) * 100 for metric, curve in curves.items()}
        assert {'sequences': len(indices), **values} == report['variants'][variant]['attributes'][attribute]
    scores[variant] = {name: {metric: float(curve[index].mean() if metric.endswith('SR') else curve[index, 20]) * 100
                              for metric, curve in curves.items()} for index, name in enumerate(names)}
    print('NATIVE_VARIANT_PASS', variant, overall[variant], flush=True)

config = read(root / 'predictions/inference_config.json')
assert config['model'] == selection['checkpoint'] and config['head_epoch'] == selection['selected_best_epoch']
assert config['write_verification'] == 'action' and config['full_shard_union_verified']
assert config['max_frames'] == config['limit_sequences'] == config['sequence_offset'] == 0
assert config['validation_split'] is None and not config['zero_init']
bootstrap = {}
for reference in ('baseline', 'c1'):
    folder = root / (reference + '_paired_report')
    paired, saved = read(folder / 'full_report.json'), read(folder / 'paired_bootstrap.json')
    assert list(paired['variants']) == [reference, 'best_native_policy_complete'] and saved['args']['iterations'] == 5000 and saved['args']['seed'] == 42
    for variant in (reference, 'best_native_policy_complete'):
        for field in ('overall_metrics_percent', 'attributes', 'mean_curves'):
            assert paired['variants'][variant][field] == report['variants'][variant][field]
    ordered = sorted(names)
    draws = np.random.default_rng(42).integers(sequence_count, size=(5000, sequence_count))
    result = {}
    for metric in thresholds:
        delta = np.array([scores['best_native_policy_complete'][name][metric] - scores[reference][name][metric] for name in ordered])
        result[metric] = {'mean_delta_percentage_points': float(delta.mean()),
                         'percentile_95_interval_percentage_points': np.percentile(delta[draws].mean(1), (2.5, 97.5)).tolist(),
                         'improved_sequences': int((delta > 0).sum()), 'worsened_sequences': int((delta < 0).sum()), 'tied_sequences': int((delta == 0).sum())}
    assert result == saved['datasets'][dataset]['metrics']
    bootstrap[reference] = result
record = {'status': 'PASS', 'dataset': dataset, 'sequences': sequence_count, 'frames': frame_count,
          'same_checkpoint': selection['checkpoint'], 'native_overall_percent': overall, 'paired_5000_seed42_bootstrap_exact': bootstrap,
          'all_actual_GT_sequence_metrics_attributes_and_curves_exact': True, 'native_metric_function_calls': 0,
          'score_not_another_models_output': True, 'review_independence': 'same-family', 'acceptance_status': 'provisional',
          'execution': {'GPU_queries': 0, 'neural_forward_calls': 0, 'optimizer_steps': 0},
          'scope': 'Independent native metric arithmetic and paired bootstrap; detailed mechanism masses use the original separate collector.'}
(root / 'independent_native_cpu_acceptance.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'dataset': dataset, 'sequences': sequence_count, 'frames': frame_count}))
