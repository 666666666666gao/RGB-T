"""Full official-toolkit scores against actual dataset annotations, not baseline outputs."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['lasher', 'rgbt234'], required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--toolkit', default='/data/wangwj/RGBT_toolkit-master/src')
    args = parser.parse_args()
    sys.path.insert(0, args.toolkit)
    from rgbt import LasHeR, RGBT234
    from rgbt.dataset.basedataset import initial_gt_file
    evaluator = LasHeR() if args.dataset == 'lasher' else RGBT234()
    root, predictions, output = Path(args.root), Path(args.predictions), Path(args.output)
    receipts = [json.loads((predictions / ('complete_shard' + str(i) + '.json')).read_text()) for i in range(2)]
    assert all(r['completed'] for r in receipts)
    assert receipts[0]['checkpoint'] == receipts[1]['checkpoint']
    expected = 245 if args.dataset == 'lasher' else 234
    names = list(evaluator.ALL)
    assert len(names) == expected and {p.name for p in root.iterdir() if p.is_dir()} == set(names)
    assert {p.stem for p in predictions.glob('*.txt')} == set(names)
    records = [r for receipt in receipts for r in receipt['records']]
    assert len(records) == expected and len(set(r['sequence'] for r in records)) == expected
    frame_count = 0
    if args.dataset == 'lasher':
        evaluator.seqs_gt = {name: np.loadtxt(root / name / 'init.txt', delimiter=',', ndmin=2) for name in names}
    else:
        evaluator.seqs_gt = initial_gt_file(str(root), names, 'visible.txt', 'infrared.txt', evaluator.bbox_transfun)
    evaluator.gt_path = str(root)
    for name in names:
        ground_truth = evaluator.seqs_gt[name] if args.dataset == 'lasher' else evaluator.seqs_gt[name]['visible']
        boxes = np.loadtxt(predictions / (name + '.txt'), ndmin=2)
        assert boxes.shape == (len(ground_truth), 4) and np.isfinite(boxes).all()
        frame_count += len(ground_truth)
    assert frame_count == sum(r['frames'] for r in records)
    # LasHeR was inventoried locally. RGBT234's newly downloaded version is
    # inventoried independently by download_rgbt234; do not substitute the old
    # server's different historical frame total for this dataset's actual GT.
    if args.dataset == 'lasher':
        assert frame_count == 220703
    tracker = 'CADTrack_ABC'
    evaluator(tracker, str(predictions))
    metrics = ['NPR', 'PR', 'SR'] if args.dataset == 'lasher' else ['MPR', 'MSR']
    measured = {metric: getattr(evaluator, metric)(tracker) for metric in metrics}
    curves = {metric: np.asarray(value[1]) for metric, value in measured.items()}
    overall = {metric: float(value[0]) * 100 for metric, value in measured.items()}
    output.mkdir(parents=True, exist_ok=True)
    per_sequence, attributes = [], {}
    for index, name in enumerate(names):
        row = {'sequence': name}
        for metric in metrics:
            row[metric] = float(curves[metric][index].mean() if metric in ['SR', 'MSR'] else curves[metric][index, 20]) * 100
        per_sequence.append(row)
    for attribute in evaluator.get_attr_list():
        rows = [names.index(name) for name in getattr(evaluator, attribute)]
        attributes[attribute] = {'sequences': len(rows)}
        for metric in metrics:
            curve = curves[metric][rows]
            attributes[attribute][metric] = float(curve.mean() if metric in ['SR', 'MSR'] else curve.mean(0)[20]) * 100
    latency = np.concatenate([np.load(predictions / (name + '_latency.npy')) for name in names])
    report = {'completed': True, 'dataset': args.dataset, 'checkpoint': receipts[0]['checkpoint'],
              'epoch': receipts[0]['epoch'], 'sequences': expected, 'frames': frame_count,
              'overall_metrics_percent': overall, 'attributes': attributes,
              'mean_curves': {metric: {'thresholds': getattr(evaluator, metric + '_fun').thr.tolist(),
                                       'values': curves[metric].mean(0).tolist()} for metric in metrics},
              'scoring_toolkit': args.toolkit, 'ground_truth_root': str(root),
              'protocol': 'Native rgbt toolkit including prediction rounding and NPR-before-PR/SR order; actual dataset GT.',
              'timing': {'fps': len(latency) / float(latency.sum()), 'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
                         'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
                         'scope': 'Decode,crop,all visual forwards,decision,state/template update; excludes initial frame. Concurrent two-GPU run, not matched baseline timing.'},
              'baseline_source': 'User-provided LasHeR only. No baseline predictions rerun, no paired CI.',
              'per_sequence': per_sequence}
    if args.dataset == 'lasher':
        report['baseline_metrics_percent_user_provided'] = {'PR': 72.74, 'NPR': 69.16, 'SR': 57.84}
        report['delta_percentage_points_vs_user_reference'] = {metric: overall[metric] - report['baseline_metrics_percent_user_provided'][metric] for metric in metrics}
    (output / 'full_report.json').write_text(json.dumps(report, indent=2))
    with (output / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence'] + metrics)
        writer.writeheader(); writer.writerows(per_sequence)
    print(json.dumps({'dataset': args.dataset, 'metrics': overall, 'report': str(output / 'full_report.json')}), flush=True)


if __name__ == '__main__':
    main()
