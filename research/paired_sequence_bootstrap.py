"""Paired sequence uncertainty for complete core tracking evaluations.

This describes sequence variation for fixed weights, not training-seed variation.
All metrics of a dataset use the same paired sequence resamples.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    assert args.iterations > 0
    rng = np.random.default_rng(args.seed)
    result = {'method': 'paired equal-weight sequence bootstrap; shared resamples across metrics',
              'scope': 'fixed checkpoints, not repeated training-seed uncertainty',
              'args': vars(args), 'datasets': {}}
    for report_path in args.reports:
        path = Path(report_path)
        report = json.loads(path.read_text())
        dataset = report['dataset']
        expected = {'lasher': (245, 220703), 'rgbt234': (234, 116649)}[dataset]
        assert (report['sequences'], report['frames']) == expected
        assert report['all_actual_ground_truth_verified']
        variants = list(report['variants'])
        assert len(variants) == 2 and dataset not in result['datasets']
        with (path.parent / 'per_sequence.csv').open(newline='') as source:
            rows = list(csv.DictReader(source))
        by_variant = {name: {row['sequence']: row for row in rows if row['variant'] == name}
                      for name in variants}
        assert set(by_variant[variants[0]]) == set(by_variant[variants[1]])
        names = sorted(by_variant[variants[0]])
        assert len(names) == expected[0] and len(rows) == len(names) * 2
        indices = rng.integers(len(names), size=(args.iterations, len(names)))
        metrics = {}
        for metric in report['variants'][variants[0]]['overall_metrics_percent']:
            delta = np.array([float(by_variant[variants[1]][name][metric])
                              - float(by_variant[variants[0]][name][metric]) for name in names])
            assert np.isfinite(delta).all()
            official_delta = (report['variants'][variants[1]]['overall_metrics_percent'][metric]
                              - report['variants'][variants[0]]['overall_metrics_percent'][metric])
            assert np.isclose(delta.mean(), official_delta, rtol=0, atol=1e-8)
            interval = np.percentile(delta[indices].mean(1), (2.5, 97.5))
            metrics[metric] = {'mean_delta_percentage_points': float(delta.mean()),
                               'percentile_95_interval_percentage_points': interval.tolist(),
                               'improved_sequences': int((delta > 0).sum()),
                               'worsened_sequences': int((delta < 0).sum()),
                               'tied_sequences': int((delta == 0).sum())}
        result['datasets'][dataset] = {'sequences': len(names), 'compared_variants': variants,
                                      'metrics': metrics}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({'completed': True, 'datasets': list(result['datasets']), 'output': str(out)}))


if __name__ == '__main__':
    main()
