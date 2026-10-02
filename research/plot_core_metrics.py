"""Export all native tracking curves and attribute changes from full reports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    variants = list(report['variants'])
    assert len(variants) == 2
    first, second = [report['variants'][v] for v in variants]
    metrics = list(first['mean_curves'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        for name in variants:
            result = report['variants'][name]
            curve = result['mean_curves'][metric]
            axis.plot(curve['thresholds'], np.asarray(curve['values']) * 100,
                      label=f"{name}: {result['overall_metrics_percent'][metric]:.4f}%")
        axis.set(title=f"{report['dataset']} {metric}", ylabel='Sequences averaged (%)',
                 xlabel='IoU threshold' if metric in ('SR', 'MSR') else 'Distance threshold')
        axis.grid(alpha=.25)
        axis.legend()
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out / ('official_curves.' + extension), dpi=180)
    plt.close(fig)
    attributes = list(first['attributes'])
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), max(4, len(attributes) * .25)), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        delta = [second['attributes'][a][metric] - first['attributes'][a][metric] for a in attributes]
        axis.barh(attributes, delta, color=['#197a65' if x >= 0 else '#b13e42' for x in delta])
        axis.axvline(0, color='black', linewidth=.8)
        axis.set(title=metric, xlabel=f'{variants[1]} - {variants[0]} (percentage points)')
        axis.grid(axis='x', alpha=.2)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out / ('attribute_deltas.' + extension), dpi=180)
    plt.close(fig)
    print(json.dumps({'dataset': report['dataset'], 'exported_all_metrics': metrics,
                      'all_attributes': len(attributes), 'output': str(out)}))


if __name__ == '__main__':
    main()
