"""Keep the official scorer and score template pollution at its actual source."""
import argparse
import json
from pathlib import Path

import numpy as np

from .mechanism_summary import box_iou_xywh, ratio
from .score import main as official_main


def summarize_sources(root, predictions, names, dataset):
    counts = dict(requests=0, commit_frames=0, valid_commit_frames=0, incorrect_source_commit_frames=0)
    modal_commits = [0, 0]
    for name in names:
        annotation = 'visible.txt' if dataset == 'rgbt234' else 'init.txt'
        truth = np.loadtxt(root / name / annotation, delimiter=',', ndmin=2)[1:]
        records = json.loads((predictions / (name + '_mechanisms.json')).read_text())
        assert len(truth) == len(records)
        for target, record in zip(truth, records):
            committed = np.asarray(record['template_commit_rgb_tir'], dtype=bool)
            counts['requests'] += int(record['template_write_requested'])
            counts['commit_frames'] += int(committed.any())
            modal_commits = [a + int(b) for a, b in zip(modal_commits, committed)]
            if not (target[2:] > 0).all() or not committed.any():
                continue
            source = record['template_control']['source_box_xywh']
            assert source is not None
            counts['valid_commit_frames'] += 1
            counts['incorrect_source_commit_frames'] += int(box_iou_xywh(np.round(source, 3), target) < .2)
    return dict(counts=counts, modal_commits_rgb_tir=modal_commits,
                incorrect_template_commit_fraction=ratio(counts['incorrect_source_commit_frames'], counts['valid_commit_frames']),
                definition='Actual template source box rounded to3 decimals, IoU<0.2, positive-size GT; localization proxy, not semantic identity. Initialization excluded.')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    args, _ = parser.parse_known_args()
    predictions, output, root = Path(args.predictions), Path(args.output), Path(args.root)
    receipts = [json.loads((predictions / f'complete_shard{index}.json').read_text()) for index in range(2)]
    fields = ('checkpoint', 'epoch', 'policy_family', 'template_policy',
              'resolved_template_interval', 'resolved_template_threshold')
    assert all(receipts[0][key] == receipts[1][key] for key in fields)
    official_main()
    report_path = output / 'full_report.json'
    report = json.loads(report_path.read_text())
    summary = summarize_sources(root, predictions, [item['sequence'] for item in report['per_sequence']], args.dataset)
    (output / 'template_source_report.json').write_text(json.dumps(summary, indent=2))
    report['template_update_control'] = {key: receipts[0][key] for key in fields}
    report['template_update_control']['source_report'] = str(output / 'template_source_report.json')
    if args.dataset == 'lasher':
        mechanism_path = Path(report['mechanism_report'])
        mechanism = json.loads(mechanism_path.read_text())
        mechanism['counts']['incorrect_template_commit_frames'] = summary['counts']['incorrect_source_commit_frames']
        assert mechanism['counts']['template_commit_frames'] == summary['counts']['valid_commit_frames']
        mechanism['incorrect_template_commit_fraction'] = summary['incorrect_template_commit_fraction']
        mechanism['definitions']['commits'] = ('Actual modal commits; template errors use actual template-source IoU<0.2. '
                                             'Memory errors still use selected-box IoU<0.2. Localization proxies, not semantic identity.')
        mechanism_path.write_text(json.dumps(mechanism, indent=2))
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
