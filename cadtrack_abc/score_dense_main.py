"""Reuse the unchanged official scorer and qualify R3 provenance separately."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['lasher', 'rgbt234', 'rgbt210'], required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline-report', help='Existing pure RGB report. LasHeR uses the supplied scalar reference.')
    parser.add_argument('--toolkit', default='/data/wangwj/RGBT_toolkit-master/src')
    args = parser.parse_args()
    predictions, output = Path(args.predictions), Path(args.output)
    receipts = [json.loads((predictions/f'complete_shard{i}.json').read_text()) for i in range(2)]
    assert all(r['completed'] and r['model_family'] == 'CADTrackDenseABC-R3' and r['epoch'] > 0 for r in receipts)
    assert len({r['checkpoint_sha256'] for r in receipts}) == 1
    assert {r['shard'] for r in receipts} == {0, 1}
    assert sum(r['frames'] for r in receipts) == {'lasher':220703, 'rgbt234':116649, 'rgbt210':104706}[args.dataset]
    command = [sys.executable, '-m', 'cadtrack_abc.score', '--dataset', args.dataset,
        '--root', args.root, '--predictions', args.predictions, '--output', str(output), '--toolkit', args.toolkit]
    subprocess.run(command, check=True)
    raw = output/'full_report.json'
    report = json.loads(raw.read_text())
    if args.dataset == 'lasher':
        baseline = dict(PR=72.74, NPR=69.16, SR=57.84)
        reference = dict(source='User-supplied scalar LasHeR reference; no repeated baseline run or paired CI.')
    else:
        baseline_path = Path(args.baseline_report)
        baseline_report = json.loads(baseline_path.read_text())
        assert baseline_report['completed'] and baseline_report['dataset'] == args.dataset
        baseline = baseline_report['overall_metrics_percent']
        reference = dict(source=str(baseline_path), sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            scope='Existing pure CAD report. Paired uncertainty requires its raw per-sequence predictions.')
    overall = report['overall_metrics_percent']
    reached = {name:overall[name] >= baseline[name] + 2 for name in overall}
    report.update(model_family='CADTrackDenseABC-R3', policy_family=receipts[0]['policy_family'],
        components=receipts[0]['components'], stage=receipts[0]['stage'],
        future_actions_trained=receipts[0]['future_actions_trained'],
        search_actions_trained=receipts[0]['search_actions_trained'],
        checkpoint_sha256=receipts[0]['checkpoint_sha256'],
        raw_report_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        raw_report_unchanged=True, baseline_reference=reference, baseline_metrics_percent=baseline,
        delta_percentage_points={name:overall[name]-baseline[name] for name in overall},
        plus_two_reached=reached, all_dataset_metrics_plus_two=all(reached.values()))
    report['baseline_source'] = reference['source']
    if args.dataset == 'lasher':
        report['mechanism_geometry_scope'] = (
            'Candidate-recall counts in the linked raw mechanism report use un-clipped '
            'float32 pair proposals. R3 evidence-calibration quality labels use the '
            'native clipped candidate boxes. The raw report phrase "as used by training '
            'labels" does not describe these R3 calibration labels; proposal recall '
            'and clipped-label quality have separate geometry conventions.')
    report['timing']['physical_gpus'] = sorted({r['physical_gpu'] for r in receipts})
    report['timing']['scope'] = ('Decode,crop,frozen visual extraction,prompt,trainable fusion/head,'
        'motion prior,zero-or-one extra visual region,candidate decisions,graded state/template update; initialization excluded. Concurrent workers; '
        'not an isolated matched baseline timing comparison.')
    (output/'qualified_full_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(dataset=args.dataset, metrics=overall,
        qualified_report=str(output/'qualified_full_report.json'), plus_two_reached=reached)), flush=True)


if __name__ == '__main__':
    main()
