"""Combine complete native scores and candidate/template measurements.

Native scores remain unchanged. Candidate coordinates retain full precision;
native localization diagnostics use saved three-decimal prediction text.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--candidate-runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    expected = {'lasher': (245, 220703), 'rgbt234': (234, 116649)}[report['dataset']]
    assert (report['sequences'], report['frames']) == expected
    assert report['all_actual_ground_truth_verified']
    seen = set()
    for folder in args.candidate_runs:
        run = Path(folder)
        assert (run / 'job_completed.txt').is_file()
        config = json.loads((run / 'predictions/inference_config.json').read_text())
        receipt = json.loads((run / 'predictions/inference_completion.json').read_text())
        assert receipt['completed'] and not receipt['smoke_only']
        assert receipt['candidate_timeline_recorded']
        assert (receipt['sequences'], receipt['frames']) == expected
        measurements = json.loads((run / 'mechanisms/candidate_mechanisms.json').read_text())
        assert measurements['dataset'] == report['dataset']
        assert measurements['actual_data_root'] == report['actual_data_root']
        variant = config['variant']
        assert variant in report['variants'] and variant not in seen
        assert set(measurements['variants']) == {variant}
        native = report['variants'][variant]
        for key in ('dataset', 'root', 'variant', 'pretrained', 'head_epoch', 'seed', 'amp_dtype'):
            assert config[key] == native['original_inference_config'][key], key
        diagnostic = measurements['variants'][variant]
        assert diagnostic['counters']['valid_tracking_frames'] == native['localization_diagnostics']['valid_tracking_frames']
        assert diagnostic['counters']['failure_events'] == native['localization_diagnostics']['failure_events']
        native['candidate_and_template_diagnostics'] = diagnostic
        native['candidate_diagnostics_source_run'] = str(run)
        if seen:
            assert report['candidate_diagnostic_protocol'] == measurements['protocol']
        else:
            report['candidate_diagnostic_protocol'] = measurements['protocol']
        seen.add(variant)
    assert seen == set(report['variants'])
    report['diagnostic_coordinate_precision'] = (
        'Candidate boxes use full precision; native localization diagnostics use saved three-decimal XYWH. '
        'Reference provenance is recorded per variant; same-run consistency does not prove independent rerun equivalence.')
    report['mechanism_measurements_pending'] = [
        'A learned discriminative compression and memory-budget curves: not implemented.',
        'B multi-future prediction and forecast coverage/calibration: not implemented.',
        'C2/C3 full online state-repair results: only internal prototypes, not these C1 benchmarks.']
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({'completed': True, 'dataset': report['dataset'], 'variants': sorted(seen),
                      'sequences': report['sequences'], 'frames': report['frames'], 'output': str(out)}))


if __name__ == '__main__':
    main()
