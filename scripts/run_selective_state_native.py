"""One developer-selected checkpoint: both full native benchmarks, all five metrics."""
import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.run_search_gain_native import partitions

REPO = Path('/data/gb/GOLA')
PYTHON = '/data/gb/envs/gola/bin/python'
DATA = {'lasher': '/data/wangwj/dataset/LasHeR/testingset', 'rgbt234': '/data/zhouy/DATASET/RGB-T234'}
EXPECTED = {'lasher': (245, 220703), 'rgbt234': (234, 116649)}


def read(path):
    return json.loads(Path(path).read_text())


def merge(root, dataset, definition, selection):
    output = root / dataset / 'predictions'
    output.mkdir()
    records, latencies, completions, configs = [], [], [], []
    for gpu, group in enumerate(definition['groups']):
        path = root / dataset / 'shards' / f'gpu{gpu}' / 'predictions'
        config, done = read(path / 'inference_config.json'), read(path / 'inference_completion.json')
        names = definition['names'][group['offset']:group['offset'] + group['count']]
        assert config['model'] == selection['parent_model'] and config['state_commit_model'] == selection['state_commit_model']
        assert config['dataset'] == dataset and config['root'] == DATA[dataset] and config['search_value'] == 'gross'
        assert config['write_verification'] == 'action' and config['policy'] == 'learned'
        assert not any(config[k] for k in ('unsafe_writes', 'disable_search', 'zero_init', 'parity_check'))
        assert config['validation_split'] is None and config['max_frames'] == 0
        assert config['sequence_offset'] == group['offset'] and config['limit_sequences'] == group['count']
        assert done['completed'] and done['sequences'] == group['count'] and done['frames'] == group['frames']
        assert [r['sequence'] for r in done['records']] == names
        assert {p.stem for p in path.glob('*.txt')} == set(names)
        for row in done['records']:
            name, frames = row['sequence'], row['frames']
            assert frames == definition['lengths'][name]
            prediction = np.loadtxt(path / (name + '.txt'), ndmin=2)
            latency = np.load(path / (name + '_latency.npy'))
            assert prediction.shape == (frames, 4) and np.isfinite(prediction).all()
            assert latency.shape == (frames - 1,) and np.isfinite(latency).all() and (latency > 0).all()
            latencies.append(latency)
            for suffix in ('.txt', '_latency.npy', '_recoverability_decisions.npz'):
                os.link(path / (name + suffix), output / (name + suffix))
        records.extend(done['records'])
        configs.append(config)
        completions.append(done)
    count, frames = EXPECTED[dataset]
    assert [r['sequence'] for r in records] == definition['names']
    assert len(records) == count and sum(r['frames'] for r in records) == frames
    for index, record in enumerate(records, 1):
        record.update(sequence_index=index, sequences=count)
    config = dict(configs[0], output=str(output), sequence_offset=0, limit_sequences=0,
                  smoke_only=False, full_shard_union_verified=True)
    (output / 'inference_config.json').write_text(json.dumps(config, indent=2))
    latency = np.concatenate(latencies)
    done = {'completed': True, 'sequences': count, 'frames': frames, 'smoke_only': False, 'records': records,
            'full_shard_union_verified': True, 'gt_scoring_completed': False,
            'fps_including_decode_crop_update': len(latency) / float(latency.sum()),
            'latency_p50_ms': float(np.percentile(latency, 50) * 1000), 'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
            'peak_cuda_allocated_mib': max(v['peak_cuda_allocated_mib'] for v in completions),
            'peak_cuda_reserved_mib': max(v['peak_cuda_reserved_mib'] for v in completions),
            'wall_seconds_including_initialization_and_diagnostic_serialization': max(v['wall_seconds_including_initialization_and_diagnostic_serialization'] for v in completions),
            'timing_note': 'Concurrent four-worker measurement; not isolated algorithm speed'}
    (output / 'inference_completion.json').write_text(json.dumps(done, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', required=True)
    p.add_argument('--review', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    selection, review = read(args.selection), read(args.review)
    assert review['status'] == 'PASS' and review['scope'] == 'COMPLETE_STATE_COMMIT_PIPELINE_SOURCE'
    assert selection['same_checkpoint_both_native_datasets'] and not selection['native_metrics_completed']
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    plan = {}
    for dataset, data in DATA.items():
        paths = sorted(p for p in Path(data).iterdir() if p.is_dir())
        lengths = [sum(p.is_file() for p in (s / 'visible').iterdir()) for s in paths]
        assert (len(paths), sum(lengths)) == EXPECTED[dataset]
        plan[dataset] = {'names': [s.name for s in paths], 'lengths': dict(zip((s.name for s in paths), lengths)),
                         'groups': partitions(lengths)}
    (root / 'plan.json').write_text(json.dumps({'selection': selection, 'datasets': plan}, indent=2))

    def record(stage, **fields):
        value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(), **fields}
        (root / 'progress.json').write_text(json.dumps(value, indent=2))
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    for stage in ('sanity', 'shards'):
        # Each dataset occupies all four cards; both use the same locked head.
        for dataset, data in DATA.items():
            children = []
            for gpu, group in enumerate(plan[dataset]['groups']):
                out = root / dataset / stage / f'gpu{gpu}' / 'predictions'
                log = (root / f'{dataset}_{stage}_gpu{gpu}.log').open('w')
                command = ['bash', 'scripts/run_temporal.sh', str(gpu), 'evaluate_recoverability',
                           '--dataset', dataset, '--root', data, '--model', selection['parent_model'],
                           '--state-commit-model', selection['state_commit_model'], '--search-value', 'gross',
                           '--write-verification', 'action', '--sequence-offset', str(group['offset']),
                           '--limit-sequences', '1' if stage == 'sanity' else str(group['count']),
                           '--max-frames', '64' if stage == 'sanity' else '0', '--output', str(out)]
                child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                children.append((child, log, gpu, out))
            record(dataset.upper() + '_' + stage.upper(), children=[{'gpu': g, 'pid': c.pid} for c, _, g, _ in children])
            exits = []
            for child, log, gpu, out in children:
                code = child.wait()
                log.close()
                exits.append((gpu, code))
            assert all(code == 0 for _, code in exits), (dataset, stage, exits, 'Original logs retained; no retry')
            for _, _, gpu, out in children:
                done = read(out / 'inference_completion.json')
                assert done['completed']
                if stage == 'sanity':
                    name = plan[dataset]['names'][plan[dataset]['groups'][gpu]['offset']]
                    assert done['sequences'] == 1 and done['records'][0]['sequence'] == name
                    assert done['frames'] == min(64, plan[dataset]['lengths'][name])
            record(dataset.upper() + '_' + stage.upper() + '_PASS')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', LD_LIBRARY_PATH='/data/gb/envs/gola/lib', PYTHONPATH=str(REPO), OMP_NUM_THREADS='4')
    metrics = []
    for dataset, data in DATA.items():
        merge(root, dataset, plan[dataset], selection)
        run = root / dataset
        references = {'baseline': f'/data/gb/outputs/online_core/{dataset}_baseline',
                      'c1': f'/data/gb/outputs/online_core_v2/{dataset}_c1',
                      'old4': f'/data/gb/outputs/recoverability_write_pair_native_{dataset}_own_b384_full_20261004',
                      'gross_parent': f'/data/gb/outputs/recoverability_search_gain_native_full_20261006/{dataset}'}
        with (root / f'{dataset}_CPU_report.log').open('w') as log:
            def cpu(module, *arguments):
                subprocess.run([PYTHON, '-u', '-m', module, *arguments], cwd=REPO, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            cpu('research.collect_core_metrics', '--dataset', dataset, '--data-root', data,
                '--variants', *references, 'state_commit', '--runs', *references.values(), str(run), '--output', str(run / 'core_report'))
            for label, reference in references.items():
                pair = run / f'{label}_paired_report'
                cpu('research.collect_core_metrics', '--dataset', dataset, '--data-root', data,
                    '--variants', label, 'state_commit', '--runs', reference, str(run), '--output', str(pair))
                cpu('research.paired_sequence_bootstrap', '--reports', str(pair / 'full_report.json'), '--output', str(pair / 'paired_bootstrap.json'))
                cpu('research.plot_core_metrics', '--report', str(pair / 'full_report.json'), '--output', str(pair))
            cpu('research.collect_recoverability_metrics', '--dataset', dataset, '--root', data,
                '--labels', 'state_commit', '--runs', str(run / 'predictions'), '--reference-labels', *references,
                '--references', *[r + '/predictions' for r in references.values()], '--output', str(run / 'mechanism_report'))
        report = read(run / 'core_report/full_report.json')
        assert report['all_actual_ground_truth_verified'] and (report['sequences'], report['frames']) == EXPECTED[dataset]
        values = report['variants']['state_commit']['overall_metrics_percent']
        assert len(report['variants']['state_commit']['attributes']) == (19 if dataset == 'lasher' else 12)
        assert set(report['variants']['state_commit']['mean_curves']) == set(values)
        baseline = report['variants']['baseline']['overall_metrics_percent']
        for metric in (('PR', 'NPR', 'SR') if dataset == 'lasher' else ('MPR', 'MSR')):
            metrics.append({'dataset': dataset, 'metric': metric, 'percent': values[metric],
                            'baseline_percent': baseline[metric], 'delta_vs_baseline_pp': values[metric] - baseline[metric],
                            'target_percent': baseline[metric] + 2, 'meets_plus_two': values[metric] >= baseline[metric] + 2})
        record('FULL_NATIVE_ACTUAL_GT_REPORT_COMPLETE', dataset=dataset, metrics=values)
    assert len(metrics) == 5
    complete = {'completed': True, 'same_state_commit_model_both_datasets': selection['state_commit_model'],
                'five_metrics': metrics, 'all_five_plus_two': all(m['meets_plus_two'] for m in metrics),
                'complete_attribute_settings': 31, 'fixed_checkpoint_paired_bootstrap': '5000/seed42 versus GOLA/C1/old4/gross_parent',
                'not_training_seed_stability': True}
    (root / 'complete_metrics.json').write_text(json.dumps(complete, indent=2))
    record('COMPLETE_BOTH_NATIVE_FULL_FIVE_METRICS_ACTUAL_GT', **complete)


if __name__ == '__main__':
    main()
