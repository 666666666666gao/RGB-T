"""One locked old4/gross configuration: full two native benchmarks and actual-GT reports."""
import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path('/data/gb/GOLA')
PYTHON = '/data/gb/envs/gola/bin/python'
MODEL = '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
DATA = {'lasher': '/data/wangwj/dataset/LasHeR/testingset', 'rgbt234': '/data/zhouy/DATASET/RGB-T234'}
EXPECTED = {'lasher': (245, 220703), 'rgbt234': (234, 116649)}


def read(path):
    return json.loads(Path(path).read_text())


def partitions(lengths):
    cumulative = np.r_[0, np.cumsum(lengths)]
    boundaries = [0]
    for part in (1, 2, 3):
        available = np.arange(boundaries[-1] + 1, len(lengths) - (3 - part))
        boundaries.append(int(available[np.argmin(abs(cumulative[available] - cumulative[-1] * part / 4))]))
    boundaries.append(len(lengths))
    return [{'offset': left, 'count': right - left, 'frames': int(cumulative[right] - cumulative[left])}
            for left, right in zip(boundaries, boundaries[1:])]


def worker(root, gpu, stage):
    plan = read(root / 'plan.json')
    for dataset in DATA:
        group = plan['datasets'][dataset]['groups'][gpu]
        run = root / dataset / stage / ('gpu' + str(gpu))
        command = ['bash', 'scripts/run_temporal.sh', str(gpu), 'evaluate_recoverability',
                   '--dataset', dataset, '--root', DATA[dataset], '--model', MODEL,
                   '--write-verification', 'action', '--search-value', 'gross', '--seed', '42',
                   '--sequence-offset', str(group['offset']), '--limit-sequences',
                   '1' if stage == 'sanity' else str(group['count']), '--max-frames',
                   '64' if stage == 'sanity' else '0', '--output', str(run / 'predictions')]
        subprocess.run(command, cwd=REPO, check=True)
        done = read(run / 'predictions/inference_completion.json')
        assert done['completed']
        (run / 'inference_completed.txt').write_text(datetime.now(timezone(timedelta(hours=8))).isoformat())


def merge(root, dataset):
    definition = read(root / 'plan.json')['datasets'][dataset]
    records, times, completions, configs = [], [], [], []
    output = root / dataset / 'predictions'
    output.mkdir()
    for gpu, group in enumerate(definition['groups']):
        shard = root / dataset / 'shards' / ('gpu' + str(gpu))
        assert (shard / 'inference_completed.txt').is_file()
        pred = shard / 'predictions'
        config, done = read(pred / 'inference_config.json'), read(pred / 'inference_completion.json')
        names = definition['names'][group['offset']:group['offset'] + group['count']]
        assert config['dataset'] == dataset and config['root'] == DATA[dataset]
        assert config['model'] == MODEL and config['head_epoch'] == 4 and config['search_value'] == 'gross'
        assert config['write_verification'] == 'action' and config['policy'] == 'learned'
        assert not any(config[key] for key in ('disable_search', 'unsafe_writes', 'zero_init', 'parity_check'))
        assert config['max_frames'] == 0 and config['validation_split'] is None
        assert config['sequence_offset'] == group['offset'] and config['limit_sequences'] == group['count']
        assert done['completed'] and done['sequences'] == group['count'] and done['frames'] == group['frames']
        assert [r['sequence'] for r in done['records']] == names
        assert {p.stem for p in pred.glob('*.txt')} == set(names)
        for row in done['records']:
            assert row['frames'] == definition['lengths'][row['sequence']]
            value = np.loadtxt(pred / (row['sequence'] + '.txt'), ndmin=2)
            latency = np.load(pred / (row['sequence'] + '_latency.npy'))
            assert value.shape == (row['frames'], 4) and np.isfinite(value).all()
            assert latency.shape == (row['frames'] - 1,) and np.isfinite(latency).all() and (latency > 0).all()
            times.append(latency)
            for suffix in ('.txt', '_latency.npy', '_recoverability_decisions.npz'):
                os.link(pred / (row['sequence'] + suffix), output / (row['sequence'] + suffix))
        records.extend(done['records'])
        completions.append(done)
        configs.append(config)
    count, frames = EXPECTED[dataset]
    assert [r['sequence'] for r in records] == definition['names']
    assert len(records) == count and sum(r['frames'] for r in records) == frames
    for index, row in enumerate(records, 1):
        row.update(sequence_index=index, sequences=count)
    config = dict(configs[0], output=str(output), sequence_offset=0, limit_sequences=0,
                  smoke_only=False, full_shard_union_verified=True,
                  source_shards=[str(root / dataset / 'shards' / ('gpu' + str(gpu))) for gpu in range(4)])
    (output / 'inference_config.json').write_text(json.dumps(config, indent=2))
    latency = np.concatenate(times)
    receipt = {'completed': True, 'sequences': count, 'frames': frames, 'smoke_only': False,
               'records': records, 'gt_scoring_completed': False, 'full_shard_union_verified': True,
               'fps_including_decode_crop_update': len(latency) / latency.sum(),
               'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
               'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
               'peak_cuda_allocated_mib': max(r['peak_cuda_allocated_mib'] for r in completions),
               'peak_cuda_reserved_mib': max(r['peak_cuda_reserved_mib'] for r in completions),
               'wall_seconds_including_initialization_and_diagnostic_serialization': max(r['wall_seconds_including_initialization_and_diagnostic_serialization'] for r in completions),
               'timing_note': 'Concurrent four-worker measurement, not isolated algorithm speed.'}
    (output / 'inference_completion.json').write_text(json.dumps(receipt, indent=2))
    acceptance = {'status': 'PASS', 'all_four_actual_shards_completed': True,
                  'all_sequences_full_frames': True, 'sequences': count, 'frames': frames,
                  'same_fixed_checkpoint': MODEL, 'search_value': 'gross'}
    (root / dataset / 'full_inference_merge_acceptance.json').write_text(json.dumps(acceptance, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--worker', type=int, choices=range(4))
    parser.add_argument('--stage', choices=('sanity', 'shards'))
    args = parser.parse_args()
    root = Path(args.output)
    assert root == Path('/data/gb/outputs/recoverability_search_gain_native_full_20261006')
    if args.worker is not None:
        assert args.stage is not None
        worker(root, args.worker, args.stage)
        return
    assert not root.exists() and Path(MODEL).is_file() and read(args.review)['status'] == 'PASS'
    internal = read('/data/gb/outputs/recoverability_search_gross_control_20261006/progress.json')
    assert internal['stage'] == 'COMPLETE_FULL98_ACTUAL_GT'
    assert internal['sequence_mean_iou']['gross_gain'] > internal['sequence_mean_iou']['old4']
    capacity = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    cards = [tuple(int(v.strip()) for v in line.split(',')) for line in capacity.splitlines()]
    assert [row[0] for row in cards] == [0, 1, 2, 3]
    assert all(memory < 1024 and utilization == 0 for _, memory, utilization in cards), capacity
    assert shutil.disk_usage('/data/gb').free > 4 * 1024**3
    root.mkdir(parents=True)
    plan = {'checkpoint': MODEL, 'head_epoch': 4, 'search_value': 'gross', 'new_training_steps': 0,
            'internal_selection': internal, 'datasets': {}}
    for dataset, data in DATA.items():
        paths = sorted(p for p in Path(data).iterdir() if p.is_dir())
        lengths = [sum(p.is_file() for p in (sequence / 'visible').iterdir()) for sequence in paths]
        assert (len(paths), sum(lengths)) == EXPECTED[dataset]
        plan['datasets'][dataset] = {'names': [p.name for p in paths],
                                     'lengths': dict(zip((p.name for p in paths), lengths)),
                                     'groups': partitions(lengths)}
    (root / 'plan.json').write_text(json.dumps(plan, indent=2))
    started = time.monotonic()

    def record(stage, **fields):
        value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 'elapsed_seconds': time.monotonic() - started, **fields}
        (root / 'progress.json').write_text(json.dumps(value, indent=2))
        with (root / 'controller_events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    for stage in ('sanity', 'shards'):
        children = []
        for gpu in range(4):
            log = (root / (stage + '_gpu' + str(gpu) + '.log')).open('w')
            process = subprocess.Popen([PYTHON, '-u', 'scripts/run_search_gain_native.py',
                                        '--output', str(root), '--review', args.review,
                                        '--worker', str(gpu), '--stage', stage],
                                       cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            children.append((process, log, gpu))
        record(stage.upper(), children=[{'gpu': g, 'pid': p.pid} for p, _, g in children])
        for process, log, gpu in children:
            code = process.wait()
            log.close()
            record(stage.upper() + '_WORKER_ENDED', gpu=gpu, exit_code=code)
            assert code == 0, f'{stage} GPU{gpu} failed; original log retained, no retry'
        if stage == 'sanity':
            for dataset in DATA:
                for gpu, group in enumerate(plan['datasets'][dataset]['groups']):
                    done = read(root / dataset / 'sanity' / ('gpu' + str(gpu)) / 'predictions/inference_completion.json')
                    name = plan['datasets'][dataset]['names'][group['offset']]
                    assert done['completed'] and done['sequences'] == 1
                    assert done['records'][0]['sequence'] == name
                    assert done['frames'] == min(64, plan['datasets'][dataset]['lengths'][name])
            record('NATIVE_SANITY_PASS_COMPLETE_BOTH_BENCHMARKS_STARTING')
    record('BOTH_FULL_NATIVE_INFERENCE_ENDED_CPU_REPORTS_STARTING')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', LD_LIBRARY_PATH='/data/gb/envs/gola/lib',
               PYTHONPATH=str(REPO), OMP_NUM_THREADS='4')
    for dataset in DATA:
        merge(root, dataset)
        with (root / (dataset + '_CPU_report.log')).open('w') as log:
            subprocess.run(['bash', 'refine-logs/runs/recoverability_search_gain/report_native_complete.sh', dataset],
                           cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([PYTHON, '-u', 'refine-logs/runs/recoverability_search_gain/audit_native_complete_cpu.py',
                            '--dataset', dataset], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        record('NATIVE_REPORT_ACTUAL_GT_PASS', dataset=dataset)
    metrics = []
    for dataset, metric_names in (('lasher', ('PR', 'NPR', 'SR')), ('rgbt234', ('MPR', 'MSR'))):
        actual = read(root / dataset / 'independent_native_cpu_acceptance.json')
        assert actual['status'] == 'PASS' and actual['same_checkpoint'] == MODEL
        for metric in metric_names:
            values = {key: score[metric] for key, score in actual['native_overall_percent'].items()}
            metrics.append({'dataset': dataset, 'metric': metric, 'percent': values,
                            'delta_vs_baseline_pp': values['gross_gain'] - values['baseline'],
                            'target_percent': values['baseline'] + 2,
                            'meets_plus_two': values['gross_gain'] >= values['baseline'] + 2,
                            'paired_vs_references': {key: value[metric] for key, value in actual['paired_5000_seed42_bootstrap_exact'].items()}})
    result = {'completed': True, 'new_training_steps': 0, 'same_checkpoint_both_datasets': MODEL,
              'search_value': 'gross', 'full_sequences': 479, 'full_frames': 337352,
              'metrics': metrics, 'goal_all_five_plus_two_met': all(row['meets_plus_two'] for row in metrics),
              'best_checkpoint_changed': False, 'development_scope': '98developer videos repeatedly used; historical native results also informed development.',
              'attributes_and_curves': 'Full19LasHeR+12RGBT234 attributes and three paired-reference curves saved under each dataset report.',
              'training_receipt': 'Protected historical old4 checkpoint; this control performs no new training.'}
    (root / 'complete_core_report.json').write_text(json.dumps(result, indent=2))
    record('COMPLETE_BOTH_NATIVE_FULL_FIVE_METRICS_ACTUAL_GT', complete_report=str(root / 'complete_core_report.json'),
           metrics=metrics, goal_all_five_plus_two_met=result['goal_all_five_plus_two_met'])


if __name__ == '__main__':
    main()
