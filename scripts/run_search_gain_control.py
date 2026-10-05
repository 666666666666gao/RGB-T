"""Fixed old4: verify legacy parity, then gross-gain search on full 98 developer videos."""
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
MODEL = Path('/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth')
REFERENCE = MODEL.parent / 'predictions'
SPLIT = Path('/data/gb/outputs/c1_initial_seed42/split.json')
DATA = Path('/data/wangwj/dataset/LasHeR/traingset')
PYTHON = '/data/gb/envs/gola/bin/python'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--review', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists() and MODEL.is_file()
    assert read(Path(args.review))['status'] == 'PASS'
    capacity = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    cards = [tuple(int(value.strip()) for value in line.split(',')) for line in capacity.splitlines()]
    assert [row[0] for row in cards] == [0, 1, 2, 3]
    assert all(memory < 1024 and utilization == 0 for _, memory, utilization in cards), capacity
    assert shutil.disk_usage('/data/gb').free > 2 * 1024**3
    split = read(SPLIT)
    names = sorted(split['validation'])
    assert len(names) == 98 and not set(names) & set(split['train'])
    assert {p.name for p in DATA.iterdir() if p.is_dir() and p.name in names} == set(names)
    reference_config = read(REFERENCE / 'inference_config.json')
    assert reference_config['model'] == str(MODEL) and reference_config['write_verification'] == 'action'
    assert not reference_config['disable_search']
    output.mkdir(parents=True)
    started = time.monotonic()

    def record(stage, **fields):
        value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 'elapsed_seconds': time.monotonic() - started, **fields}
        (output / 'progress.json').write_text(json.dumps(value, indent=2) + '\n')
        with (output / 'controller_events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    def wave(stage, jobs, max_frames):
        children = []
        for gpu, offset, count, mode, destination in jobs:
            command = ['bash', 'scripts/run_temporal.sh', str(gpu), 'evaluate_recoverability',
                       '--dataset', 'lasher', '--root', str(DATA), '--model', str(MODEL),
                       '--validation-split', str(SPLIT), '--write-verification', 'action',
                       '--search-value', mode, '--sequence-offset', str(offset),
                       '--limit-sequences', str(count), '--max-frames', str(max_frames),
                       '--output', str(destination)]
            log = (output / (stage + '_gpu' + str(gpu) + '.log')).open('w')
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            children.append((process, log, gpu, destination))
        record(stage, children=[{'pid': p.pid, 'gpu': gpu, 'output': str(dest)}
                                for p, _, gpu, dest in children])
        for process, log, gpu, destination in children:
            code = process.wait()
            log.close()
            record(stage + '_WORKER_ENDED', gpu=gpu, exit_code=code)
            assert code == 0, f'{stage} GPU{gpu} failed; inspect original worker log'
            assert read(destination / 'inference_completion.json')['completed']

    sanity = [(gpu, gpu // 2, 1, 'weighted' if gpu % 2 == 0 else 'gross',
               output / 'sanity' / ('gpu' + str(gpu))) for gpu in range(4)]
    wave('SANITY', sanity, 96)
    for gpu, offset, _, mode, destination in sanity:
        receipt = read(destination / 'inference_completion.json')
        assert receipt['sequences'] == 1 and receipt['records'][0]['sequence'] == names[offset]
        actual = np.loadtxt(destination / (names[offset] + '.txt'), ndmin=2)
        if mode == 'weighted':
            expected = np.loadtxt(REFERENCE / (names[offset] + '.txt'), ndmin=2)[:len(actual)]
            assert np.array_equal(actual, expected), 'Legacy old4 prediction parity failed'
        record('SANITY_VERIFIED', gpu=gpu, mode=mode, frames=len(actual),
               stats=receipt['records'][0]['stats'], legacy_old4_parity=mode == 'weighted')
    record('SANITY_PASS_FULL98_STARTING', no_training=True, frozen_checkpoint=str(MODEL))
    jobs = [(gpu, offset, count, 'gross', output / 'shards' / ('gpu' + str(gpu)))
            for gpu, (offset, count) in enumerate(zip((0, 25, 50, 74), (25, 25, 24, 24)))]
    wave('FULL98', jobs, 0)
    merged = output / 'predictions'
    merged.mkdir()
    records, latencies, receipts, configs = [], [], [], []
    for gpu, offset, count, mode, destination in jobs:
        config, receipt = read(destination / 'inference_config.json'), read(destination / 'inference_completion.json')
        assert config['search_value'] == mode and config['model'] == str(MODEL)
        assert config['max_frames'] == 0 and config['write_verification'] == 'action'
        assert not config['disable_search'] and not config['unsafe_writes']
        assert [r['sequence'] for r in receipt['records']] == names[offset:offset + count]
        assert {p.stem for p in destination.glob('*.txt')} == set(names[offset:offset + count])
        for row in receipt['records']:
            for suffix in ('.txt', '_latency.npy', '_recoverability_decisions.npz'):
                os.link(destination / (row['sequence'] + suffix), merged / (row['sequence'] + suffix))
            latency = np.load(destination / (row['sequence'] + '_latency.npy'))
            assert len(latency) == row['frames'] - 1 and np.isfinite(latency).all() and (latency > 0).all()
            latencies.append(latency)
        records.extend(receipt['records'])
        receipts.append(receipt)
        configs.append(config)
    assert [r['sequence'] for r in records] == names and sum(r['frames'] for r in records) == 49418
    for index, row in enumerate(records, 1):
        row.update(sequence_index=index, sequences=98)
    config = dict(configs[0], output=str(merged), sequence_offset=0, limit_sequences=0,
                  full_shard_union_verified=True, source_shards=[str(j[-1]) for j in jobs])
    (merged / 'inference_config.json').write_text(json.dumps(config, indent=2))
    latency = np.concatenate(latencies)
    receipt = {'completed': True, 'sequences': 98, 'frames': 49418, 'smoke_only': True,
               'records': records, 'gt_scoring_completed': False,
               'fps_including_decode_crop_update': len(latency) / latency.sum(),
               'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
               'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
               'peak_cuda_allocated_mib': max(r['peak_cuda_allocated_mib'] for r in receipts),
               'peak_cuda_reserved_mib': max(r['peak_cuda_reserved_mib'] for r in receipts),
               'wall_seconds_including_initialization_and_diagnostic_serialization': time.monotonic() - started}
    (merged / 'inference_completion.json').write_text(json.dumps(receipt, indent=2))
    record('FULL98_INFERENCE_COMPLETE_CPU_GT_REPORT_STARTING')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(REPO),
               LD_LIBRARY_PATH='/data/gb/envs/gola/lib', OMP_NUM_THREADS='4')
    command = [PYTHON, '-u', '-m', 'research.collect_recoverability_metrics',
               '--dataset', 'lasher', '--root', str(DATA), '--split', str(SPLIT),
               '--labels', 'gross_gain', '--runs', str(merged),
               '--reference-labels', 'baseline', 'c1', 'old4', '--references',
               '/data/gb/outputs/abc_internal_validation_v1/baseline/predictions',
               '/data/gb/outputs/abc_internal_validation_v1/c1/predictions', str(REFERENCE),
               '--output', str(output / 'report')]
    with (output / 'CPU_GT_report.log').open('w') as log:
        subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(output / 'report' / 'full_recoverability_report.json')
    record('COMPLETE_FULL98_ACTUAL_GT', no_training=True, official_metrics_completed=False,
           sequence_mean_iou={key: value['sequence_mean_iou'] for key, value in report['variants'].items()},
           paired=report['paired'], best_weight_changed=False)


if __name__ == '__main__':
    main()
