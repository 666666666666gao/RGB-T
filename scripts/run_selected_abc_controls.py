"""Diagnose the fixed trained ABC with full developer videos; no training or TEST."""
import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.run_candidate_relation_training import environment
from scripts.run_search_gain_native import partitions

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/data/gb/envs/gola/bin/python'
ORIGINAL = Path('/data/gb/outputs/candidate_relation_raw_ABC_20261006_v2')
RECOVERY = Path('/data/gb/outputs/candidate_relation_native_recovery_20261006')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    review = read(args.review)
    assert review['status'] == 'PASS' and review['scope'] == 'SELECTED_NATIVE_DIAGNOSTIC_CONTROLS_SOURCE'
    native = read(RECOVERY / 'native/complete_metrics.json')
    assert native['completed'] and read(RECOVERY / 'reconstruction_verification.json')['passed']
    selected = read(RECOVERY / 'selected_model.json')
    model = native['same_ABC_model_both_datasets']
    assert model == selected['parent_model'] and selected['selected_epoch'] == 5
    source = read(ORIGINAL / 'full98/relations_lr5_best/predictions/inference_config.json')
    names = sorted(read(source['validation_split'])['validation'])
    data = Path(source['root'])
    lengths = [sum(p.is_file() for p in (data / name / 'visible').iterdir()) for name in names]
    assert len(names) == 98 and sum(lengths) == 49418
    groups = partitions(lengths)
    root = Path(args.output)
    assert not root.exists() and shutil.disk_usage('/data/gb').free > 20 * 1024**3
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    values = [tuple(int(x.strip()) for x in line.split(',')) for line in cards.splitlines()]
    assert [v[0] for v in values] == [0, 1, 2, 3] and all(v[1] < 500 and v[2] == 0 for v in values)
    root.mkdir(parents=True)
    controls = {'no_extra_search': '--disable-search', 'pause_off': '--unsafe-writes'}
    plan = {'model': model, 'head_epoch': 5, 'groups': groups, 'sequences': names,
            'frames': 49418, 'controls': controls, 'same_locked_model': True,
            'scope': 'Reused TRAIN-root developer98 diagnostics, not independent confirmation, eight-combination ablation or native accuracy',
            'pause_off_scope': 'Only disables query pause; identity-memory verification and candidate selection remain active',
            'search_off_scope': 'Removes extra visual evidence and its contextual relation input; not full B-off or matched visual cost'}
    (root / 'plan.json').write_text(json.dumps(plan, indent=2))

    def record(stage, **fields):
        value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(), **fields}
        (root / 'progress.json').write_text(json.dumps(value, indent=2))
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    for stage in ('sanity', 'full'):
        for control, flag in controls.items():
            children = []
            for gpu, group in enumerate(groups):
                output = root / control / stage / f'gpu{gpu}' / 'predictions'
                command = [PYTHON, '-u', '-m', 'research.evaluate_recoverability', '--dataset', 'lasher',
                           '--root', str(data), '--model', model, '--pretrained', source['pretrained'],
                           '--c1-head', source['c1_head'], '--motion-run', source['motion_run'],
                           '--validation-split', source['validation_split'], '--search-value', 'gross',
                           '--write-verification', 'action', '--seed', '42', flag,
                           '--sequence-offset', str(group['offset']), '--limit-sequences', '1' if stage == 'sanity' else str(group['count']),
                           '--max-frames', '64' if stage == 'sanity' else '0', '--output', str(output)]
                log = (root / f'{control}_{stage}_gpu{gpu}.log').open('w')
                child = subprocess.Popen(command, cwd=REPO, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
                children.append((child, log, gpu, group, output))
            record(control.upper() + '_' + stage.upper(), children=[{'gpu': g, 'pid': c.pid} for c, _, g, _, _ in children])
            for child, log, gpu, group, output in children:
                code = child.wait(); log.close()
                assert code == 0, (control, stage, gpu, 'Read original log; no retry')
                receipt, config = read(output / 'inference_completion.json'), read(output / 'inference_config.json')
                assert receipt['completed'] and config['model'] == model and config['head_epoch'] == 5
                assert config['disable_search'] == (control == 'no_extra_search') and config['unsafe_writes'] == (control == 'pause_off')
                assert not config['zero_init'] and not config['parity_check']
                assert receipt['sequences'] == (1 if stage == 'sanity' else group['count'])
                assert receipt['frames'] == (min(64, lengths[group['offset']]) if stage == 'sanity' else group['frames'])
            record(control.upper() + '_' + stage.upper() + '_PASS')
    for control in controls:
        merged = root / control / 'predictions'; merged.mkdir()
        records, completions, latencies = [], [], []
        for gpu, group in enumerate(groups):
            shard = root / control / 'full' / f'gpu{gpu}' / 'predictions'
            done = read(shard / 'inference_completion.json')
            records.extend(done['records']); completions.append(done)
            subset = names[group['offset']:group['offset'] + group['count']]
            assert {p.stem for p in shard.glob('*.txt')} == set(subset)
            for name in subset:
                latencies.append(np.load(shard / (name + '_latency.npy')))
                for suffix in ('.txt', '_latency.npy', '_recoverability_decisions.npz'):
                    os.link(shard / (name + suffix), merged / (name + suffix))
        assert len(list(merged.glob('*.txt'))) == 98
        assert [r['sequence'] for r in records] == names and sum(r['frames'] for r in records) == 49418
        config = read(root / control / 'full/gpu0/predictions/inference_config.json')
        config.update(output=str(merged), sequence_offset=0, limit_sequences=0, full_shard_union_verified=True)
        (merged / 'inference_config.json').write_text(json.dumps(config, indent=2))
        latency = np.concatenate(latencies)
        receipt = {'completed': True, 'sequences': 98, 'frames': 49418, 'records': records, 'smoke_only': True,
                   'full_shard_union_verified': True, 'gt_scoring_completed': False,
                   'fps_including_decode_crop_update': len(latency) / float(latency.sum()),
                   'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
                   'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
                   'peak_cuda_allocated_mib': max(c['peak_cuda_allocated_mib'] for c in completions),
                   'peak_cuda_reserved_mib': max(c['peak_cuda_reserved_mib'] for c in completions)}
        (merged / 'inference_completion.json').write_text(json.dumps(receipt, indent=2))
    original_report = read(ORIGINAL / 'full98_report/full_recoverability_report.json')['args']
    references = original_report['references'] + [str(ORIGINAL / 'full98/relations_lr5_best/predictions')]
    reference_labels = original_report['reference_labels'] + ['selected_ABC']
    with (root / 'CPU_report.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'research.collect_recoverability_metrics', '--dataset', 'lasher',
                        '--root', str(data), '--split', source['validation_split'], '--labels', *controls,
                        '--runs', *[str(root / c / 'predictions') for c in controls],
                        '--reference-labels', *reference_labels, '--references', *references,
                        '--output', str(root / 'report')], cwd=REPO, env=environment(''), stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(root / 'report/full_recoverability_report.json')
    assert report['completed'] and report['sequences'] == 98
    assert all(report['variants'][c]['frames'] == 49418 for c in controls)
    record('COMPLETE_SELECTED_ABC_FULL98_DIAGNOSTIC_CONTROLS', trained_model=model,
           new_training=False, native_TEST_inference=False, deleted_weights=False)


if __name__ == '__main__':
    main()
