"""Collect deployed relation-policy states for the next TRAIN data-aggregation round."""
import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.run_candidate_relation_training import environment

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/data/gb/envs/gola/bin/python'
ORIGINAL = Path('/data/gb/outputs/candidate_relation_raw_ABC_20261006_v2')
MODEL = '/data/gb/outputs/candidate_relation_native_recovery_20261006/reconstructed_epoch5/best.pth'
REFERENCE = ORIGINAL / 'full98/relations_lr5_best/predictions'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    review = read(args.review)
    assert review['status'] == 'PASS' and review['scope'] == 'CURRENT_RELATION_POLICY_COLLECTION_SOURCE'
    source = read(ORIGINAL / 'fit_full/relations_lr5/config.json')
    assert len(source['train']) == 9 and source['train_clips'] == 2576
    assert Path(MODEL).is_file()
    disk_free = shutil.disk_usage('/data/gb').free
    assert disk_free > 20 * 1024**3
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True)
    values = [tuple(int(x.strip()) for x in line.split(',')) for line in cards.splitlines()]
    assert [v[0] for v in values] == [0, 1, 2, 3] and all(v[1] < 500 and v[2] == 0 for v in values)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    split = read(source['source_configs'][0]['split'])
    train_rows = {}
    for folder in source['train']:
        config = read(Path(folder) / 'config.json')
        assert config['partition'] == 'train'
        with np.load(Path(folder) / 'samples.npz') as arrays:
            keep = arrays['original_choice'].astype(int)
            iou = arrays['current_iou'][np.arange(len(keep)), 0, keep]
            for job, quality in zip(config['jobs'], iou):
                key = (job['sequence'], job['query_frame'])
                if key not in train_rows:
                    train_rows[key] = dict(job, prior_keep_iou=float(quality))
    assert len(train_rows) == 2576 and {name for name, _ in train_rows} == set(split['train'])
    rng = np.random.default_rng(42)
    strata = [('failed', 128, lambda q: q < .2),
              ('ambiguous', 64, lambda q: .2 <= q < .5),
              ('normal', 64, lambda q: q >= .5)]
    selected, stratum_counts = [], {}
    for label, count, belongs in strata:
        eligible = [dict(job, prior_state_category=label) for job in train_rows.values()
                    if belongs(job['prior_keep_iou'])]
        assert len(eligible) >= count, (label, len(eligible), count)
        rng.shuffle(eligible)
        selected.extend(eligible[:count])
        stratum_counts[label] = {'eligible': len(eligible), 'selected': count}
    assert len({(j['sequence'], j['query_frame']) for j in selected}) == 256
    validation = read(Path(source['validation']) / 'config.json')['jobs']
    assert len(validation) == 196 and {j['sequence'] for j in validation} == set(split['validation'])
    assert len({(j['sequence'], j['query_frame']) for j in validation}) == 196
    groups = {'train': [selected[gpu::4] for gpu in range(4)],
              'validation': [validation[gpu::4] for gpu in range(4)]}
    plan = {'model': MODEL, 'head_epoch': 5, 'model_family': 'ABC_candidate_relations',
            'prefix_search_value': 'gross', 'prefix_write_verification': 'action', 'future_policy': 'own',
            'original_train_roots': source['train'], 'original_validation': source['validation'],
            'strata': stratum_counts, 'train_queries': 256, 'validation_queries': 196,
            'groups': groups, 'batch_clips': 16, 'forward_batch': 64, 'max_prefix': 1024,
            'disk_free_before_bytes': disk_free, 'reserved_output_budget_bytes': 20 * 1024**3,
            'scope': 'TRAIN supervision selection by prior TRAIN IoU; reused developer validation; no TEST input, training or accuracy claim',
            'next_training': 'Retain old policy states and add this deployed-current policy; do not silently deduplicate distinct policies'}
    (root / 'plan.json').write_text(json.dumps(plan, indent=2))
    started = time.perf_counter()

    def record(stage, **fields):
        value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 'elapsed_seconds': time.perf_counter() - started, **fields}
        (root / 'progress.json').write_text(json.dumps(value, indent=2))
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    for stage in ('sanity', 'full'):
        for partition in ('train', 'validation'):
            children = []
            for gpu, group in enumerate(groups[partition]):
                jobs = group[:1] if stage == 'sanity' else group
                jobs_path = root / f'{partition}_{stage}_jobs_gpu{gpu}.json'
                jobs_path.write_text(json.dumps({'jobs': jobs}, indent=2))
                output = root / partition / stage / f'gpu{gpu}'
                command = [PYTHON, '-u', '-m', 'research.collect_recoverability',
                           '--root', source['source_configs'][0]['root'],
                           '--cache', source['source_configs'][0]['cache'],
                           '--split', source['source_configs'][0]['split'], '--partition', partition,
                           '--pretrained', source['source_configs'][0]['pretrained'], '--head', source['c1_head'],
                           '--motion-run', source['source_configs'][0]['motion_run'], '--prefix-model', MODEL,
                           '--future-policy', 'own', '--prefix-search-value', 'gross',
                           '--prefix-write-verification', 'action', '--jobs-file', str(jobs_path),
                           '--batch-clips', '1' if stage == 'sanity' else '16', '--forward-batch', '64',
                           '--max-prefix', '1024', '--seed', '42', '--output', str(output)]
                log = (root / f'{partition}_{stage}_gpu{gpu}.log').open('w')
                child = subprocess.Popen(command, cwd=REPO, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
                children.append((child, log, gpu, output, jobs))
            record(partition.upper() + '_' + stage.upper(),
                   children=[{'gpu': gpu, 'pid': child.pid} for child, _, gpu, _, _ in children])
            for child, log, gpu, output, jobs in children:
                code = child.wait(); log.close()
                assert code == 0, (partition, stage, gpu, 'Read original log; never retry unchanged')
                done, config = read(output / 'completion.json'), read(output / 'config.json')
                assert done['completed'] and done['clips'] == len(jobs) and not done['decision_input_contains_future']
                assert config['prefix_model'] == MODEL and config['prefix_checkpoint_epoch'] == 5
                assert config['prefix_model_family'] == 'ABC_candidate_relations'
                assert config['prefix_search_value'] == 'gross' and config['prefix_write_verification'] == 'action'
                assert config['future_policy_mode'] == 'own' and config['future_checkpoint_epoch'] == 5
                assert [(j['sequence'], j['query_frame']) for j in config['jobs']] == [(j['sequence'], j['query_frame']) for j in jobs]
                if stage == 'sanity' and partition == 'validation':
                    job = jobs[0]
                    native_root = Path(read(REFERENCE / 'inference_config.json')['root'])
                    initial = np.fromstring((native_root / job['sequence'] / 'init.txt').read_text().splitlines()[0], sep=',')
                    initial[2:] += initial[:2]
                    with np.load(REFERENCE / (job['sequence'] + '_recoverability_decisions.npz')) as reference:
                        choice = reference['choice'].astype(int)
                        boxes = reference['boxes_xyxy'][np.arange(len(choice)), choice // 5, choice % 5]
                    prediction = np.concatenate((initial[None], boxes), axis=0).astype(np.float32)
                    with np.load(output / 'samples.npz') as arrays:
                        valid = arrays['history_valid'][0]
                        frames = arrays['history_frames'][0, valid]
                        actual = arrays['history_boxes'][0, valid]
                    assert np.array_equal(frames, np.arange(job['query_frame']))
                    assert np.array_equal(actual, prediction[frames]), (gpu, job, 'Deployed gross-prefix parity failed')
            record(partition.upper() + '_' + stage.upper() + '_PASS', clips=sum(len(j) for _, _, _, _, j in children))
    record('COMPLETE_CURRENT_RELATION_POLICY_COLLECTION', train_queries=256, validation_queries=196,
           new_training=False, new_TEST_inference=False, weights_deleted=False)


if __name__ == '__main__':
    main()
