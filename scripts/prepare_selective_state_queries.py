"""CPU eligibility: all TRAIN videos plus one query per historical event; no TEST."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hints', required=True)
    p.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json')
    p.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    p.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    split = json.loads(Path(args.split).read_text())
    assert len(split['train']) == 881 and len(split['validation']) == 98
    assert not set(split['train']) & set(split['validation'])
    hints = json.loads(Path(args.hints).read_text())
    assert len(hints['transitions']) == 716 and len(hints['write_episodes']) == 98
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    lookup = {dataset[i].get_name(): i for i in range(len(dataset))}
    rng, jobs, excluded, eligible = np.random.default_rng(42), {}, [], {}
    for partition in ('train', 'validation'):
        for name in sorted(split[partition]):
            sequence = dataset[lookup[name]]
            gt = np.stack([sequence[t].get_bounding_box() for t in range(len(sequence))])
            known = np.isfinite(gt).all(1) & (gt[:, 2:] > gt[:, :2]).all(1)
            prefix = np.r_[0, np.cumsum(known)]
            # Two actual videos have only 27/32 frames: preserve them for the
            # explicitly shorter H3 arm; they cannot supply full H32 labels.
            h = 3 if len(sequence) <= 33 else 32
            values = [q for q in range(1, len(sequence) - h) if known[q] and prefix[q + h + 1] - prefix[q + 1] >= 3]
            assert values, ('No eligible query in sequence', name)
            eligible[name] = set(values)
            q = int(rng.choice(values))
            jobs[name, q] = {'sequence': name, 'query_frame': q, 'partition': partition,
                            'sources': ['uniform_video_coverage'], 'historical_event_ids': [], 'total_frames': len(sequence), 'horizon': h}
    for kind, source in [('transition', hints['transitions']), ('write_episode', hints['write_episodes'])]:
        for item in source:
            name, q = item['sequence'], item['query_frame']
            assert name in split['train']
            if q not in eligible[name]:
                excluded.append({'source': kind, **item, 'reason': 'H32/current-GT eligibility'})
                continue
            key = name, q
            if key not in jobs:
                jobs[key] = {'sequence': name, 'query_frame': q, 'partition': 'train', 'sources': [],
                             'historical_event_ids': [], 'total_frames': len(dataset[lookup[name]]),
                             'horizon': 3 if len(dataset[lookup[name]]) <= 33 else 32}
            jobs[key]['sources'].append(kind)
            jobs[key]['historical_event_ids'].append(item.get('event_id', f'{name}:historical_transition:{q}'))
    grouped = {}
    for row in jobs.values():
        grouped.setdefault(row['sequence'], []).append(row)
    shards, costs = [[] for _ in range(4)], [0] * 4
    for name, group in sorted(grouped.items(), key=lambda pair: -(max(j['query_frame'] for j in pair[1]) + 192 * len(pair[1]))):
        gpu = int(np.argmin(costs))
        shards[gpu] += sorted(group, key=lambda j: j['query_frame'])
        costs[gpu] += max(j['query_frame'] for j in group) + 192 * len(group)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for gpu, shard in enumerate(shards):
        (out / f'gpu{gpu}_jobs.json').write_text(json.dumps({'jobs': shard}, indent=2))
    report = {'status': 'CPU_H32_QUERIES_PREPARED', 'jobs': list(jobs.values()), 'excluded': excluded,
              'queries': len(jobs), 'train_queries': sum(j['partition'] == 'train' for j in jobs.values()),
              'validation_queries': sum(j['partition'] == 'validation' for j in jobs.values()),
              'train_sequences': 881, 'validation_sequences': 98, 'shard_queries': list(map(len, shards)),
              'estimated_visual_forwards_by_shard': costs,
              'short_H3_only_sequences': sorted({j['sequence'] for j in jobs.values() if j['horizon'] == 3}),
              'event_unit': 'Historical event timestamps regenerated with actual parent; duplicate timestamp merged. Training weights aggregate each actual consecutive known-state event.',
              'GT_role': 'TRAIN eligibility only; no native TEST reads', 'NN_forwards': 0}
    (out / 'plan.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('status', 'queries', 'train_queries', 'validation_queries', 'shard_queries', 'estimated_visual_forwards_by_shard')}))


if __name__ == '__main__':
    main()
