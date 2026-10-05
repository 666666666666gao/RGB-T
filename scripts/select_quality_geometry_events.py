"""One query per known-GT normal/failure episode in saved TRAIN-root developer runs."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.collect_candidate_metrics import ground_truth
from research.collect_core_metrics import localization_quality


def runs(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return zip(starts, ends)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    root = Path(args.predictions)
    config = json.loads((root / 'inference_config.json').read_text())
    done = json.loads((root / 'inference_completion.json').read_text())
    split = json.loads(Path(args.split).read_text())
    assert config['root'] == '/data/wangwj/dataset/LasHeR/traingset'
    assert config['dataset'] == 'lasher' and config['search_value'] == 'gross'
    assert config['validation_split'] == args.split and done['completed'] and done['sequences'] == 98
    assert not set(split['train']) & set(split['validation'])
    groups = {name: [] for name in ('normal', 'recovered_H32', 'unrecovered_H32')}
    for record in done['records']:
        name = record['sequence']
        assert name in split['validation']
        prediction = np.loadtxt(root / (name + '.txt'), ndmin=2)
        with np.load(root / (name + '_recoverability_decisions.npz')) as timeline:
            valid = timeline['valid'].reshape(len(prediction) - 1, 35)
            actual = timeline['choice'].astype(int)
            quality = timeline['current_quality'].reshape(len(valid), 35)
            raw_score = timeline['raw_score'].reshape(len(valid), 35)
        rows = np.arange(len(valid))
        score = quality - .01 * (np.arange(35) >= 5)[None]
        available = valid.copy()
        available[:, :5] = False
        available[rows, actual] = True
        best = np.where(available, score, -np.inf).argmax(1)
        chosen = np.where(score[rows, best] > score[rows, actual] + .03, best, actual)
        changed = chosen != actual
        # Choices above use predictions only; GT below selects diagnostic events.
        overlap, known = localization_quality(prediction, ground_truth(config['root'], name, 'lasher'), 'lasher')
        for family, mask in [('normal', known[1:] & (overlap[1:] >= .5)),
                             ('failure', known[1:] & (overlap[1:] < .2))]:
            for start, end in runs(mask):
                eligible = [row for row in range(start, end) if changed[row] and row + 33 < len(prediction)
                            and known[row + 2:row + 34].sum() >= 3]
                if not eligible:
                    continue
                row = eligible[0]
                query = row + 1
                recovered = any((known[t:t + 3] & (overlap[t:t + 3] >= .5)).all()
                                for t in range(query + 1, query + 31))
                kind = 'normal' if family == 'normal' else 'recovered_H32' if recovered else 'unrecovered_H32'
                groups[kind].append({'sequence': name, 'event_id': f'{family}_{start + 1}_{end}',
                                     'event_kind': kind, 'query_frame': query,
                                     'saved_baseline_index': int(actual[row]), 'saved_quality_index': int(chosen[row]),
                                     'total_frames': len(prediction),
                                     'legal_write_pair': bool(raw_score[row, chosen[row]] > .84),
                                     'known_future_labels_H32': int(known[query + 1:query + 33].sum())})
    jobs = []
    for kind, events in groups.items():
        unique = {}
        for event in sorted(events, key=lambda e: (not e['legal_write_pair'], e['query_frame'], e['sequence'])):
            if event['sequence'] not in unique:
                unique[event['sequence']] = event
        assert len(unique) >= 4, (kind, len(unique))
        jobs += list(unique.values())[:4]
    assert len({(j['sequence'], j['event_id']) for j in jobs}) == 12
    output = Path(args.output)
    assert not output.exists()
    output.write_text(json.dumps({'status': 'CPU_EVENT_SELECTION_COMPLETE', 'jobs': jobs,
        'available_event_counts': {k: len(v) for k, v in groups.items()},
        'available_legal_write_pairs': {k: sum(e['legal_write_pair'] for e in v) for k, v in groups.items()},
        'selected_counts': {k: sum(j['event_kind'] == k for j in jobs) for k in groups},
        'selected_legal_write_pairs': sum(j['legal_write_pair'] for j in jobs),
        'scope': 'Previously used TRAIN-root validation videos. Event labels use saved baseline consequences, not hypothetical Q outcomes.',
        'event_unit': 'One query per maximal consecutive known normal/failure run; unrecovered means no three correct future frames within H32, not permanent disappearance.',
        'execution': {'NN_forwards': 0, 'optimizer_steps': 0, 'GPU_queries': 0, 'native_test_reads': False}}, indent=2))
    print(json.dumps({'status': 'CPU_EVENT_SELECTION_COMPLETE', 'queries': len(jobs),
                      'available_events': {k: len(v) for k, v in groups.items()},
                      'selected_legal_write_pairs': sum(j['legal_write_pair'] for j in jobs)}))


if __name__ == '__main__':
    main()
