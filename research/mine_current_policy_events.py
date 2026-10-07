"""Mine TRAIN action queries from closed, actual locked-policy trajectories.

This CPU-only diagnostic reads completed sequences while the other shards may
still run. GT is used only after inference for query selection. It does not
reconstruct internal states or label predicted advantages as final net scores.
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .collect_candidate_metrics import ground_truth
from .collect_core_metrics import failure_events, localization_quality


def representatives(mask, value):
    """One largest-gap frame per consecutive known event run."""
    indices = np.flatnonzero(mask)
    return [int(run[np.argmax(value[run])]) for run in np.split(
        indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(run)]


def mine_sequence(folder, record, data_root):
    name, frames = record['sequence'], record['frames']
    prediction = np.loadtxt(folder / (name + '.txt'), dtype=np.float64, ndmin=2)
    gt = ground_truth(data_root, name, 'lasher')
    assert prediction.shape == gt.shape == (frames, 4), name
    with np.load(folder / (name + '_recoverability_decisions.npz')) as timeline:
        boxes = timeline['boxes_xyxy'].reshape(frames - 1, 35, 4).copy()
        boxes[:, :, 2:] -= boxes[:, :, :2]
        valid = timeline['valid'].reshape(frames - 1, 35)
        choice, keep = timeline['choice'], timeline['original_choice']
        rows = np.arange(frames - 1)
        assert valid[rows, choice].all() and valid[rows, keep].all()
        assert (valid.sum(1) <= 10).all() and ((keep >= 0) & (keep < 5)).all()
        assert np.allclose(boxes[rows, choice], prediction[1:], rtol=0, atol=.000501)
        quality = []
        for candidate in range(35):
            q, known = localization_quality(np.concatenate((prediction[:1], boxes[:, candidate])), gt, 'lasher')
            quality.append(q[1:])
        quality = np.stack(quality, axis=1)
        known = known[1:]
        selected, original = quality[rows, choice], quality[rows, keep]
        best = np.where(valid, quality, -1.).max(1)
        best_local = np.where(valid[:, :5], quality[:, :5], -1.).max(1)
        pause, writes = timeline['pause'], timeline['template_updated']
        raw = timeline['raw_score'].reshape(frames - 1, 35)[rows, choice]
        assert np.array_equal(writes, (raw > .84) & ~pause)
        executed, requested = timeline['extra_executed'], timeline['search_requested']
        assert np.array_equal(executed, valid[:, 5:].any(1)) and (executed <= requested).all()
        changed = choice != keep
        masks = {
            'actual_intervention': known & (changed | pause),
            'intervention_iou_drop_gt_0.1': known & changed & (original - selected > .1),
            'intervention_iou_gain_gt_0.1': known & changed & (selected - original > .1),
            'recoverable_missed_candidate': known & (selected < .2) & (best >= .5),
            'extra_reintroduced_but_rejected': known & (best_local < .5) & (best >= .5) & (selected < .5),
            'wrong_localization_template_write': known & writes & (selected < .2),
        }
        tags = {}

        def add(frame, tag):
            if 1 <= frame < frames and known[frame - 1]:
                tags.setdefault(frame, set()).add(tag)

        for index in np.flatnonzero(masks['actual_intervention']):
            add(int(index) + 1, 'actual_intervention')
        for tag, mask in masks.items():
            if tag != 'actual_intervention':
                for index in representatives(mask, np.abs(best - selected) + np.abs(original - selected)):
                    add(index + 1, tag)

        # Exact selected timeline boxes, not the rounded prediction text, drive
        # failure labels. Keep existing three-frame failure/recovery definitions.
        exact = np.concatenate((prediction[:1], boxes[rows, choice]))
        output_quality, output_known = localization_quality(exact, gt, 'lasher')
        failures = failure_events(output_quality, output_known)
        for event in failures:
            start = event['start_frame_zero_based']
            for offset in (-2, -1, 0, 1, 3):
                add(start + offset, 'failure_turn_%+d' % offset)
            if event['recovered']:
                add(event['recovery_start_frame_zero_based'], 'observed_recovery_start')

        # Retain normal non-intervention controls outside +/-3 frames of the
        # mined states; deterministic even spacing, not all adjacent easy frames.
        excluded = np.zeros(frames - 1, dtype=bool)
        for frame in tags:
            excluded[max(0, frame - 4):min(frames - 1, frame + 3)] = True
        normal = np.flatnonzero(known & (selected >= .5) & ~changed & ~pause & ~executed & ~excluded)
        if len(normal):
            for index in np.unique(normal[np.linspace(0, len(normal) - 1, min(2, len(normal)), dtype=int)]):
                add(int(index) + 1, 'normal_control')

        queries = []
        for frame, labels in sorted(tags.items()):
            index = frame - 1
            # Existing H3 collector requires query through query+3 valid. Keep
            # other events in the inventory, without pretending they are jobs.
            h3_valid = frame + 3 < frames and bool(output_known[frame:frame + 4].all())
            queries.append(dict(sequence=name, query_frame=frame, tags=sorted(labels),
                current_iou=float(selected[index]), c1_keep_iou=float(original[index]),
                best_local_iou=float(best_local[index]), best_observed_iou=float(best[index]),
                actual_choice=int(choice[index]), c1_keep_choice=int(keep[index]),
                extra_executed=bool(executed[index]), query_pause=bool(pause[index]),
                template_updated=bool(writes[index]), selected_raw_score=float(raw[index]),
                H3_collector_eligible=h3_valid))
        counts = {tag: int(mask.sum()) for tag, mask in masks.items()}
        counts.update(known_tracking_frames=int(known.sum()), failed_tracking_frames=int((known & (selected < .2)).sum()),
            actual_extra_forwards=int(executed.sum()), failure_events=len(failures),
            recovered_events=sum(event['recovered'] for event in failures),
            censored_failure_events=sum(not event['recovered'] for event in failures))
    return queries, counts, failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sequences-per-shard', type=int, default=0, help='Sanity subset; 0 reads every currently closed sequence')
    args = p.parse_args()
    assert args.sequences_per_shard >= 0
    root = Path(args.trace_root)
    plan = json.loads((root / 'plan.json').read_text())
    split = json.loads(Path(plan['split']).read_text())
    assert len(split['train']) == 881 and len(split['validation']) == 98 and not set(split['train']) & set(split['validation'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    inventory, queries, events, totals, per_sequence = [], [], [], Counter(), []
    for gpu in range(4):
        folder = root / 'full' / ('gpu%d' % gpu) / 'predictions'
        config = json.loads((folder / 'inference_config.json').read_text())
        assert config['model'] == plan['model'] and config['search_value'] == 'gross' and config['write_verification'] == 'action'
        assert config['ABC_model_family'] == 'ABC_candidate_relations' and config['head_epoch'] == 5
        assert config['policy'] == 'learned' and not config['zero_init'] and not config['disable_search'] and not config['unsafe_writes']
        assert all(config[key] is None for key in ('commit_model', 'state_commit_model', 'identity_projection', 'reference_mode'))
        assert config['identity_weight'] == 0. and config['max_frames'] == 0 and config['train_split'] == plan['split']
        # The evaluator flushes a newline only after all sequence artifacts close.
        lines = (folder / 'progress.jsonl').read_bytes().split(b'\n')[:-1]
        records = [json.loads(line) for line in lines]
        if args.sequences_per_shard:
            records = records[:args.sequences_per_shard]
        allowed = {row['sequence']: row['frames'] for row in plan['groups'][gpu]}
        for record in records:
            name = record['sequence']
            assert name in split['train'] and name not in split['validation'] and allowed[name] == record['frames']
            selected, counts, failures = mine_sequence(folder, record, plan['root'])
            inventory.append(dict(gpu=gpu, sequence=name, frames=record['frames'], closed_record=record))
            queries.extend(selected)
            events.extend(dict(sequence=name, **event) for event in failures)
            totals.update(counts)
            per_sequence.append(dict(sequence=name, gpu=gpu, frames=record['frames'], query_representatives=len(selected), **counts))
    assert inventory and len({row['sequence'] for row in inventory}) == len(inventory)
    tag_counts = Counter(tag for query in queries for tag in query['tags'])
    jobs = [dict(sequence=row['sequence'], query_frame=row['query_frame']) for row in queries if row['H3_collector_eligible']]
    required_max_prefix = max([8] + [row['query_frame'] for row in jobs])
    report = dict(created_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        trace_root=str(root), policy=dict(model=plan['model'], family='ABC_candidate_relations', epoch=5, search_value='gross', write_verification='action'),
        scope='CPU TRAIN event mining; completed sequences only; no optimizer, new weights or native TEST accuracy',
        closed_sequences=len(inventory), closed_frames=sum(row['frames'] for row in inventory),
        planned_sequences=881, planned_frames=plan['frames'], full_TRAIN_inventory=len(inventory) == 881,
        counts=dict(totals), query_representatives=len(queries), tags=dict(tag_counts), H3_jobs=len(jobs),
        ineligible_H3_queries=len(queries) - len(jobs),
        required_collector_max_prefix=required_max_prefix,
        H3_collection_contract='Pass --max-prefix required_collector_max_prefix; the default1024 can reject full-video late queries',
        labels='Actual dataset GT only after predictions; IoU localization proxies, not full distractor semantic identities',
        near_threshold_events='Not mined: stored predicted_advantage omits final harm/write net score; cannot claim a net-margin threshold',
        states='Queries are timestamps for causal prefix replay, never states reconstructed from rounded text predictions',
        event_sampling='All GT-known actual interventions; largest gap per contiguous missed/write episode; failure offsets -2,-1,0,+1,+3; observed recovery; up to2 normal controls persequence',
        censored_failures='Not recovered within recorded sequence, not proven permanently unrecoverable')
    for filename, value in (('report.json', report), ('closed_inventory.json', inventory), ('queries.json', dict(queries=queries)),
                            ('jobs.json', dict(jobs=jobs, required_collector_max_prefix=required_max_prefix)), ('failure_events.json', events), ('per_sequence.json', per_sequence)):
        (output / filename).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
