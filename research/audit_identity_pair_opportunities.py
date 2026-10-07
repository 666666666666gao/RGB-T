"""CPU-only audit of existing paired GT supervision and frozen C1 score gaps.

No image loading, model inference, optimization or native-test access. The
cosine margin bound applies to the paired C1 proxy, not deployed ABC scores.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def quantiles(values):
    return dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'),
                    np.percentile(values, (0, 25, 50, 75, 95, 100)).tolist())) if len(values) else None


def audit(roots, partition):
    fields = ('sequence_index', 'sample_index', 'contaminated', 'valid', 'quality', 'c1_score')
    rows, names = [], {}
    for root in roots:
        folder = root / partition
        config = json.loads((folder / 'config.json').read_text())
        done = json.loads((folder / 'completion.json').read_text())
        assert done['completed'] and config['partition'] == partition
        with np.load(folder / 'samples.npz') as source:
            row = {field: source[field].copy() for field in fields}
        assert len(row['valid']) == done['completed_queries'] == config['expected_samples']
        assert all(np.isfinite(value).all() for value in row.values())
        rows.append(row)
        names.update({int(key): value for key, value in config['sequence_names'].items()})
    data = {field: np.concatenate([row[field] for row in rows]) for field in fields}
    n = len(data['valid'])
    assert len(np.unique(data['sample_index'])) == n
    valid, quality, score = data['valid'], data['quality'], data['c1_score']
    assert valid.shape == quality.shape == score.shape == (n, 5) and valid.any(1).all()
    assert ((quality >= 0) & (quality <= 1)).all()
    keep = np.where(valid, score, -np.inf).argmax(1)
    keep_q = quality[np.arange(n), keep]
    best_q = np.where(valid, quality, -1).max(1)
    positive, negative = valid & (quality >= .5), valid & (quality < .2)
    pair_count = positive.sum(1) * negative.sum(1)
    has_pair = pair_count > 0
    recalled = positive.any(1)
    failed = keep_q < .2
    can_improve = best_q > keep_q + .05
    missed_positive = (keep_q < .5) & recalled
    states = {'keep_correct': keep_q >= .5,
              'keep_ambiguous': (keep_q >= .2) & (keep_q < .5),
              'keep_failed_positive_present': failed & recalled,
              'keep_failed_positive_missing': failed & ~recalled}
    assert sum(mask.sum() for mask in states.values()) == n
    keep_score = score[np.arange(n), keep]
    easiest_positive = np.where(positive, score, -np.inf).max(1)
    gap = keep_score - easiest_positive
    # Mean of two normalized cosine supports is bounded [-1,1]. A frozen
    # C1-score deficit greater than2*alpha cannot be crossed by this residual.
    summary = dict(queries=n, sequences=len(np.unique(data['sequence_index'])),
        c1_query_mean_iou=float(keep_q.mean()), oracle_query_mean_iou=float(best_q.mean()),
        current_query_oracle_gap_pp=float((best_q - keep_q).mean() * 100),
        has_positive_negative_pair_queries=int(has_pair.sum()), positive_negative_pairs=int(pair_count.sum()),
        frozen_c1_failed_queries=int(failed.sum()), failed_with_correct_candidate=int((failed & recalled).sum()),
        missed_positive_queries=int(missed_positive.sum()),
        missed_positive_without_any_training_pair=int((missed_positive & ~has_pair).sum()),
        possible_localization_gain_above_005_queries=int(can_improve.sum()),
        possible_localization_gain_without_any_training_pair=int((can_improve & ~has_pair).sum()),
        template_perturbed_queries=int(data['contaminated'].sum()),
        easiest_correct_candidate_c1_score_deficit=quantiles(gap[missed_positive]),
        identity_only_proxy_bound={str(alpha): dict(max_cosine_score_correction=2 * alpha,
            missed_positive_deficit_exceeds_bound=int((missed_positive & (gap > 2 * alpha)).sum()),
            failed_recallable_deficit_exceeds_bound=int((failed & recalled & (gap > 2 * alpha)).sum()))
            for alpha in (.05, .1, .2)},
        strata={name: dict(queries=int(mask.sum()), pair_supervised_queries=int((mask & has_pair).sum()),
            training_pairs=int(pair_count[mask].sum()), positive_present=int((mask & recalled).sum()),
            oracle_quality_above_keep_005=int((mask & can_improve).sum())) for name, mask in states.items()})
    records = []
    for i in range(n):
        records.append(dict(partition=partition, sequence=names[int(data['sequence_index'][i])],
            sample_index=int(data['sample_index'][i]), c1_candidate=int(keep[i]),
            c1_iou=float(keep_q[i]), oracle_best_iou=float(best_q[i]),
            positive_candidates=int(positive[i].sum()), negative_candidates=int(negative[i].sum()),
            training_pairs=int(pair_count[i]), missed_positive=bool(missed_positive[i]),
            easiest_positive_c1_score_deficit=float(gap[i]) if recalled[i] else None,
            template_perturbed=bool(data['contaminated'][i])))
    return summary, records, set(data['sequence_index'].tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs=4, required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    started = time.perf_counter()
    result, records, partitions = {}, [], []
    for partition, count, sequences in (('train', 14096, 881), ('validation', 1568, 98)):
        summary, current, names = audit(list(map(Path, args.inputs)), partition)
        assert summary['queries'] == count and summary['sequences'] == sequences
        result[partition] = summary
        records.extend(current)
        partitions.append(names)
    assert not partitions[0] & partitions[1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=True, partitions=result, optimizer_updates=0, model_inference_calls=0,
        image_reads=0, CUDA_operations=0, formal_TEST_access=False,
        scope='Existing supervised paired crops and C1 proxy; not own-policy states or deployed ABC utility',
        thresholds=dict(positive_iou=.5, negative_iou=.2, minimum_localization_gain=.05),
        missing_pairs_definition='No valid positive IoU>=.5 versus negative IoU<.2 pair; unknown not made negative',
        bounds_definition='Only a mathematical ceiling for frozen C1+alpha*cosine proxy; not a learned margin or native recovery rate',
        elapsed_seconds=time.perf_counter() - started)
    (output / 'opportunity_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (output / 'per_query_opportunities.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
