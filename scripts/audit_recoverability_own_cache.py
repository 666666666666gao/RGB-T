"""Read-only audit of completed own-policy prefixes and the matched-query plan."""
import json
from pathlib import Path

import numpy as np

from research.recoverability_modules import DECISION_FIELDS
from research.train_recoverability import LABEL_FIELDS


def main():
    root = Path('/data/gb/outputs')
    old_roots = [root / f'recoverability_train_s{seed}_20261003' for seed in (42, 43, 44)]
    own_roots = [root / f'recoverability_own_policy_{part}_20261003' for part in ('train', 'validation')]
    configs = [json.loads((p / 'config.json').read_text()) for p in old_roots + own_roots]
    split = json.loads(Path(configs[0]['split']).read_text())
    records = []
    for path, config in zip(old_roots + own_roots, configs):
        receipt = json.loads((path / 'completion.json').read_text())
        assert receipt['completed'] and not receipt['decision_input_contains_future']
        for key in ('split', 'root', 'cache', 'head', 'pretrained', 'motion_run', 'max_prefix',
                    'regions', 'future_policy', 'future_horizon'):
            assert config[key] == configs[0][key], key
        jobs = config['jobs']
        assert len(jobs) == config['clips'] == receipt['clips']
        assert {j['sequence'] for j in jobs} <= set(split[config['partition']])
        with np.load(path / 'samples.npz') as archive:
            data = {key: archive[key] for key in DECISION_FIELDS + LABEL_FIELDS}
            assert all(len(value) == len(jobs) and np.isfinite(value).all() for value in data.values())
            rows = np.arange(len(jobs))
            assert data['valid'][rows, 0, data['original_choice'].astype(int)].all()
            assert np.array_equal(data['action_valid'][..., 0], data['valid'])
            assert np.array_equal(data['action_valid'][..., 1], data['valid'] & (data['raw_score'] > .84))
            assert data['history_valid'][:, -1].all()
            for row, job in enumerate(jobs):
                valid = data['history_valid'][row]
                assert int(valid.sum()) == job['query_frame']
                assert np.array_equal(data['history_frames'][row, valid], np.arange(job['query_frame']))
            result = {'root': str(path), 'clips': len(jobs), 'partition': config['partition'],
                      'valid_actions': int(data['action_valid'].sum()),
                      'after256': sum(j['query_frame'] > 256 for j in jobs),
                      'max_query': max(j['query_frame'] for j in jobs),
                      'decision_fields': list(DECISION_FIELDS), 'future_fields_excluded': True}
            if 'prefix_counts' in archive:
                counts = archive['prefix_counts']
                assert counts.shape == (len(jobs), 4)
                assert counts.sum(0).tolist() == receipt['actual_prefix_counts_sum']
                result['actual_prefix_counts_sum'] = counts.sum(0).tolist()
            records.append(result)
    old = {(j['sequence'], j['query_frame']) for c in configs[:3] for j in c['jobs']}
    own = {(j['sequence'], j['query_frame']) for j in configs[3]['jobs']}
    val = {(j['sequence'], j['query_frame']) for j in configs[4]['jobs']}
    assert not {q[0] for q in old | own} & {q[0] for q in val}
    output = {'completed': True, 'neural_forwards': 0, 'optimization_updates': 0,
              'records': records, 'original_queries': len(old), 'own_queries': len(own),
              'paired_existing_queries': len(old & own), 'new_query_times': len(own - old),
              'planned_both_arm_train_queries': len(old | own),
              'planned_epochs': 30, 'planned_batch_size': 128,
              'planned_updates_each': 30 * ((len(old | own) + 127) // 128),
              'future_continuation': 'frozen C1; only prefix policy changes',
              'validation_same_queries': len(val)}
    path = root / 'recoverability_own_policy_audit_20261003'
    path.mkdir(exist_ok=True)
    (path / 'cache_audit.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in output.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
