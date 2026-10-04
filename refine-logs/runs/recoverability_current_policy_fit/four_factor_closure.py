"""Combine actually audited four-cell internal results; never claim native completion."""
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
SETUP = Path('/data/gb/setup')
OUTPUTS = Path('/data/gb/outputs')
STEMS = {'oracle': 'current_policy_fit', 'selector': 'current_policy_selector_fit'}
OLD = 'write_pair_reference_own_b384'
SPLIT = Path('/data/gb/outputs/c1_initial_seed42/split.json')
names = sorted(json.loads(SPLIT.read_text())['validation'])
assert len(names) == len(set(names)) == 98
vectors, rows, accepted = {}, {}, {}
for mode, stem in STEMS.items():
    fit = json.loads((SETUP / f'{stem}_actual_full_fit_cpu_audit_20261004.json').read_text())
    audit = json.loads((SETUP / f'{stem}_actual_full98_cpu_audit_20261004.json').read_text())
    assert fit['status'] == audit['status'] == 'PASS'
    assert fit['both_teacher_arms_passed'] and fit['parent_epoch'] == 4
    assert fit['epochs_per_arm'] == 60 and fit['optimizer_steps_per_arm'] == 180
    assert audit['scope']['frames_per_variant'] == 49418
    assert not audit['scope']['formal_native_test_datasets_read']
    accepted[mode] = {'fit': str(SETUP / f'{stem}_actual_full_fit_cpu_audit_20261004.json'),
                      'full98': str(SETUP / f'{stem}_actual_full98_cpu_audit_20261004.json')}
    for policy in ('c1', 'own'):
        label = f'{stem}_{policy}_b384'
        root = OUTPUTS / f'recoverability_{label}_full_20261004'
        assert (root / 'job_completed.txt').is_file()
        with (root / 'report/per_sequence.csv').open() as stream:
            table = list(csv.DictReader(stream))
        for variant in (label, OLD, 'baseline', 'c1'):
            per_sequence = {item['sequence']: float(item['mean_valid_iou'])
                            for item in table if item['variant'] == variant}
            assert set(per_sequence) == set(names)
            value = np.array([per_sequence[name] for name in names])
            assert np.isfinite(value).all()
            verified = audit['variants'][variant]
            assert verified['per_sequence_metrics_exact_to_report_csv']
            assert abs(value.mean() - verified['sequence_mean_iou']) <= 1e-12
            if variant in vectors:
                assert np.array_equal(value, vectors[variant])
            vectors[variant] = value
        verified = audit['variants'][label]
        rows[label] = {'root': str(root), 'checkpoint': str(root / 'best.pth'),
                       'best_epoch': fit['arms'][policy]['best_epoch'],
                       'sequence_mean_iou': verified['sequence_mean_iou'],
                       'frame_mean_iou': verified['frame_mean_iou'],
                       'failure_events': verified['failure_event_count'],
                       'failed_frames': verified['failed_frames'],
                       'median_recovery_delay_frames': verified['median_recovery_delay_frames'],
                       'mechanism': verified['independently_recomputed_mechanism_counters'],
                       'efficiency': verified['efficiency']}

resamples = np.random.default_rng(42).integers(0, 98, size=(5000, 98))
pairs = [
    ('current_policy_selector_fit_c1_b384', 'current_policy_fit_c1_b384'),
    ('current_policy_selector_fit_own_b384', 'current_policy_fit_own_b384'),
    ('current_policy_fit_own_b384', 'current_policy_fit_c1_b384'),
    ('current_policy_selector_fit_own_b384', 'current_policy_selector_fit_c1_b384'),
] + [(label, reference) for label in rows for reference in (OLD, 'baseline', 'c1')]
paired = {}
for label, reference in pairs:
    delta = (vectors[label] - vectors[reference]) * 100
    paired[label + '_vs_' + reference] = {
        'delta_sequence_iou_percentage_points': float(delta.mean()),
        'paired_sequence_95_ci': np.percentile(delta[resamples].mean(1), [2.5, 97.5]).tolist(),
        'improved_sequences': int((delta > 0).sum()),
        'worsened_sequences': int((delta < 0).sum()),
        'tied_sequences': int((delta == 0).sum()),
    }
best = max(rows, key=lambda label: rows[label]['sequence_mean_iou'])
better_than_old = rows[best]['sequence_mean_iou'] > float(vectors[OLD].mean())
record = {
    'status': 'FOUR_ACTUAL_FULL98_CELLS_AUDITED',
    'observed_at': datetime.now().astimezone().isoformat(),
    'scope': 'Development held-out98 complete videos, sequence IoU; no formal native accuracy',
    'accepted_audits': accepted, 'cells': rows, 'paired': paired,
    'best_current_cell': best, 'best_current_cell_exceeds_global_old4_internal': better_than_old,
    'native_evaluation_candidate': rows[best]['checkpoint'] if better_than_old else None,
    'global_checkpoint_promoted': False, 'allfive_plus2_native_goal_proven': False,
    'bootstrap_resamples': 5000,
    'limitations': [
        'Bootstrap describes fixed-checkpoint sequence uncertainty, not variation between training seeds.',
        'This repeatedly used internal development split is not a new locked confirmation set.',
        'An internal candidate requires full LasHeR and RGBT234 evaluation with the same checkpoint; global old4 remains protected.',
    ],
    'execution': {'neural_forwards': 0, 'GPU_queries': 0, 'weights_modified': False},
}
destination = SETUP / 'current_policy_four_fit_factor_closure_20261004.json'
destination.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': record['status'], 'path': str(destination), 'best_current_cell': best,
                  'better_than_old4_internal': better_than_old, 'means':
                  {label: row['sequence_mean_iou'] for label, row in rows.items()}}, indent=2))
