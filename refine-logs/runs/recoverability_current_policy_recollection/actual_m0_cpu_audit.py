"""Accept the actual two-query current-policy collector pair on CPU only."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    review = Path('/data/gb/setup/recoverability_current_policy_recollection')
    source = json.loads((review / 'source_review.json').read_text())
    assert source['status'] == 'PASS_SOURCE_ONLY'
    prefix = source['current_prefix_model']
    jobs = json.loads((review / 'm0_jobs.json').read_text())['jobs']
    assert len(jobs) == 2 and all(1 <= job['query_frame'] <= 64 for job in jobs)
    original = json.loads(Path(source['original_jobs_directory'], 'train.json').read_text())
    assert all(job in original for job in jobs)
    split = json.loads(Path('/data/gb/outputs/c1_initial_seed42/split.json').read_text())
    assert all(job['sequence'] in split['train'] for job in jobs)
    assert not set(split['train']) & set(split['validation'])
    arrays, outputs, counts, pause_actions = {}, {}, {}, {}
    for policy in ('c1', 'own'):
        root = Path('/data/gb/outputs/recoverability_current_policy_m0_20261004') / policy
        assert (root / 'job_completed.txt').is_file()
        config = json.loads((root / 'config.json').read_text())
        completion = json.loads((root / 'completion.json').read_text())
        progress = json.loads((root / 'progress.json').read_text())
        assert config['jobs'] == jobs and config['clips'] == completion['clips'] == 2
        assert config['partition'] == completion['partition'] == 'train'
        assert completion['completed'] is True and completion['decision_input_contains_future'] is False
        assert progress['completed_clips'] == progress['clips'] == 2
        assert config['prefix_model'] == prefix and config['prefix_checkpoint_epoch'] == 25
        assert config['future_policy_mode'] == policy and config['prefix_write_verification'] == 'action'
        assert config['batch_clips'] == config['forward_batch'] == 1
        assert config['max_prefix'] == 1024 and config['seed'] == 42
        assert config['head'] == '/data/gb/outputs/c1_initial_seed42/best.pth'
        assert config['motion_run'] == '/data/gb/outputs/abc_joint_v1_seed42'
        assert config['future_horizon'] == 3 and config['decision_inputs_copied_before_future_decode'] is True
        if policy == 'own':
            assert config['future_checkpoint_epoch'] == 25
            assert config['future_execution'] == 'per-action serial, identical deployed step and write verification'
        with np.load(root / 'samples.npz') as archive:
            data = {key: archive[key].copy() for key in archive.files}
        assert all(value.shape[0] == 2 and np.isfinite(value).all() for value in data.values())
        assert data['valid'].shape == (2, 7, 5)
        assert data['future_iou'].shape == (2, 7, 5, 2, 3)
        assert data['history_valid'].shape == (2, 1024)
        assert data['history_valid'][:, -1].all()
        for row, job in enumerate(jobs):
            frames = data['history_frames'][row][data['history_valid'][row]]
            assert np.all(frames < job['query_frame']) and frames[-1] == job['query_frame'] - 1
        np.testing.assert_array_equal(data['action_valid'][..., 0], data['valid'])
        np.testing.assert_array_equal(data['action_valid'][..., 1], data['valid'] & (data['raw_score'] > .84))
        assert data['action_valid'][np.arange(2), 0, data['original_choice'], 0].all()
        assert int(data['action_valid'].sum()) == completion['valid_actions']
        assert np.all((0 <= data['future_iou']) & (data['future_iou'] <= 1))
        wrong = data['wrong_update_fraction']
        assert np.all((0 <= wrong) & (wrong <= 1))
        np.testing.assert_array_equal(wrong * 4, np.round(wrong * 4))
        np.testing.assert_array_equal(data['prefix_counts'].sum(0), completion['actual_prefix_counts_sum'])
        arrays[policy], outputs[policy] = data, str(root)
        counts[policy] = completion['actual_prefix_counts_sum']
        pause_actions[policy] = int(data['action_valid'][..., 1].sum())
    assert set(arrays['c1']) == set(arrays['own'])
    nonfuture = sorted(set(arrays['c1']) - {'future_iou', 'wrong_update_fraction'})
    for key in nonfuture:
        np.testing.assert_array_equal(arrays['c1'][key], arrays['own'][key], err_msg=key)
    receipt = {
        'status': 'PASS_ACTUAL_MATCHED_M0', 'accepted_at': datetime.now(timezone.utc).isoformat(),
        'prefix_model': prefix, 'prefix_checkpoint_epoch': 25, 'jobs': jobs,
        'completed_policies': ['c1', 'own'], 'outputs': outputs,
        'nonfuture_arrays_bitwise_equal': True, 'nonfuture_array_names': nonfuture,
        'actual_prefix_counts_sum': counts, 'pause_action_count': pause_actions,
        'future_iou_max_absolute_difference': float(np.abs(arrays['c1']['future_iou'] - arrays['own']['future_iou']).max()),
        'wrong_update_fraction_max_absolute_difference': float(np.abs(arrays['c1']['wrong_update_fraction'] - arrays['own']['wrong_update_fraction']).max()),
        'scope': 'MATCHED_FULL_COLLECTION_ONLY',
        'source_review': str(review / 'source_review.json'),
        'review_independence': 'same-family', 'acceptance_status': 'provisional',
        'state_distribution_shift_proven': False, 'tracking_accuracy_claim_allowed': False,
        'full_training_allowed': False, 'optimizer_updates': 0,
        'limits': 'Two short TRAIN queries establish current-checkpoint CLI completion and paired inputs. Source review supplies unchanged causal step semantics; this CPU audit does not replay future boxes or prove broad state changes.'
    }
    destination = review / 'actual_m0_acceptance.json'
    assert not destination.exists()
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
