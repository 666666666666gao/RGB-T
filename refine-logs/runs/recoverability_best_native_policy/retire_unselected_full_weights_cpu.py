"""Retire fifteen fully evaluated weights after fixed full-video selection; keep the native model and teachers."""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
setup = Path('/data/gb/setup')
outputs = Path('/data/gb/outputs').resolve()
read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
proof_path = setup / 'best_native_policy_unselected_full_weights_retirement_proof_20261005.json'
completed_path = setup / 'best_native_policy_unselected_full_weights_retirement_completed_20261005.json'
assert not proof_path.exists() and not completed_path.exists()
full = read(setup / 'best_native_policy_full_fit_cpu_acceptance_20261005.json')
selection = read(setup / 'best_native_policy_native_selection_20261005.json')
progress = read(setup / 'best_native_policy_pipeline_progress_20261005.json')
prior = read(setup / 'best_native_policy_m0_weights_retirement_completed_20261005.json')
assert full['status'] == selection['status'] == 'PASS' and prior['status'] == 'REMOVED_VERIFIED'
assert full['all_four_full_fits_CPU_passed'] and full['epochs_per_arm'] == 60 and full['optimizer_steps_per_arm'] == 300
assert selection['both_datasets_same_fixed_checkpoint'] and selection['selected_best_epoch'] == 40
assert progress['stage'] == 'BOTH_COMPLETE_NATIVE_BENCHMARKS'
owner_state = subprocess.run(['ps', '-o', 'stat=', '-p', '632884'], capture_output=True, text=True, check=True).stdout.strip()
assert owner_state and not owner_state.startswith('Z')
selected = outputs / 'recoverability_best_native_policy_budgeted_lr4_full_20261005/epoch_040.pth'
assert Path(selection['checkpoint']).resolve() == selected
assert len(selection['candidates']) == 16
evaluated = {row['model']: row for row in selection['candidates']}
assert len(evaluated) == 16
targets, all_current = [], set()
for arm in ('pairwise_lr4', 'pairwise_lr5', 'budgeted_lr4', 'budgeted_lr5'):
    root = outputs / ('recoverability_best_native_policy_' + arm + '_full_20261005')
    receipt = full['arms'][arm]
    assert Path(receipt['root']).resolve() == root
    assert sorted(p.name for p in root.glob('*.pth')) == ['best.pth', 'epoch_020.pth', 'epoch_040.pth', 'epoch_060.pth']
    done = read(root / 'completion.json')
    assert done['completed'] and done['epochs'] == 60 and done['optimizer_steps'] == 300
    for name, info in receipt['checkpoints'].items():
        path = root / name
        assert path.resolve().is_relative_to(outputs) and path.is_file()
        assert info['ABC_changed'] == {'A': True, 'B': True, 'C': True} and info['all_C1_tensors_exact'] and info['cached_CPU_replay_pass']
        row = evaluated[str(path)]
        score = read(Path(row['predictions']).parent / 'full98_selection_score.json')
        assert score['completed'] and score['actual_TRAIN_GT_scored'] and not score['native_test_data_read']
        assert score['model'] == str(path) and score['epoch'] == row['epoch'] == info['epoch']
        assert score['sequences'] == 98 and score['frames'] == 49418 and len(score['per_sequence']) == 98
        assert score['sequence_mean_iou'] == row['sequence_mean_iou']
        digest = sha(path)
        assert prior['protected_sha256'][str(path)] == digest
        all_current.add(str(path))
        if path != selected:
            targets.append({'path': str(path), 'epoch': info['epoch'], 'bytes': path.stat().st_size, 'sha256': digest})
assert all_current == set(evaluated) and len(targets) == 15
target_paths = {row['path'] for row in targets}
protected = {path: digest for path, digest in prior['protected_sha256'].items() if path not in target_paths}
assert len(protected) == 9 and str(selected) in protected and not target_paths & set(protected)
assert all(sha(path) == digest for path, digest in protected.items())
processes = subprocess.run(['ps', '-eo', 'pid=,args='], text=True, capture_output=True, check=True).stdout
assert not any(path in line for path in target_paths for line in processes.splitlines())
proof = {'status': 'PREDELETION_PASS', 'reason': 'All sixteen trained checkpoints completed full98 actual-GT scoring; one positive checkpoint fixed before both native tests. Remaining original controller and final merger read only the selected weight and accepted metadata.',
         'files': targets, 'file_count': 15, 'total_bytes': sum(row['bytes'] for row in targets),
         'protected_sha256': protected, 'selected_checkpoint': str(selected),
         'all16_full98_actual_GT_completed': True, 'no_live_exact_target_weight_path_references': True,
         'actual_controller_stage': progress['stage'], 'all_training_logs_configs_metrics_CPU_receipts_and_full98_predictions_kept': True,
         'GPU_queries': 0, 'neural_calls': 0, 'original_controller_modified': False}
proof_path.write_text(json.dumps(proof, indent=2) + '\n')
for row in targets:
    Path(row['path']).unlink()
assert all(not Path(path).exists() for path in target_paths)
assert all(sha(path) == digest for path, digest in protected.items())
proof.update(status='REMOVED_VERIFIED', completed_at_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),
             protected_before_after_exact=True,
             cumulative_removed_weight_count=prior['cumulative_removed_weight_count'] + 15,
             cumulative_removed_nominal_bytes=prior['cumulative_removed_nominal_bytes'] + proof['total_bytes'])
completed_path.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps({key: proof[key] for key in ('status', 'file_count', 'total_bytes', 'selected_checkpoint', 'completed_at_cst', 'cumulative_removed_weight_count', 'cumulative_removed_nominal_bytes')}), flush=True)
