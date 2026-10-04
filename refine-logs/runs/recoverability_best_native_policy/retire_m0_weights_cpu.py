"""Retire only the eight completed new-data capacity-check weights; preserve full fits and teachers."""
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
m0 = read(setup / 'best_native_policy_m0_fit_cpu_acceptance_20261005.json')
full = read(setup / 'best_native_policy_full_fit_cpu_acceptance_20261005.json')
progress = read(setup / 'best_native_policy_pipeline_progress_20261005.json')
assert m0['status'] == full['status'] == 'PASS'
assert m0['epochs_per_arm'] == 3 and m0['optimizer_steps_per_arm'] == 15
assert full['epochs_per_arm'] == 60 and full['optimizer_steps_per_arm'] == 300
assert m0['all_four_full_fits_CPU_passed'] and full['all_four_full_fits_CPU_passed']
assert progress['stage'].startswith('FULL98_')
arms = ('pairwise_lr4', 'pairwise_lr5', 'budgeted_lr4', 'budgeted_lr5')
assert set(m0['arms']) == set(full['arms']) == set(arms)
targets, protected = [], {}
for arm in arms:
    root = Path(m0['arms'][arm]['root'])
    assert root.resolve() == outputs / ('recoverability_best_native_policy_' + arm + '_m0_20261005')
    assert sorted(p.name for p in root.glob('*.pth')) == ['best.pth', 'epoch_003.pth']
    config, done = read(root / 'config.json'), read(root / 'completion.json')
    assert config['epochs'] == done['epochs'] == 3 and done['optimizer_steps'] == 15 and done['completed']
    for name, ck in m0['arms'][arm]['checkpoints'].items():
        assert name in ('best.pth', 'epoch_003.pth')
        assert all(ck['ABC_changed'].values()) and ck['all_C1_tensors_exact'] and ck['cached_CPU_replay_pass']
        path = root / name
        assert path.resolve().is_relative_to(outputs) and path.is_file()
        targets.append({'path': str(path), 'epoch': ck['epoch'], 'bytes': path.stat().st_size, 'sha256': sha(path)})
    full_root = Path(full['arms'][arm]['root'])
    assert full_root.resolve() == outputs / ('recoverability_best_native_policy_' + arm + '_full_20261005')
    for name, ck in full['arms'][arm]['checkpoints'].items():
        assert all(ck['ABC_changed'].values()) and ck['all_C1_tensors_exact'] and ck['cached_CPU_replay_pass']
        protected[str(full_root / name)] = sha(full_root / name)
prior = read(setup / 'full_coverage_inferior_weights_retirement_completed_20261005.json')
for path, digest in prior['protected_sha256'].items():
    assert sha(path) == digest
    protected[path] = digest
assert len(targets) == 8 and not set(row['path'] for row in targets) & set(protected)
processes = subprocess.run(['ps', '-eo', 'pid=,args='], text=True, capture_output=True, check=True).stdout
assert not any(row['path'] in line for row in targets for line in processes.splitlines())
proof_path = setup / 'best_native_policy_m0_weights_retirement_proof_20261005.json'
completed_path = setup / 'best_native_policy_m0_weights_retirement_completed_20261005.json'
assert not proof_path.exists() and not completed_path.exists()
proof = {'status': 'PREDELETION_PASS', 'reason': 'Capacity-only runs completed; all full60/300 fits separately completed and passed CPU. Current continuous selection uses only full-fit weights.',
         'files': targets, 'file_count': 8, 'total_bytes': sum(row['bytes'] for row in targets),
         'protected_sha256': protected, 'no_live_capacity_weight_path_references': True,
         'actual_current_controller_stage': progress['stage'], 'full_fit_CPU_accepted_at_cst': full['accepted_at_cst'],
         'capacity_logs_metrics_configs_CPU_receipts_and_best_validation_kept': True,
         'new_GPU_queries': 0, 'neural_calls': 0}
proof_path.write_text(json.dumps(proof, indent=2) + '\n')
for row in targets:
    Path(row['path']).unlink()
assert all(not Path(row['path']).exists() for row in targets)
assert all(sha(path) == digest for path, digest in protected.items())
proof.update(status='REMOVED_VERIFIED', completed_at_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),
             protected_before_after_exact=True,
             cumulative_removed_weight_count=prior['cumulative_removed_weight_count'] + 8,
             cumulative_removed_nominal_bytes=prior['cumulative_removed_nominal_bytes'] + proof['total_bytes'])
completed_path.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps({k: proof[k] for k in ('status', 'file_count', 'total_bytes', 'completed_at_cst', 'cumulative_removed_weight_count', 'cumulative_removed_nominal_bytes')}), flush=True)
