"""Select once on developer VAL after actual full fitting, then evaluate all test frames."""
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

setup = Path('/data/gb/setup')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
review = json.loads((setup / 'best_policy_native_source_review_20261004.json').read_text())
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
accepted = json.loads((setup / 'best_policy_fit_actual_full_fit_cpu_audit_20261004.json').read_text())
assert accepted['status'] == 'PASS' and accepted['all_four_lr_ranking_arms_passed']
assert accepted['epochs_per_arm'] == 60 and accepted['optimizer_steps_per_arm'] == 180 and accepted['batch_size'] == 416
arms = ('pairwise_lr4', 'pairwise_lr5', 'budgeted_lr4', 'budgeted_lr5')
assert set(accepted['arms']) == set(arms) and all(accepted['arms'][arm]['status'] == 'PASS' for arm in arms)
selected = max(arms, key=lambda arm: accepted['arms'][arm]['best_validation']['utility'])
row = accepted['arms'][selected]
checkpoint = Path(row['root']) / 'best.pth'
assert sha(checkpoint) == row['artifact_sha256']['best.pth']
runner = setup / 'run_best_policy_native_complete_20261004.sh'
assert sha(runner) == review['source_sha256']['scripts/run_best_policy_native_complete.sh']
selection_path = setup / 'best_policy_native_selection_20261004.json'
launch_path = setup / 'best_policy_native_launch_20261004.json'
base = Path('/data/gb/outputs/recoverability_best_policy_native_full_20261004')
assert not selection_path.exists() and not launch_path.exists() and not base.exists()
observed = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout
used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in observed.splitlines()}
assert set(used) == {0, 1, 2, 3} and all(value < 500 for value in used.values()), used
selection = {'status': 'PASS', 'selected_arm': selected, 'checkpoint': str(checkpoint),
             'checkpoint_sha256': sha(checkpoint), 'selected_best_epoch': row['best_epoch'],
             'best0_is_parent_not_new_learning': row['best0_is_parent_not_new_learning'],
             'all_four_full_fits_CPU_passed': True, 'completed_epochs': 60, 'optimizer_steps': 180,
             'selection_rule': 'Maximum developer VAL128 utility; first arm in declared order on ties; no native-test selection.',
             'cached_utility_by_arm': {arm: accepted['arms'][arm]['best_validation']['utility'] for arm in arms},
             'both_datasets_same_fixed_checkpoint': True, 'new_complete_native_inference_requested_by_user': True,
             'selected_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
selection_path.write_text(json.dumps(selection, indent=2) + '\n')
rows = []
for gpu in range(4):
    log = setup / ('best_policy_native_gpu' + str(gpu) + '_20261004.log')
    with log.open('wb') as stream:
        proc = subprocess.Popen(['bash', str(runner), str(gpu)], cwd='/data/gb/GOLA',
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    rows.append({'gpu': gpu, 'runner_pid': proc.pid, 'log': str(log), 'datasets_in_order': ['lasher', 'rgbt234']})
record = {'status': 'LAUNCHED_NOT_COMPLETE', 'selection': selection, 'runs': rows,
          'expected': {'lasher': {'sequences': 245, 'frames': 220703}, 'rgbt234': {'sequences': 234, 'frames': 116649}},
          'all_test_frames_required': True, 'temperature_or_power_queries': False,
          'reports_complete': False, 'launched_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
launch_path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
