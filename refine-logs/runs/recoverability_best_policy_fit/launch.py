"""Launch one accepted stage once on the four available devices."""
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

stage = sys.argv[1]
assert stage in ('m0', 'full')
setup = Path('/data/gb/setup')
private = Path('/data/gb/experiments/recoverability_best_policy_fit_20261004')
launch_path = setup / ('best_policy_fit_' + stage + '_launch_20261004.json')
assert not launch_path.exists()
assert json.loads((setup / 'best_policy_fit_source_staging_20261004.json').read_text())['status'] == 'PASS'
assert json.loads((setup / 'best_policy_fit_source_review_20261004.json').read_text())['status'] == 'PASS'
gate = json.loads(Path('/data/gb/outputs/recoverability_best_policy_merged_20261004/best_policy_cache_cpu_gate.json').read_text())
assert gate['status'] == 'PASS' and gate['matched_partitions'] == {'train': 902, 'validation': 128}
if stage == 'full':
    accepted = json.loads((setup / 'best_policy_fit_m0_acceptance_20261004.json').read_text())
    assert accepted['status'] == 'PASS' and accepted['all_four_lr_ranking_arms_passed']
assignments = [('pairwise_lr4', 0), ('pairwise_lr5', 1), ('budgeted_lr4', 2), ('budgeted_lr5', 3)]
roots = [Path('/data/gb/outputs/recoverability_best_policy_' + arm + '_' + stage + '_20261004') for arm, _ in assignments]
assert all(not root.exists() for root in roots)
observed = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout
used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in observed.splitlines()}
assert set(used) == {0, 1, 2, 3} and all(value < 500 for value in used.values()), used
rows = []
for (arm, gpu), root in zip(assignments, roots):
    log = setup / (root.name + '.log')
    with log.open('wb') as stream:
        process = subprocess.Popen(['bash', str(private / 'scripts/run_recoverability_best_policy_fit.sh'), str(gpu), arm, stage],
                                   cwd=private, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    rows.append({'arm': arm, 'gpu': gpu, 'runner_pid': process.pid, 'root': str(root), 'log': str(log)})
record = {'status': 'LAUNCHED_NOT_COMPLETED', 'stage': stage, 'runs': rows,
          'launched_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'batch_size': 416, 'seed': 42, 'all_ABC_active': True, 'C1_frozen': True,
          'memory_before_mib': used, 'temperature_or_power_queries': False}
launch_path.write_text(json.dumps(record, indent=2) + '\n')
monitor_log = setup / ('best_policy_fit_' + stage + '_memory_monitor_20261004.log')
with monitor_log.open('wb') as stream:
    monitor = subprocess.Popen([sys.executable, str(setup / 'best_policy_fit_memory_monitor_20261004.py'), stage],
                               cwd=private, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
record['memory_monitor_pid'] = monitor.pid
launch_path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
