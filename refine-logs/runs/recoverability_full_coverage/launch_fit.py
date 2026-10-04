"""Start one real full fit per GPU after every-video GT cache acceptance."""
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

setup = Path('/data/gb/setup')
private = Path('/data/gb/experiments/recoverability_best_policy_fit_20261004')
read = lambda path: json.loads(Path(path).read_text())
review = read(setup / 'full_coverage_pipeline_source_review_20261004.json')
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
gate = read('/data/gb/outputs/recoverability_full_coverage_merged_20261004/full_coverage_cache_cpu_gate.json')
assert gate['status'] == 'PASS' and gate['matched_partitions'] == {'train': 881, 'validation': 98} and gate['all881_98_video_names_exact']
assert gate['all_GT_current_and_history_replayed'] and gate['all_schema_and_sources_verified']
sanity = read(setup / 'best_policy_fit_m0_acceptance_20261004.json')
assert sanity['status'] == 'PASS' and sanity['all_four_lr_ranking_arms_passed'] and sanity['batch_size'] == 416
assignments = [('pairwise_lr4', 0), ('pairwise_lr5', 1), ('budgeted_lr4', 2), ('budgeted_lr5', 3)]
roots = [Path('/data/gb/outputs/recoverability_full_coverage_' + arm + '_full_20261004') for arm, _ in assignments]
out = setup / 'full_coverage_fit_launch_20261004.json'
assert not out.exists() and all(not root.exists() for root in roots)
observed = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout
used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in observed.splitlines()}
assert set(used) == {0, 1, 2, 3} and all(value < 500 for value in used.values()), used
rows = []
for (arm, gpu), root in zip(assignments, roots):
    log = setup / (root.name + '.log')
    with log.open('wb') as stream:
        proc = subprocess.Popen(['bash', str(setup / 'run_full_coverage_fit_20261004.sh'), str(gpu), arm, 'full'],
                                cwd=private, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    rows.append({'arm': arm, 'gpu': gpu, 'runner_pid': proc.pid, 'root': str(root), 'log': str(log)})
record = {'status': 'LAUNCHED_NOT_COMPLETED', 'runs': rows, 'batch_size': 416, 'epochs': 60, 'optimizer_steps_expected': 180,
          'all881_98_video_names_exact': True, 'all_ABC_active': True, 'C1_frozen': True,
          'seed': 42, 'temperature_or_power_queries': False,
          'launched_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
out.write_text(json.dumps(record, indent=2) + '\n')
with (setup / 'full_coverage_fit_memory_monitor_20261004.log').open('wb') as stream:
    monitor = subprocess.Popen([sys.executable, str(setup / 'full_coverage_memory_monitor_20261004.py')],
                               cwd=private, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
record['memory_monitor_pid'] = monitor.pid
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
