"""Launch one reviewed collection queue per idle GPU; sanity and full are separate."""
import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--phase', choices=('sanity', 'full'), required=True)
args = p.parse_args()
setup = Path('/data/gb/setup')
read = lambda path: json.loads(Path(path).read_text())
review = read(setup / 'best_native_policy_collection_source_review_20261005.json')
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
prepared = read(setup / 'best_native_policy_jobs_cpu_acceptance_20261005.json')
assert prepared['status'] == 'PASS' and prepared['all881_98_names_twice_and_disjoint']
parts = ('sanity',) if args.phase == 'sanity' else ('train', 'validation')
if args.phase == 'full':
    assert read(setup / 'best_native_policy_sanity_cpu_acceptance_20261005.json')['status'] == 'PASS'
out = setup / f'best_native_policy_{args.phase}_launch_20261005.json'
assert not out.exists()
for part in parts:
    for gpu in range(4):
        row = prepared['shards'][f'{part}_gpu{gpu}']
        assert len(read(row['jobs_file'])['jobs']) == row['clips']
        assert not Path(f'/data/gb/outputs/recoverability_best_native_policy_collect_{part}_gpu{gpu}_20261005').exists()
observed = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout
used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in observed.splitlines()}
assert set(used) == {0, 1, 2, 3} and all(value < 500 for value in used.values()), used
rows = []
for gpu in range(4):
    log = setup / f'best_native_policy_{args.phase}_gpu{gpu}_20261005.log'
    with log.open('wb') as stream:
        proc = subprocess.Popen(['bash', str(setup / 'run_best_native_policy_collect_20261005.sh'), str(gpu), args.phase],
                                cwd='/data/gb/GOLA', stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    rows.append({'gpu': gpu, 'runner_pid': proc.pid, 'log': str(log)})
record = {'status': 'LAUNCHED_NOT_COMPLETED', 'phase': args.phase, 'runs': rows,
          'teacher': prepared['teacher'], 'teacher_epoch': 4, 'collector_and_architecture_changed': False,
          'training_started': False, 'native_started': False, 'power_or_temperature_queries': False,
          'launched_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
