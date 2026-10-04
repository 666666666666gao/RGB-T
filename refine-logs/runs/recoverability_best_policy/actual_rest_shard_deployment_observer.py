import hashlib
import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

setup, main = Path('/data/gb/setup'), Path('/data/gb/GOLA')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
review_path = setup / 'best_policy_shard_source_review_20261004.json'
review = json.loads(review_path.read_text())
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
accept_path = setup / 'best_policy_probe_actual_cpu_acceptance_20261004.json'
accept = json.loads(accept_path.read_text())
assert accept['status'] == 'PASS' and accept['both_partitions_passed'] and accept['clips_per_partition'] == 16
assert accept['prefix_epoch'] == 4 and accept['future_teacher_is_global_best']
runner = setup / 'run_recoverability_best_policy_shard_20261004.sh'
assert sha(runner) == review['source_sha256']['scripts/run_recoverability_best_policy_shard.sh']
assignments = [('train0', 0, 296), ('train1', 1, 295), ('validation', 2, 112), ('train2', 3, 295)]
all_jobs = {}
for shard, gpu, count in assignments:
    jobs_path = setup / ('best_policy_jobs_' + shard + '_rest_20261004.json')
    assert sha(jobs_path) == review['source_sha256']['refine-logs/runs/recoverability_best_policy/jobs_' + shard + '_rest.json']
    all_jobs[shard] = json.loads(jobs_path.read_text())['jobs']
    assert len(all_jobs[shard]) == count
train_probe = json.loads((setup / 'best_policy_jobs_train_probe_20261004.json').read_text())['jobs']
val_probe = json.loads((setup / 'best_policy_jobs_validation_probe_20261004.json').read_text())['jobs']
original = Path('/data/gb/outputs/recoverability_current_policy_merged_20261004/own')
assert train_probe + all_jobs['train0'] + all_jobs['train1'] + all_jobs['train2'] == json.loads((original / 'train/config.json').read_text())['jobs']
assert val_probe + all_jobs['validation'] == json.loads((original / 'validation/config.json').read_text())['jobs']
parent = Path('/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth')
assert sha(parent) == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
observed = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in observed.splitlines()}
assert all(value < 500 for value in used.values()), used
rows = []
for shard, gpu, count in assignments:
    root = Path('/data/gb/outputs/recoverability_best_policy_collect_' + shard + '_rest_20261004')
    assert not root.exists()
    log = setup / (root.name + '.log')
    with log.open('wb') as stream:
        proc = subprocess.Popen(['bash', str(runner), str(gpu), shard], stdout=stream, stderr=subprocess.STDOUT,
                                cwd=main, start_new_session=True)
    rows.append({'shard': shard, 'gpu': gpu, 'runner_pid': proc.pid, 'clips': count, 'root': str(root), 'log': str(log)})
record = {'status': 'LAUNCHED_NOT_COMPLETED', 'launched_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'runs': rows, 'source_review_sha256': sha(review_path), 'actual_probe_CPU_acceptance_sha256': sha(accept_path),
          'parent_sha256': sha(parent), 'same902_128_original_queries_exact': True, 'memory_before_mib': used,
          'collector': 'Unmodified /data/gb/GOLA/research/collect_recoverability.py', 'temperature_or_power_queries': False,
          'batch_clips': 16, 'forward_batch': 64, 'model_or_inference_changes': False,
          'scope': 'Actual bestold4 own-policy TRAIN-data collection; no new method weights or tracking result.'}
(setup / 'best_policy_rest_shard_launch_20261004.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
