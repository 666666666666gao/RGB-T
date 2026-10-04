"""Launch four original collectors after exact every-video CPU coverage preparation."""
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

setup = Path('/data/gb/setup')
read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
review = read(setup / 'full_coverage_collect_source_review_20261004.json')
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
prepared = read(setup / 'full_coverage_jobs_cpu_acceptance_20261004.json')
assert prepared['status'] == 'PASS' and prepared['eligible_train_videos'] == 881 and prepared['eligible_validation_videos'] == 98
assert prepared['all_train_video_names_exact'] and prepared['all_validation_video_names_exact']
assert sha(prepared['teacher']) == prepared['teacher_sha256']
runner = setup / 'run_full_coverage_collect_20261004.sh'
assert sha(runner) == review['source_sha256']['scripts/run_full_coverage_collect.sh']
for part in ('train', 'validation'):
    jobs = []
    for gpu in range(4):
        row = prepared['shards'][part + '_gpu' + str(gpu)]
        shard = read(row['jobs_file'])['jobs']
        assert len(shard) == row['clips'] and row['gpu'] == gpu
        assert not Path('/data/gb/outputs/recoverability_full_coverage_collect_' + part + '_gpu' + str(gpu) + '_20261004').exists()
        jobs.extend(shard)
    assert jobs == prepared['jobs'][part]
out = setup / 'full_coverage_collection_launch_20261004.json'
assert not out.exists()
observed = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout
used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in observed.splitlines()}
assert set(used) == {0, 1, 2, 3} and all(value < 500 for value in used.values()), used
rows = []
for gpu in range(4):
    log = setup / ('full_coverage_collect_gpu' + str(gpu) + '_20261004.log')
    with log.open('wb') as stream:
        proc = subprocess.Popen(['bash', str(runner), str(gpu)], cwd='/data/gb/GOLA', stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    rows.append({'gpu': gpu, 'runner_pid': proc.pid, 'log': str(log)})
record = {'status': 'LAUNCHED_NOT_COMPLETED', 'runs': rows, 'teacher': prepared['teacher'],
          'teacher_epoch': prepared['teacher_epoch'], 'full881_98_video_coverage_required': True,
          'batch_clips': 16, 'forward_batch': 64, 'temperature_or_power_queries': False,
          'collector_changed': False, 'training_started': False, 'native_started': False,
          'launched_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat()}
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
