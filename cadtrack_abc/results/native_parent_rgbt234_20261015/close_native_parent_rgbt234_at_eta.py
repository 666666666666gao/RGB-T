"""Validate the completed parallel parent evaluation and score it on CPU."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess

project = Path('/home/gaob/CADTrack-ABC')
root = project / 'outputs/native_parent_rgbt234_20261015'
launch = json.loads((root / 'launch.json').read_text())
receipt_paths = [root / 'predictions' / f'complete_shard{i}.json' for i in range(2)]
state = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
    processes_present=[Path('/proc', str(job['pid'])).exists() for job in launch['jobs']],
    complete_receipts_present=[path.exists() for path in receipt_paths])
print(json.dumps(state), flush=True)
assert not any(state['processes_present']) and all(state['complete_receipts_present'])
receipts = [json.loads(path.read_text()) for path in receipt_paths]
assert all(row['completed'] and row['dataset'] == 'rgbt234' and row['epoch'] == 2
           and row['policy_family'] == 'native' and row['checkpoint'] == launch['checkpoint'] for row in receipts)
records = [row for receipt in receipts for row in receipt['records']]
assert len(records) == len({row['sequence'] for row in records}) == 234
assert sum(row['frames'] for row in records) == 116649
checkpoint_sha = hashlib.sha256(Path(launch['checkpoint']).read_bytes()).hexdigest()
assert checkpoint_sha == '5c6b6fcbf18b11918ae3495ccfa927cf56a3f5c7f3cecd76aabaaa94d80cc988'
reports = root / 'reports'
assert not (reports / 'full_report.json').exists()
command = ['/data/wangwj/envs/CADTrack/bin/python', '-m', 'cadtrack_abc.score',
    '--dataset', 'rgbt234', '--root', launch['dataset_root'],
    '--predictions', str(root / 'predictions'), '--output', str(reports)]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
           OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
with (root / 'score_cpu.log').open('w') as stream:
    child = subprocess.run(command, cwd=project, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
report = json.loads((reports / 'full_report.json').read_text())
assert report['completed'] and report['sequences'] == 234 and report['frames'] == 116649
assert set(report['overall_metrics_percent']) == {'MPR', 'MSR'}
result = dict(status='FULL_NATIVE_ENHANCED_PARENT_RGBT234_SCORED',
    observed_utc=datetime.now(timezone.utc).isoformat(), checkpoint=launch['checkpoint'],
    checkpoint_sha256=checkpoint_sha, epoch=2, sequences=234, frames=116649,
    overall_metrics_percent=report['overall_metrics_percent'],
    scoring_command=command, scoring_exit_code=child.returncode,
    scoring_source_sha256=hashlib.sha256((project / 'cadtrack_abc/score.py').read_bytes()).hexdigest(),
    detached_inference_exit_codes='Not independently captured; both completion receipts and process exit observed.',
    scope='Previously trained native epoch2 enhanced parent, not pure CAD baseline or new search_actions.',
    new_neural_jobs_started=False, original_collection_modified=False)
(root / 'completion_verified.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
