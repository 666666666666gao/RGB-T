"""Wait for actual four fits and full videos, then run the reviewed CPU audits."""
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
PRIVATE = Path('/data/gb/experiments/recoverability_current_policy_fit_20261004')
REVIEW = Path('/data/gb/setup/current_policy_completion_review')
SETUP = Path('/data/gb/setup')
PIPELINE = SETUP / 'current_policy_completion_pipeline_20261004'
PYTHON = '/data/gb/envs/gola/bin/python'
STEMS = ('current_policy_fit', 'current_policy_selector_fit')
roots = [Path('/data/gb/outputs') / f'recoverability_{stem}_{policy}_b384_full_20261004'
         for stem in STEMS for policy in ('c1', 'own')]
source = json.loads((REVIEW / 'completion_audit_queue_source_review.json').read_text())
assert source['status'] == 'PASS' and source['review_independence'] == 'same-family'
assert source['acceptance_status'] == 'provisional'
PIPELINE.mkdir(parents=True, exist_ok=False)


def state(phase):
    record = {'phase': phase, 'observed_at': datetime.now().astimezone().isoformat()}
    (PIPELINE / 'state.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


state('WAIT_ALL4_TRAINING_COMPLETIONS')
while not all((root / 'training_completed.txt').is_file() for root in roots):
    time.sleep(240)
state('ACTUAL_CPU_FULL_FIT_AUDITS')
for mode, stem in zip(('oracle', 'selector'), STEMS):
    subprocess.run([PYTHON, str(REVIEW / f'{mode}_actual_full_fit_cpu_audit.py')], cwd=PRIVATE, check=True)
    receipt = json.loads((SETUP / f'{stem}_actual_full_fit_cpu_audit_20261004.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['both_teacher_arms_passed']
    assert receipt['epochs_per_arm'] == 60 and receipt['optimizer_steps_per_arm'] == 180
state('WAIT_ALL4_FULL98_REPORTS')
while not all((root / 'job_completed.txt').is_file() for root in roots):
    time.sleep(240)
state('ACTUAL_CPU_FULL98_AUDITS')
for mode in ('oracle', 'selector'):
    subprocess.run([PYTHON, str(REVIEW / f'{mode}_actual_full98_cpu_audit.py')], cwd=PRIVATE, check=True)
state('ACTUAL_FOUR_CELL_COMPARISON')
subprocess.run([PYTHON, str(REVIEW / 'four_factor_closure.py')], cwd=PRIVATE, check=True)
state('ALL_ACTUAL_INTERNAL_AUDITS_COMPLETED')
(PIPELINE / 'job_completed.txt').write_text(datetime.now().astimezone().isoformat() + '\n')
