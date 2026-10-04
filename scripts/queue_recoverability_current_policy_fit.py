"""Wait for the matched collection, run four real M0s, audit, then start full fits."""
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

PYTHON = '/data/gb/envs/gola/bin/python'
PRIVATE = Path('/data/gb/experiments/recoverability_current_policy_fit_20261004')
REVIEW = Path('/data/gb/setup/current_policy_fit_review')
QUEUE_REVIEW = Path('/data/gb/setup/current_policy_four_fit_queue_source_review_20261004.json')
PIPELINE = Path('/data/gb/setup/current_policy_four_fit_pipeline_20261004')
CACHE = Path('/data/gb/outputs/recoverability_current_policy_merged_20261004')
CELLS = [('oracle', 'c1', 0), ('selector', 'c1', 1), ('oracle', 'own', 2), ('selector', 'own', 3)]
STEMS = {'oracle': 'recoverability_current_policy_fit', 'selector': 'recoverability_current_policy_selector_fit'}
ACCEPTANCE = {
    'oracle': Path('/data/gb/setup/current_policy_fit_m0_acceptance_20261004.json'),
    'selector': Path('/data/gb/setup/current_policy_selector_fit_m0_acceptance_20261004.json'),
}


def state(phase, **details):
    record = {'phase': phase, 'observed_at': datetime.now().astimezone().isoformat(), **details}
    (PIPELINE / 'state.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


def run_root(mode, policy, stage):
    return Path('/data/gb/outputs') / f'{STEMS[mode]}_{policy}_b384_{stage}_20261004'


def launch(stage):
    jobs = []
    for mode, policy, gpu in CELLS:
        script = PRIVATE / 'scripts' / f'run_{STEMS[mode]}.sh'
        session = f'current_policy_{mode}_{policy}_b384_{stage}_20261004'
        log = PIPELINE / f'{mode}_{policy}_{stage}.log'
        command = shlex.join(['bash', str(script), str(gpu), policy, stage])
        command += ' > ' + shlex.quote(str(log)) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, command], check=True)
        pane = subprocess.check_output(['tmux', 'list-panes', '-t', session, '-F', '#{pane_pid}'], text=True).strip()
        jobs.append({'mode': mode, 'policy': policy, 'gpu': gpu, 'session': session,
                     'pane_pid': int(pane), 'output': str(run_root(mode, policy, stage)), 'log': str(log)})
    state(stage.upper() + '_LAUNCHED', jobs=jobs)


review = json.loads(QUEUE_REVIEW.read_text())
assert review['status'] == 'PASS' and review['review_independence'] == 'same-family'
assert review['acceptance_status'] == 'provisional'
PIPELINE.mkdir(parents=True, exist_ok=False)
markers = [Path('/data/gb/outputs') / f'recoverability_current_policy_{policy}_{part}_shard{shard}_20261004/job_completed.txt'
           for policy in ('c1', 'own') for part in ('train', 'validation') for shard in range(4)]
while not all(marker.is_file() for marker in markers):
    state('WAIT_COMPLETE_COLLECTION', completed_artifacts=sum(marker.is_file() for marker in markers), expected_artifacts=16)
    time.sleep(240)
state('CPU_MATCHED_CACHE_MERGE')
subprocess.run([PYTHON, str(REVIEW / 'merge_current_policy_caches.py'), '--output', str(CACHE)], check=True)
cache = json.loads((CACHE / 'paired_cache_gate.json').read_text())
assert cache['status'] == 'PASS' and cache['all_non_future_arrays_exact']
assert cache['matched_partitions'] == {'train': 902, 'validation': 128}
launch('m0')
while not all((run_root(mode, policy, 'm0') / 'job_completed.txt').is_file() for mode, policy, _ in CELLS):
    time.sleep(240)
state('ACTUAL_CPU_M0_ACCEPTANCE')
for helper in ('actual_m0_cpu_audit.py', 'selector_actual_m0_cpu_audit.py'):
    subprocess.run([PYTHON, str(REVIEW / helper)], cwd=PRIVATE, check=True)
for mode, path in ACCEPTANCE.items():
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'PASS' and receipt['both_teacher_arms_passed']
    assert receipt['batch_size'] == 384 and receipt['optimizer_steps_per_arm'] == 9
    assert receipt['parent_epoch'] == 4
state('BOTH_ACTUAL_M0_ACCEPTANCES_PASS', acceptance_paths={k: str(v) for k, v in ACCEPTANCE.items()})
launch('full')
(PIPELINE / 'full_launch_completed.txt').write_text(datetime.now().astimezone().isoformat() + '\n')
