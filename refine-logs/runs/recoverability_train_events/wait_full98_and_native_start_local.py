"""Observe the same evaluation owner near the measured full98 finish estimate."""
import datetime
import json
import pathlib
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
folder = pathlib.Path(r'C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/recoverability_train_events')
first = json.loads((folder / 'actual_full98_first_wave_and_prior_runtime_intake.json').read_text(encoding='utf-8'))
training = json.loads((folder / 'actual_complete420_training_acceptance_observation.json').read_text(encoding='utf-8'))
assert training['full_CPU_acceptance']['status'] == 'PASS'
assert training['full_CPU_acceptance']['all_four_ABC_fits_passed']
assert all(row['optimizer_steps'] == 420 for row in training['full_CPU_acceptance']['arms'].values())
assert first['evaluation']['pid'] == 1423909 and first['evaluation']['stage'] == 'FULL98_SLOT0_RUNNING'
prior = first['completed_prior_full98_wall_seconds']
assert len(prior) == 16
deadline = datetime.datetime.fromisoformat(first['evaluation']['at_cst']) + datetime.timedelta(
    seconds=4 * max(row['wall_seconds'] for row in prior) - 180)
print(json.dumps({'first_actual_inspection_cst': deadline.isoformat(), 'same_owner_pid': 1423909,
                  'estimate_scope': 'Four serial waves of four complete98-video checkpoints, using maximum actual prior same49418-frame runtime; CPU scoring/model load overhead is not guaranteed.'}), flush=True)
time.sleep(max(0., (deadline - datetime.datetime.now(deadline.tzinfo)).total_seconds()))
code = """
import datetime,json,pathlib,subprocess
setup=pathlib.Path('/data/gb/setup')
state=json.loads((setup/'train_events_complete_evaluation_progress_20261005.json').read_text())
pids=[state['pid']]+[row['pid'] for row in state.get('children',[])]
actual=subprocess.run(['ps','-o','pid,ppid,stat,etimes,comm','-p',','.join(map(str,pids))],capture_output=True,text=True).stdout
live=[int(row.split()[0]) for row in actual.splitlines()[1:] if not row.split()[2].startswith('Z')]
path=setup/'train_events_native_selection_20261005.json'
selection=json.loads(path.read_text()) if path.exists() else None
print(json.dumps({'observed_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  'evaluation':state,'actual_ps':actual,'live_pids':live,'selection':selection,
                  'neural_forward_calls_by_observer':0,'optimizer_steps':0,'GPU_queries':0}))
"""
stages = {'BOTH_COMPLETE_NATIVE_BENCHMARKS_RUNNING', 'BOTH_NATIVE_INFERENCE_COMPLETE_ALL_METRICS_SCORING',
          'COMPLETE420_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY'}
while True:
    result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                            input=code, text=True, encoding='utf-8', capture_output=True, check=True)
    value = json.loads(result.stdout)
    (folder / 'actual_full98_selection_native_start_observation.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'at_cst': value['observed_at_cst'], 'stage': value['evaluation']['stage'],
                      'live_pids': value['live_pids'], 'selection_present': value['selection'] is not None}), flush=True)
    if value['evaluation']['stage'] in stages:
        selected = value['selection']
        assert selected['status'] == 'PASS' and selected['both_datasets_same_fixed_checkpoint']
        assert selected['optimizer_steps'] == 420 and selected['selected_best_epoch'] > 0
        assert len(selected['candidates']) == 16 and all(row['epoch'] > 0 for row in selected['candidates'])
        if value['evaluation']['stage'] == 'BOTH_COMPLETE_NATIVE_BENCHMARKS_RUNNING':
            assert 1423909 in value['live_pids'] and any(row['pid'] in value['live_pids'] for row in value['evaluation']['children'])
        break
    assert 1423909 in value['live_pids'], value['actual_ps']
    time.sleep(300)
