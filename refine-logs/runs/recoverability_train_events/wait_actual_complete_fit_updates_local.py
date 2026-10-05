"""Inspect the sole successor after measured M0 warm-up; confirm all four actual optimizers."""
import datetime
import json
import pathlib
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
folder = pathlib.Path(r'C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\refine-logs\runs\recoverability_train_events')
launch = json.loads((folder / 'actual_complete_fit_after_M0_launch.json').read_text())
deadline = datetime.datetime.fromisoformat(launch['started_at_cst']) + datetime.timedelta(minutes=3)
time.sleep(max(0., (deadline - datetime.datetime.now(deadline.tzinfo)).total_seconds()))
code = """
import datetime,json,pathlib,subprocess
root=pathlib.Path('/data/gb/GOLA/refine-logs/runs/recoverability_train_events')
setup=pathlib.Path('/data/gb/setup')
fit=json.loads((setup/'train_events_complete_fit_progress_20261005.json').read_text())
evaluation=json.loads((setup/'train_events_complete_evaluation_progress_20261005.json').read_text())
plan=json.loads((root/'fit_plan.json').read_text())
pids=[fit['pid'],evaluation['pid']]+[row['pid'] for row in fit.get('children',[])]
actual=subprocess.run(['ps','-o','pid,ppid,stat,etimes,comm','-p',','.join(map(str,pids))],capture_output=True,text=True)
live=[int(line.split()[0]) for line in actual.stdout.splitlines()[1:] if not line.split()[2].startswith('Z')]
arms=[]
for arm in plan['arms']:
 output=pathlib.Path(arm['output_full'])
 path=output/'train.jsonl'
 lines=path.read_text().splitlines() if path.exists() else []
 row=json.loads(lines[-1]) if lines else None
 config=json.loads((output/'config.json').read_text()) if (output/'config.json').exists() else None
 arms.append({'arm':arm['arm'],'gpu':arm['gpu'],'last_actual_optimizer_record':row,'config':{key:config[key] for key in ('epochs','batch_size','lr','seed','train_clips','validation_clips','frozen_modules','init_checkpoint')} if config else None})
print(json.dumps({'observed_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'fit':fit,'evaluation':evaluation,'actual_process_status':actual.stdout,'live_pids':live,'arms':arms,'power_or_temperature_queried':False,'neural_forward_calls_by_observer':0,'optimizer_steps_by_observer':0}))
"""
while True:
    reply = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                           input=code, capture_output=True, text=True, encoding='utf-8', check=True)
    value = json.loads(reply.stdout)
    (folder / 'actual_all4_complete_fit_optimization_start_observation.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    ready = all(row['last_actual_optimizer_record'] and row['last_actual_optimizer_record']['optimizer_steps'] > 0
                for row in value['arms'])
    print(json.dumps({'at_cst': value['observed_at_cst'], 'all_four_actual_optimizer_updates_confirmed': ready,
                      'stage': value['fit']['stage'], 'steps': [row['last_actual_optimizer_record']['optimizer_steps'] if row['last_actual_optimizer_record'] else 0 for row in value['arms']],
                      'live_pids': value['live_pids']}), flush=True)
    if ready:
        for row in value['arms']:
            assert all(amount > 0 for amount in row['last_actual_optimizer_record']['module_gradient_norms'].values())
            assert row['config']['batch_size'] == 384 and row['config']['frozen_modules'] == []
        break
    assert launch['coordinator_pid'] in value['live_pids'], value['actual_process_status']
    time.sleep(300)
