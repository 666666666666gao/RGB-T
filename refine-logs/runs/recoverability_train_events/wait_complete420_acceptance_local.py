"""First inspect near the measured full-fit finish estimate, then every300 seconds."""
import datetime
import json
import pathlib
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
folder = pathlib.Path(r'C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\refine-logs\runs\recoverability_train_events')
initial = json.loads((folder / 'actual_all4_complete_fit_optimization_start_observation.json').read_text())
launch = json.loads((folder / 'actual_complete_fit_after_M0_launch.json').read_text())
capacity = json.loads((folder / 'actual_collection_capacity_and_full_start_observation.json').read_text())['receipts']['m0_fit_cpu_acceptance']
estimate = max(row['elapsed_seconds'] / row['epochs'] * initial['arms'][index]['config']['epochs']
               for index, row in enumerate(capacity['arms'].values()))
deadline = datetime.datetime.fromisoformat(launch['started_at_cst']) + datetime.timedelta(seconds=estimate - 180)
time.sleep(max(0., (deadline - datetime.datetime.now(deadline.tzinfo)).total_seconds()))
code = """
import datetime,json,pathlib,subprocess
setup=pathlib.Path('/data/gb/setup')
folder=pathlib.Path('/data/gb/GOLA/refine-logs/runs/recoverability_train_events')
fit=json.loads((setup/'train_events_complete_fit_progress_20261005.json').read_text())
evaluation=json.loads((setup/'train_events_complete_evaluation_progress_20261005.json').read_text())
pids=[fit['pid'],evaluation['pid']]+[row['pid'] for row in fit.get('children',[])]+[row['pid'] for row in evaluation.get('children',[])]
actual=subprocess.run(['ps','-o','pid,ppid,stat,etimes,comm','-p',','.join(map(str,pids))],capture_output=True,text=True).stdout
live=[int(line.split()[0]) for line in actual.splitlines()[1:] if not line.split()[2].startswith('Z')]
rows=[]
for arm in json.loads((folder/'fit_plan.json').read_text())['arms']:
 output=pathlib.Path(arm['output_full'])
 logs=(output/'train.jsonl').read_text().splitlines()
 completion=json.loads((output/'completion.json').read_text()) if (output/'completion.json').exists() else None
 rows.append({'arm':arm['arm'],'last_actual_optimizer_record':json.loads(logs[-1]),'completion':completion})
path=setup/'train_events_full_fit_cpu_acceptance_20261005.json'
receipt=json.loads(path.read_text()) if path.exists() else None
print(json.dumps({'observed_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'fit':fit,'evaluation':evaluation,'actual_process_status':actual,'live_pids':live,'arms':rows,'full_CPU_acceptance':receipt,'new_neural_forward_calls':0,'new_optimizer_steps':0,'power_or_temperature_queried':False}))
"""
while True:
    reply = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                           input=code, capture_output=True, text=True, encoding='utf-8', check=True)
    value = json.loads(reply.stdout)
    (folder / 'actual_complete420_training_acceptance_observation.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'at_cst': value['observed_at_cst'], 'fit_stage': value['fit']['stage'],
                      'evaluation_stage': value['evaluation']['stage'],
                      'steps': [row['last_actual_optimizer_record']['optimizer_steps'] for row in value['arms']],
                      'live_pids': value['live_pids']}), flush=True)
    if value['fit']['stage'] == 'FOUR_COMPLETE_MATCHED420UPDATE_ABC_FITS_READY_FOR_FULL_VIDEO_SELECTION':
        assert value['full_CPU_acceptance']['status'] == 'PASS' and value['full_CPU_acceptance']['all_four_ABC_fits_passed']
        assert all(row['optimizer_steps'] == 420 for row in value['full_CPU_acceptance']['arms'].values())
        break
    assert launch['coordinator_pid'] in value['live_pids'], value['actual_process_status']
    time.sleep(300)
