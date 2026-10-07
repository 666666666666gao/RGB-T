import json
import subprocess
from pathlib import Path

base=Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/current_event_rollout_fit/scalarfix')
payload='''import datetime,json
from pathlib import Path
root=Path('/data/gb/outputs/current_event_rollout_training_20261007_scalarfix')
progress=json.loads((root/'progress.json').read_text())
rows=[]
for gpu,(arm,ranking,lr) in enumerate(json.loads((root/'plan.json').read_text())['ARMS']):
 folder=root/'fit_full'/arm
 row=dict(gpu=gpu,arm=arm,output=str(folder),config_present=(folder/'config.json').exists(),initial_parity_present=(folder/'initial_parity.json').exists(),completion_present=(folder/'completion.json').exists())
 if row['config_present']:
  config=json.loads((folder/'config.json').read_text())
  row['config']={key:config[key] for key in ('epochs','batch_size','lr','seed','action_ranking','train_clips','validation_clips','checkpoint_selection')}
 if row['initial_parity_present']:
  row['initial_parity']=json.loads((folder/'initial_parity.json').read_text())
 if (folder/'train.jsonl').exists():
  lines=[line for line in (folder/'train.jsonl').read_text().splitlines(keepends=True) if line.endswith('\\n')]
  row['completed_update_records']=len(lines)
  if lines:
   last=json.loads(lines[-1])
   row['last_complete_update']={key:last[key] for key in ('epoch','optimizer_steps','loss','module_gradient_norms','peak_cuda_mib')}
 if row['completion_present']:
  row['completion']=json.loads((folder/'completion.json').read_text())
 rows.append(row)
print(json.dumps(dict(at_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),stage=progress['stage'],arms=rows,CPU_only=True,new_NN_or_optimizer=False,current_stop_request=json.loads((root/'user_stop_request.json').read_text()))),flush=True)
'''
result=subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'],input=payload,text=True,encoding='utf-8',capture_output=True)
(base/'current_fit_CPU_stdout.txt').write_text(result.stdout,encoding='utf-8')
(base/'current_fit_CPU_stderr.txt').write_text(result.stderr,encoding='utf-8')
result.check_returncode()
record=json.loads(result.stdout)
(base/'current_fit_CPU.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
summary=dict(at_cst=record['at_cst'],stage=record['stage'],arms=[{key:value for key,value in row.items() if key not in ('initial_parity','config','completion')} for row in record['arms']],current_stop_request=record['current_stop_request'])
print(json.dumps(summary),flush=True)
