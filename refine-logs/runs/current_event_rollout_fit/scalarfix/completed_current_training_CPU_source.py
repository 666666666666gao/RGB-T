"""Read and archive completed current-training receipts without running a model."""
import datetime
import hashlib
import json
import math
import subprocess
from pathlib import Path

folder = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
repo = folder / 'GOLA-source'
destination = repo / 'refine-logs/runs/current_event_rollout_fit/scalarfix'
payload = '''import datetime,hashlib,json,math
from pathlib import Path
root=Path('/data/gb/outputs/current_event_rollout_training_20261007_scalarfix')
stop=json.loads((root/'user_stop_after_current_training_complete.json').read_text())
assert stop['completed'] and stop['all_four_fit_workers_closed']
assert stop['total_optimizer_updates']==49440
assert not stop['new_full98_or_native_evaluation_started']
request=json.loads((root/'user_stop_request.json').read_text())
assert request['boundary']=='complete_current_training'
owner_fields=Path('/proc',str(stop['owner_pid']),'stat').read_text().split(') ',1)[1].split()
assert owner_fields[0]=='T'
def process(pid):
 path=Path('/proc',str(pid),'stat')
 if not path.exists():return dict(pid=pid,alive=False,state=None)
 fields=path.read_text().split(') ',1)[1].split()
 return dict(pid=pid,alive=fields[0]!='Z',state=fields[0],exit_code=int(fields[49]) if fields[0]=='Z' else None)
arms=[]
for item in stop['receipts']:
 state=process(item['process']['pid'])
 assert not state['alive'] and state.get('exit_code') in (0,None)
 output=Path(item['output'])
 assert output.parent==root/'fit_full'
 completion=json.loads((output/'completion.json').read_text())
 assert completion==item['completion']
 assert completion['completed'] and completion['epochs']==24 and completion['optimizer_steps']==12360
 assert completion['modules_changed']=={'A':True,'B':True,'C':True}
 assert all(value>0 for value in completion['max_module_gradient_norms'].values())
 assert completion['frozen_C1_weights_exact'] and not completion['GOLA_and_motion_new_gradients']
 assert not completion['native_TEST_metrics_new']
 config=json.loads((output/'config.json').read_text())
 assert config['epochs']==24 and config['batch_size']==32
 assert config['train_clips']==16452 and config['validation_clips']==1824
 parity=json.loads((output/'initial_parity.json').read_text())
 assert parity['train']['initial_parent_decision_changes']==parity['validation']['initial_parent_decision_changes']==0
 metrics=json.loads((output/'metrics.json').read_text())
 assert len(metrics)==25 and [row['epoch'] for row in metrics]==list(range(25))
 assert all(math.isfinite(value) for row in metrics for value in row.values())
 best=completion['checkpoints']['best.pth']
 last=completion['checkpoints']['last.pth']
 assert best['strict_reload'] and last['strict_reload'] and last['epoch']==24
 assert best['validation']['utility']==max(row['utility'] for row in metrics)
 assert best['validation']=={key:value for key,value in metrics[best['epoch']].items() if key!='epoch'}
 assert last['validation']=={key:value for key,value in metrics[-1].items() if key!='epoch'}
 count=0
 with (output/'train.jsonl').open() as stream:
  for line in stream:
   row=json.loads(line);count+=1
   assert row['optimizer_steps']==count and math.isfinite(row['loss'])
 assert count==12360
 weights={name:dict(path=str(output/name),bytes=(output/name).stat().st_size) for name in ('best.pth','last.pth','initial.pth')}
 arms.append(dict(arm=output.name,gpu=item['gpu'],process=state,config=config,initial_parity=parity,completion=completion,metrics=metrics,verified_optimizer_log_records=count,weights=weights))
assert len(arms)==4 and sum(row['verified_optimizer_log_records'] for row in arms)==49440
launch=json.loads((root/'user_stop_guard_launch.json').read_text())
source=Path('/data/gb/GOLA/research/train_event_recoverability.py').read_bytes()
print(json.dumps(dict(at_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),CPU_only=True,new_NN_or_optimizer=False,user_request=request,stop_completion=stop,coordinator=dict(pid=stop['owner_pid'],state=owner_fields[0],intentionally_held=True),guard=process(launch['pid']),arms=arms,all_four_training_completed_and_verified=True,native_TEST_metrics_new=False,weights_retained_pending_native_evaluation=True,training_source_SHA256=hashlib.sha256(source).hexdigest())),flush=True)
'''
result = subprocess.run(
    ['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
    input=payload, text=True, encoding='utf-8', capture_output=True,
)
(destination / 'completed_current_training_CPU_stdout.txt').write_text(result.stdout, encoding='utf-8')
(destination / 'completed_current_training_CPU_stderr.txt').write_text(result.stderr, encoding='utf-8')
result.check_returncode()
record = json.loads(result.stdout)
local_source = repo / 'research/train_event_recoverability.py'
assert hashlib.sha256(local_source.read_bytes()).hexdigest() == record['training_source_SHA256']
(destination / 'completed_current_training_CPU.json').write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
)
summary = dict(
    at_cst=record['at_cst'],
    all_four_training_completed_and_verified=record['all_four_training_completed_and_verified'],
    total_optimizer_updates=sum(row['verified_optimizer_log_records'] for row in record['arms']),
    arms=[dict(arm=row['arm'], best_epoch=row['completion']['checkpoints']['best.pth']['epoch'],
               initial_utility=row['metrics'][0]['utility'],
               best_utility=row['completion']['checkpoints']['best.pth']['validation']['utility'],
               last_utility=row['metrics'][-1]['utility']) for row in record['arms']],
    native_TEST_metrics_new=False,
)
with Path('C:/Users/gb/memory/2026-10-08.md').open('a', encoding='utf-8') as stream:
    stream.write('\nRGBT CPU completed-current-training intake: ' + json.dumps(summary) + '\n')
print(json.dumps(summary), flush=True)
