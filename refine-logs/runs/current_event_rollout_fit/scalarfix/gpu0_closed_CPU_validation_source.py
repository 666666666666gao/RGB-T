import json
import subprocess
from pathlib import Path

base = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/current_event_rollout_fit/scalarfix')
payload = '''import datetime,json,os,subprocess
from pathlib import Path
root=Path('/data/gb/outputs/current_event_rollout_training_20261007_scalarfix')
folder=root/'collection/train/gpu0'
done=json.loads((folder/'completion.json').read_text())
config=json.loads((folder/'config.json').read_text())
plan=json.loads((root/'plan.json').read_text())
assert done['completed'] and done['partition']=='train'
assert done['clips']==config['clips']==len(plan['shard_jobs']['train'][0])
assert config['jobs']==plan['shard_jobs']['train'][0]
assert not Path('/proc/2809292').exists(), 'Wait for the actual original GPU0 collector to close'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='/data/gb/GOLA',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
result=subprocess.run(['/data/gb/envs/gola/bin/python','-u','-m','scripts.validate_event_collection','--roots',str(folder)],cwd='/data/gb/GOLA',env=env,capture_output=True,text=True)
print(result.stdout,flush=True)
print(result.stderr,flush=True)
result.check_returncode()
print(json.dumps(dict(status='PASS',at_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),root=str(folder),completion=done,validator_record=json.loads(result.stdout),CPU_only=True,new_NN_or_optimizer=False,actual_original_collector_closed=True)),flush=True)
'''
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'], input=payload, text=True, encoding='utf-8', capture_output=True)
(base/'gpu0_closed_CPU_validation_stdout.txt').write_text(result.stdout,encoding='utf-8')
(base/'gpu0_closed_CPU_validation_stderr.txt').write_text(result.stderr,encoding='utf-8')
result.check_returncode()
record=json.loads(result.stdout.strip().splitlines()[-1])
(base/'gpu0_closed_CPU_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
summary={key:record[key] for key in ('status','at_cst','root','validator_record','CPU_only','new_NN_or_optimizer','actual_original_collector_closed')}
summary['valid_actions']=record['completion']['valid_actions']
with Path('C:/Users/gb/memory/2026-10-08.md').open('a',encoding='utf-8') as stream:
    stream.write('\nRGBT closed GPU0 event shard CPU validation: '+json.dumps(summary)+'; existing NN queue unchanged.\n')
print(json.dumps(summary),flush=True)
