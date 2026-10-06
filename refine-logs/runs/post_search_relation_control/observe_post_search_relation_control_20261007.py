"""Observe one original reviewed owner;240 seconds or measured near-end ETA."""
import datetime
import json
import subprocess
import time
from pathlib import Path

base = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/post_search_relation_control')
assert not (base / 'sole_observer_complete.json').exists()
launch = json.loads((base / 'actual_pipeline_launch.json').read_text())

tz = datetime.timezone(datetime.timedelta(hours=8))
now = datetime.datetime.now(tz)
next_read = now + datetime.timedelta(seconds=180)
payload = '''import json
from pathlib import Path
from datetime import datetime,timedelta,timezone
root=Path('/data/gb/outputs/post_search_relation_control_20261007')
progress=json.loads((root/'progress.json').read_text());native=root/'native/progress.json'
observed_progress=json.loads(native.read_text()) if native.is_file() and progress['stage']=='ONE_MODEL_LOCKED_BEFORE_NATIVE' else progress
def alive(pid):
 path=Path('/proc/'+str(pid)+'/stat')
 return path.exists() and path.read_text().split(') ',1)[1].split()[0]!='Z'
value={'observed_at_cst':datetime.now(timezone(timedelta(hours=8))).isoformat(),'owner_pid':OWNER,
 'owner_alive':alive(OWNER),'progress':progress,'observed_phase':observed_progress,
 'children_alive':[{**c,'alive':alive(c['pid'])} for c in observed_progress.get('children',[])],
 'scope':'Read-only owner/progress/closed receipts/current train log; no NN/reports/cleanup'}
if progress['stage'] in ('FIT_SANITY','FIT_FULL'):
 stage=progress['stage'].lower();arms=[('one_way_lr5',64),('post_search_lr5',64),('one_way_lr3',64),('post_search_lr3',64)]
 rows=[]
 for name,epochs in arms:
  path=root/stage/name/'train.jsonl'
  if path.is_file():
   lines=path.read_text().splitlines()
   if lines:rows.append({'arm':name,**json.loads(lines[-1])})
 value['actual_training_rows']=rows
 if progress['stage']=='FIT_FULL':
  estimates=[json.loads((root/'fit_sanity'/name/'completion.json').read_text())['elapsed_seconds']*epochs/2 for name,epochs in arms]
  value['estimated_phase_seconds_from_closed_sanity']=max(estimates)
if observed_progress['stage'] in ('LASHER_SHARDS','RGBT234_SHARDS'):
 value['estimated_phase_seconds_from_prior_native']=3332 if observed_progress['stage']=='LASHER_SHARDS' else 1069
check=root/'causal_pair_check.json'
if check.is_file():value['closed_causal_pair_check']=json.loads(check.read_text())
print(json.dumps(value))
'''.replace('OWNER', str(launch['pid']))
while True:
    (base / 'pipeline_next_observation.json').write_text(json.dumps({'next_poll_cst': next_read.isoformat(),
        'pipeline_pid': launch['pid'], 'minimum_repeat_seconds': 240, 'first_read_seconds': 180,
        'passive_only': True}, indent=2) + '\n')
    while datetime.datetime.now(tz) < next_read:
        time.sleep(min(60, max(0, (next_read - datetime.datetime.now(tz)).total_seconds())))
    result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'], input=payload,
                            text=True, encoding='utf-8', capture_output=True)
    (base / 'pipeline_observation_stdout.txt').write_text(result.stdout, encoding='utf-8')
    (base / 'pipeline_observation_stderr.txt').write_text(result.stderr, encoding='utf-8')
    result.check_returncode()
    actual = json.loads(result.stdout)
    (base / 'pipeline_observation.json').write_text(json.dumps(actual, indent=2) + '\n')
    with (base / 'pipeline_observations.jsonl').open('a') as stream: stream.write(json.dumps(actual) + '\n')
    print(json.dumps(actual), flush=True)
    now = datetime.datetime.now(tz)
    if actual['progress']['stage'] == 'COMPLETE_POST_SEARCH_RELATION_CONTROL_AND_NATIVE' or not actual['owner_alive']:
        (base / 'sole_observer_complete.json').write_text(json.dumps({'at_cst': now.isoformat(), 'final_observation': actual}, indent=2) + '\n')
        break
    next_read = now + datetime.timedelta(seconds=240)
    if actual['progress']['stage'] == 'FIT_FULL':
        next_read = max(next_read, datetime.datetime.fromisoformat(actual['progress']['at_cst']) +
                        datetime.timedelta(seconds=actual['estimated_phase_seconds_from_closed_sanity'] - 180))
    if actual['observed_phase']['stage'] in ('LASHER_SHARDS', 'RGBT234_SHARDS'):
        next_read = max(next_read, datetime.datetime.fromisoformat(actual['observed_phase']['at_cst']) +
                        datetime.timedelta(seconds=actual['estimated_phase_seconds_from_prior_native'] - 180))
