"""One read-only sample of the actual uneven four-arm fit; no NN instrumentation."""
import json
import subprocess
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
output = parent / 'GOLA-source/refine-logs/runs/current_policy_aggregation/profile_output'
output.mkdir(exist_ok=True)
destination = output / 'actual_uneven_fit_runtime_20261007.json'
assert not destination.exists()
payload = '''import datetime,json,statistics,subprocess
from pathlib import Path
root=Path('/data/gb/outputs/current_policy_aggregation_20261006')
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
arms=[('old_lr5',602124,72),('aggregate_lr5',602125,64),('old_lr3',602126,72),('aggregate_lr3',602127,64)]
profiles=[]
for name,pid,epochs in arms:
 folder=root/'fit_full'/name
 rows=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
 first=[row for row in rows if row['step']==1]
 per_epoch=[b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(first,first[1:])]
 recent=per_epoch[-5:]
 estimate=statistics.median(recent)
 latest=rows[-1]
 remaining_steps=576-latest['optimizer_steps']
 updates_per_epoch=8 if name.startswith('old_') else 9
 remaining_seconds=remaining_steps*estimate/updates_per_epoch
 completed=(folder/'completion.json').is_file()
 receipt=json.loads((folder/'completion.json').read_text()) if completed else None
 config=json.loads((folder/'config.json').read_text())
 profiles.append({'arm':name,'gpu':len(profiles),'pid':pid,'alive':Path('/proc/'+str(pid)).exists(),
  'completed':completed,'actual_completion':receipt,'config_epochs':config['epochs'],
  'latest_actual_train_row':latest,'last_five_complete_epoch_seconds':recent,
  'median_recent_epoch_seconds_including_validation':estimate,
  'estimated_remaining_training_seconds':remaining_seconds,
  'estimated_training_end_cst':(at+datetime.timedelta(seconds=remaining_seconds)).isoformat(),
  'estimate_scope':'Latest logged update and last five first-step intervals; excludes final reload and report',
  'source_model_unchanged':True})
commands=[['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.current.sm,pstate','--format=csv,noheader,nounits'],
 ['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],
 ['ps','-p','589745,602124,602125,602126,602127','-o','pid,ppid,stat,etime,pcpu,rss,args','--no-headers']]
snapshots=[{'command':command,'output':subprocess.check_output(command,text=True)} for command in commands]
result={'at_cst':at.isoformat(),'owner_alive':Path('/proc/589745').exists(),
 'stage':json.loads((root/'progress.json').read_text())['stage'],'arms':profiles,'resource_samples':snapshots,
 'no_power_or_temperature_query':True,'no_NN_source_change_restart_or_checkpoint_cleanup':True,
 'instrumentation_changelog':[{'path':'profile_output/actual_uneven_fit_runtime_20261007.json','change':'created standalone read-only sample; no application instrumentation'}]}
print(json.dumps(result))
'''
compile(payload, '<read-only-runtime-profile>', 'exec')
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                        input=payload, text=True, encoding='utf-8', capture_output=True)
(output / 'runtime_profile_stdout.txt').write_text(result.stdout, encoding='utf-8')
(output / 'runtime_profile_stderr.txt').write_text(result.stderr, encoding='utf-8')
result.check_returncode()
profile = json.loads(result.stdout)
destination.write_text(json.dumps(profile, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'at_cst':profile['at_cst'],'owner_alive':profile['owner_alive'],'stage':profile['stage'],
 'arms':[{'arm':a['arm'],'alive':a['alive'],'completed':a['completed'],
          'epoch':a['latest_actual_train_row']['epoch'],'optimizer_steps':a['latest_actual_train_row']['optimizer_steps'],
          'recent_epoch_seconds':a['median_recent_epoch_seconds_including_validation'],
          'estimated_training_end_cst':a['estimated_training_end_cst']} for a in profile['arms']],
 'resource_samples':profile['resource_samples'],'NN_source_modified':False}),flush=True)
