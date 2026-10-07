import json
import subprocess
from pathlib import Path

base=Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/current_event_rollout_fit/scalarfix')
payload='''import datetime,json
from pathlib import Path
root=Path('/data/gb/outputs/current_event_rollout_training_20261007_scalarfix')
plan=json.loads((root/'plan.json').read_text())
split=json.loads(Path('/data/gb/outputs/c1_initial_seed42/split.json').read_text())
events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
closed=[row for row in events if row['stage']=='COLLECT_FULL_TRAIN_ALL_PROCESSES_CLOSED_PASS'][-1]
assert closed['exits']==[[gpu,0] for gpu in range(4)]
shards=[]
all_jobs=set()
all_names=set()
for gpu in range(4):
 folder=root/'collection/train'/f'gpu{gpu}'
 config=json.loads((folder/'config.json').read_text())
 done=json.loads((folder/'completion.json').read_text())
 assert done['completed'] and done['partition']==config['partition']=='train'
 assert config['jobs']==plan['shard_jobs']['train'][gpu]
 assert done['clips']==config['clips']==len(config['jobs'])
 assert config['prefix_model_family']=='ABC_candidate_relations' and config['prefix_checkpoint_epoch']==5
 assert config['prefix_search_value']=='gross' and config['prefix_write_verification']=='action'
 assert config['future_policy_mode']=='own' and config['future_horizon']==3
 assert done['exact_actual_prefix_parity'] and not done['decision_input_contains_future']
 assert all(row['all_actual_prefix_outputs_and_decisions_exact'] and row['duplicate_prefix_replays']==0 for row in done['records'])
 jobs={(row['sequence'],row['query_frame']) for row in config['jobs']}
 names={row['sequence'] for row in done['records']}
 assert len(jobs)==done['clips']==sum(row['queries'] for row in done['records'])
 assert len(names)==done['sequences'] and names=={name for name,_ in jobs}
 assert not all_jobs.intersection(jobs) and not all_names.intersection(names)
 all_jobs.update(jobs);all_names.update(names)
 shards.append(dict(gpu=gpu,root=str(folder),clips=done['clips'],sequences=done['sequences'],valid_actions=done['valid_actions'],exact_actual_prefix_parity=done['exact_actual_prefix_parity'],decision_input_contains_future=done['decision_input_contains_future']))
assert len(all_jobs)==plan['TRAIN_jobs']==16452
assert all_names==set(split['train']) and len(all_names)==881 and not all_names.intersection(split['validation'])
progress=json.loads((root/'progress.json').read_text())
print(json.dumps(dict(status='PASS',at_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),closed_success_event=closed,shards=shards,clips=len(all_jobs),sequences=len(all_names),valid_actions=sum(row['valid_actions'] for row in shards),actual_original_four_process_exit_codes=[0,0,0,0],exact_native_prefix_receipts_all=True,scope='Complete TRAIN receipt/config/plan membership and successful process exits; full serialized tensor validation remains in queued TRAIN+DEV verifier',current_stage=progress['stage'],DEV_expected_jobs=plan['DEV_jobs'],current_children=progress.get('children',[]),CPU_only=True,new_NN_or_optimizer=False,queue_recipe_changed=False)),flush=True)
'''
result=subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'],input=payload,text=True,encoding='utf-8',capture_output=True)
(base/'full_TRAIN_closed_receipts_CPU_stdout.txt').write_text(result.stdout,encoding='utf-8')
(base/'full_TRAIN_closed_receipts_CPU_stderr.txt').write_text(result.stderr,encoding='utf-8')
result.check_returncode()
record=json.loads(result.stdout)
(base/'full_TRAIN_closed_receipts_CPU.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-08.md').open('a',encoding='utf-8') as stream:
 stream.write('\nRGBT completeTRAINeventreceipts '+record['at_cst']+': '+json.dumps(record)+'; CPU-only complete membership/receipts/actualexit0, not model training or native TEST scores.\n')
print(json.dumps({key:record[key] for key in ('status','at_cst','sequences','clips','valid_actions','actual_original_four_process_exit_codes','current_stage','DEV_expected_jobs','CPU_only','new_NN_or_optimizer','queue_recipe_changed')}),flush=True)
