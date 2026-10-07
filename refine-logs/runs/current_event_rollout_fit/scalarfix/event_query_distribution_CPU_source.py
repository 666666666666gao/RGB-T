import json
import subprocess
from pathlib import Path

base=Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/current_event_rollout_fit/scalarfix')
payload='''import datetime,json
from collections import Counter
from pathlib import Path
root=Path('/data/gb/outputs/current_event_rollout_training_20261007_scalarfix')
plan=json.loads((root/'plan.json').read_text())
split=json.loads(Path('/data/gb/outputs/c1_initial_seed42/split.json').read_text())
report={}
for partition,expected in [('train',16452),('validation',1824)]:
 jobs=[job for rows in plan['shard_jobs'][partition] for job in rows]
 count=Counter(job['sequence'] for job in jobs)
 assert len(jobs)==expected and set(count)==set(split['train' if partition=='train' else 'validation'])
 assert len({(job['sequence'],job['query_frame']) for job in jobs})==expected
 top=count.most_common(10)
 report[partition]=dict(queries=expected,videos=len(count),largest_video=top[0][0],largest_video_queries=top[0][1],largest_video_share=top[0][1]/expected,top5_query_share=sum(value for _,value in top[:5])/expected,top10_query_share=sum(value for _,value in top)/expected,equal_video_weight=1/len(count),counts=dict(sorted(count.items())),shards=[dict(gpu=gpu,queries=len(rows),videos=len({job['sequence'] for job in rows}),largest=Counter(job['sequence'] for job in rows).most_common(1)[0]) for gpu,rows in enumerate(plan['shard_jobs'][partition])])
print(json.dumps(dict(status='PASS',at_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),report=report,scope='Locked event-query inventory concentration only; no new predictions, no estimate of why accuracy changes',CPU_only=True,new_NN_or_optimizer=False,architecture_reward_or_recipe_changed=False)),flush=True)
'''
result=subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'],input=payload,text=True,encoding='utf-8',capture_output=True)
(base/'event_query_distribution_CPU_stdout.txt').write_text(result.stdout,encoding='utf-8')
(base/'event_query_distribution_CPU_stderr.txt').write_text(result.stderr,encoding='utf-8')
result.check_returncode()
record=json.loads(result.stdout)
(base/'event_query_distribution_CPU.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
summary={partition:{key:value for key,value in row.items() if key!='counts'} for partition,row in record['report'].items()}
with Path('C:/Users/gb/memory/2026-10-08.md').open('a',encoding='utf-8') as stream:
 stream.write('\nRGBT '+record['at_cst']+' CPU lockedevent query distribution '+json.dumps(summary)+'; cached validation train_event_recoverability.py75-78/174 query-mean H3 utility, complete full98 remains final native candidate selector. No new NN/optimizer/recipe changes; no causal performance inference.\n')
print(json.dumps(summary),flush=True)
