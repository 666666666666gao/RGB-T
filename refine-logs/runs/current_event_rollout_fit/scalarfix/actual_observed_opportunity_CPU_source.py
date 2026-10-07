import json,subprocess
from pathlib import Path

payload='''import json,numpy as np
from pathlib import Path
root=Path('/data/gb/outputs/current_event_rollout_training_20261007_scalarfix/collection/train')
references={path.name.removesuffix('_recoverability_decisions.npz'):path for path in Path('/data/gb/outputs/current_policy_train_traces_20261007/full').rglob('*_recoverability_decisions.npz')}
rows=[]
for gpu in range(4):
 folder=root/('gpu%d'%gpu)
 records=[json.loads(line) for line in (folder/'progress.jsonl').read_bytes().split(b'\\n')[:-1]][:4]
 for record in records:
  name=record['sequence']
  with np.load(folder/(name+'_queries.npz')) as archive,np.load(references[name]) as native:
   queries=archive['query_frames'];n=len(queries);idx=np.arange(n);t=queries-1
   valid=archive['action_valid'];current=archive['current_iou'];future=archive['future_iou'].mean(-1)
   utility=.7*current[...,None]+.3*future-.1*archive['wrong_update_fraction']
   parent=archive['parent_flat_action'];native_parent=2*native['choice'][t]+native['pause'][t]
   assert np.array_equal(parent,native_parent)
   assert np.array_equal(archive['parent_search_triggered'],native['search_requested'][t])
   executed=native['extra_executed'][t].astype(bool);region=native['searched_region'][t].astype(int)
   available=np.zeros((n,7),dtype=bool);available[:,0]=True
   assert ((region[executed]>=1)&(region[executed]<=6)).all()
   available[idx[executed],region[executed]]=True
   native_valid=native['valid'][t].reshape(n,7,5)
   expected_valid=archive['valid']&available[...,None]
   assert np.array_equal(native_valid,expected_valid),name
   native_boxes=native['boxes_xyxy'][t].reshape(n,7,5,4).astype(np.float32)
   assert np.array_equal(native_boxes[native_valid],archive['image_boxes'][native_valid]),name
   observed=valid&available[...,None,None]
   flatutility=utility.reshape(n,-1);flatvalid=observed.reshape(n,-1)
   assert flatvalid[idx,parent].all()
   pvalue=flatutility[idx,parent]
   observed_action=np.where(flatvalid,flatutility,-np.inf).argmax(-1)
   observed_value=flatutility[idx,observed_action]
   local_action=np.where(valid[:,0].reshape(n,-1),flatutility[:,:10],-np.inf).argmax(-1)
   local_value=flatutility[idx,local_action]
   region_cost=np.zeros((7,1,1));region_cost[1:]=.01
   all_net=np.where(valid,utility-region_cost,-np.inf).reshape(n,-1)
   all_action=all_net.argmax(-1);all_value=all_net[idx,all_action]
   actual_cost=.01*archive['parent_search_triggered']
   cur=current.reshape(n,-1)
   for i,query in enumerate(queries):
    covered=current[i][archive['valid'][i]&available[i,:,None]]
    local_covered=current[i,0][archive['valid'][i,0]]
    rows.append(dict(sequence=name,query=int(query),actual_extra_executed=bool(executed[i]),actual_searched_region=int(region[i]),
     parent_action=int(parent[i]),parent_current_iou=float(cur[i,parent[i]//2]),
     observed_GT_action=int(observed_action[i]),observed_gain=float(observed_value[i]-pvalue[i]),
     observed_GT_current_iou=float(cur[i,observed_action[i]//2]),
     observed_GT_future_mean=float(future.reshape(n,-1)[i,observed_action[i]]),
     parent_future_mean=float(future.reshape(n,-1)[i,parent[i]]),
     local_GT_gain=float(local_value[i]-pvalue[i]),
     all_regions_GT_action=int(all_action[i]),all_regions_net_gain=float(all_value[i]-(pvalue[i]-actual_cost[i])),
     search_allocation_gap=float(all_value[i]-(observed_value[i]-actual_cost[i])),
     actual_available_has_correct=bool((covered>=.5).any()),local_has_correct=bool((local_covered>=.5).any())))
opportunities=[row for row in rows if row['observed_gain']>.03]
failures=[row for row in rows if row['parent_current_iou']<.2]
extra_opportunities=[row for row in opportunities if row['observed_GT_action']//10>0]
report=dict(CPU_only=True,closed_videos=len({row['sequence'] for row in rows}),closed_queries=len(rows),
 actual_extra_executed_queries=sum(row['actual_extra_executed'] for row in rows),
 exact_actual_available_masks_and_float32_boxes=True,
 actual_observed_GT_utility_gain_gt_003=len(opportunities),
 local_only_GT_utility_gain_gt_003=sum(row['local_GT_gain']>.03 for row in rows),
 observed_GT_best_extra_actions=len(extra_opportunities),
 observed_positive_gain_lower_H3_mean=sum(row['observed_GT_future_mean']<row['parent_future_mean']-.03 for row in opportunities),
 parent_failed_queries=len(failures),failed_with_actual_available_correct=sum(row['actual_available_has_correct'] for row in failures),
 actual_extra_introduced_correct_missing_local=sum(row['actual_extra_executed'] and not row['local_has_correct'] and row['actual_available_has_correct'] for row in rows),
 parent_failed_utility_optimum_correct=sum(row['parent_current_iou']<.2 and row['observed_GT_current_iou']>=.5 for row in opportunities),
 all_regions_GT_net_gain_gt_003=sum(row['all_regions_net_gain']>.03 for row in rows),
 GT_search_allocation_gap_gt_003=sum(row['search_allocation_gap']>.03 for row in rows),
 utility='.7 currentIoU+.3 H3futuremean-.1 wrong-updatefraction',
 cost='Observed decision-only comparison cancels the already incurred search cost; global region oracle pays .01 iff it chooses extra, actual parent/observed set pay actual request .01',
 scope='Same biased first4 closed videos per original shard:367 TRAIN queries, GT-assisted H3 upper bound; actual chosen region only, not deployable performance/native accuracy',
 new_NN_or_optimizer=False,architecture_reward_or_recipe_changed=False)
print(json.dumps(dict(report=report,queries=rows)))
'''
result=subprocess.run(['ssh','-T','2027','env CUDA_VISIBLE_DEVICES= /data/gb/envs/gola/bin/python -'],input=payload,text=True,encoding='utf-8',capture_output=True)
base=Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/current_event_rollout_fit/scalarfix')
(base/'actual_observed_opportunity_CPU_stdout.txt').write_text(result.stdout,encoding='utf-8')
(base/'actual_observed_opportunity_CPU_stderr.txt').write_text(result.stderr,encoding='utf-8')
result.check_returncode();value=json.loads(result.stdout)
(base/'actual_observed_opportunity_CPU.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value['report']),flush=True)
