"""Qualify TRAIN-only event windows and normal controls; no model or image reads."""
import datetime
import json
from pathlib import Path
import numpy as np
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped
root=Path('/data/gb/GOLA')
folder=root/'refine-logs/runs/recoverability_state_commit'
reference=json.loads((root/'refine-logs/runs/recoverability_best_native_policy/actual_jobs_cpu_acceptance.json').read_text())['reference']
split=json.loads(Path(reference['split']).read_text())
assert len(split['train'])==881 and len(split['validation'])==98 and not set(split['train'])&set(split['validation'])
dataset=MultiModalObjectTrackingDataset_MemoryMapped.load(reference['root'],str(root/reference['cache']))
index={dataset[i].get_name():i for i in range(len(dataset))}
events=json.loads((root/'refine-logs/runs/recoverability_best_native_policy/actual_train_event_query_eligibility_proposal.json').read_text())['new_event_jobs']
assert len(events)==716 and all(j['sequence'] in split['train'] for j in events)
base=json.loads((root/'refine-logs/runs/recoverability_best_native_policy/native_complete/training/pairwise_lr4/config.json').read_text())['train_jobs']
assert len(base)==1762
base_path=Path('/data/gb/outputs/recoverability_best_native_policy_merged_20261005/own/train')
base_config=json.loads((base_path/'config.json').read_text())
assert [(j['sequence'],j['query_frame']) for j in base_config['jobs']]==[tuple(j[:2]) for j in base]
with np.load(base_path/'samples.npz',allow_pickle=False) as archive:
 current=archive['current_iou'].copy()
 choices=archive['original_choice'].copy()
 raw=archive['raw_score'].copy()
assert current.shape==raw.shape==(1762,7,5) and choices.shape==(1762,)
offsets=(-2,-1,0,1,3)
def eligible(sequence,query):
 data=dataset[index[sequence]]
 if query<=0 or query+32>=len(data):return False
 boxes=np.stack([data[t].get_bounding_box() for t in [0]+list(range(query,query+33))])
 return bool(np.isfinite(boxes).all() and (boxes[:,2:]>boxes[:,:2]).all())
qualified=[j for j in events if all(eligible(j['sequence'],j['query_frame']+offset) for offset in offsets)]
rng=np.random.default_rng(42)
selected=[];seen=set()
for late in (False,True):
 candidates=[j for j in qualified if (j['query_frame']>256)==late]
 rng.shuffle(candidates)
 picked=0
 for j in candidates:
  if j['sequence'] in seen:continue
  selected.append(j);seen.add(j['sequence']);picked+=1
  if picked==4:break
 assert picked==4
jobs=[]
for event in selected:
 for offset in offsets:
  jobs.append({'sequence':event['sequence'],'query_frame':event['query_frame']+offset,
               'event_id':event['sequence']+':failure:'+str(event['query_frame']),
               'source_event_frame':event['query_frame'],'event_offset':offset,
               'sampling_role':'old4 TRAIN failure-transition window; current actor outcome unobserved'})
normal_candidates=[]
for i,row in enumerate(base):
 sequence,query=row[:2]
 keep=int(choices[i])
 if sequence in seen or current[i,0,keep]<.5 or not eligible(sequence,query):continue
 normal_candidates.append({'sequence':sequence,'query_frame':query,'event_id':sequence+':normal:'+str(query),
  'sampling_role':'C1 keep normal in closed TRAIN cache; actor outcome not yet observed',
  'cached_C1_keep_iou':float(current[i,0,keep]),'cached_C1_keep_raw_score':float(raw[i,0,keep]),
  'cached_C1_keep_raw_write_eligible':bool(raw[i,0,keep]>.84)})
for writable in (False,True):
 candidates=[j for j in normal_candidates if j['cached_C1_keep_raw_write_eligible']==writable]
 rng.shuffle(candidates);picked=0
 for j in candidates:
  if j['sequence'] in seen:continue
  jobs.append(j);seen.add(j['sequence']);picked+=1
  if picked==4:break
 assert picked==4
assert len(jobs)==48 and len({(j['sequence'],j['query_frame']) for j in jobs})==48
assert len({(j['sequence'],j['event_id']) for j in jobs})==16 and len(seen)==16
assert all(j['sequence'] in split['train'] and eligible(j['sequence'],j['query_frame']) for j in jobs)
packet={'status':'ACTUAL_TRAIN_GT_QUALIFIED_QUERY_JOBS_PREPARED_ONLY','computed_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'jobs':jobs,'query_states':48,'independent_sampling_event_units':16,'sequences':16,'failure_event_units':8,'normal_reference_states':8,
 'event_window_offsets':list(offsets),'all716_events_with_complete_legal_offsets_horizon32':len(qualified),
 'event_prefix_strata':{'through256':4,'after256':4},'normal_cached_C1_raw_write_strata':{'eligible':4,'ineligible':4},
 'source':'closed old4 TRAIN716 transition times and original1762 TRAIN candidate labels',
 'root':reference['root'],'cache':reference['cache'],'split':reference['split'],
 'GT_role':'TRAIN sample eligibility only; values do not enter actor inputs',
 'actor_checkpoint':'NOT_SELECTED_FOR_THIS_PROBE; original complete full98/native chain must close first',
 'scope':'Prepared matched query causal diagnosis, not additional training or native metrics. Query roles and write strata reflect old cached C1 reference, not observed current-policy decisions.',
 'execution':{'images_decoded':0,'NN_calls':0,'optimizer_steps':0,'GPU_queries':0,'live_NN_progress_reads':0,'native_TEST_reads':0}}
(folder/'prepared_geometry_commit_train_jobs_cpu.json').write_text(json.dumps(packet,indent=2)+'\n')
shards=[]
for slot in range(4):
 name='prepared_geometry_commit_train_gpu'+str(slot)+'.json'
 subset=jobs[slot::4]
 (folder/name).write_text(json.dumps({'jobs':subset,'source':'prepared_geometry_commit_train_jobs_cpu.json','TRAIN_GT_eligible':True,'NN_started':False},indent=2)+'\n')
 shards.append({'gpu_slot':slot,'file':name,'queries':len(subset)})
print(json.dumps({k:packet[k] for k in ['status','computed_at_cst','query_states','independent_sampling_event_units','sequences','all716_events_with_complete_legal_offsets_horizon32','event_prefix_strata','normal_cached_C1_raw_write_strata']}|{'shards':shards,'NN_calls':0}))
