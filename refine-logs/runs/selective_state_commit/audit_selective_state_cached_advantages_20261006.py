"""Audit completed cached head decisions on CPU; never train or rerun tracking."""
import json
import subprocess
from pathlib import Path

base=Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/selective_state_commit')
payload='''import json,os
from collections import defaultdict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.chdir('/data/gb/GOLA')
import numpy as np
import torch
from research.train_selective_state_commit import load_partition
from research.selective_state_commit import SelectiveStateCommitHead,choose_action
torch.set_num_threads(4)
root=Path('/data/gb/outputs/selective_state_training_endpoint_20261006')
collection=Path('/data/gb/outputs/selective_state_collection_20261006')
full=json.loads((root/'full98_report/full_recoverability_report.json').read_text())
assert full['completed'] and full['sequences']==98
selection=json.loads((root/'selected_model.json').read_text())
rows=[]
with torch.no_grad():
 for arm,horizon,penalty in [('H3_lost0',3,0.),('H3_lost01',3,.1),('H32_lost0',32,0.),('H32_lost01',32,.1)]:
  datasets={part:load_partition(collection,part,horizon,penalty,torch.device('cpu')) for part in ('train','validation')}
  for checkpoint in ('best','last'):
   saved=torch.load(root/'fit_full'/arm/(checkpoint+'.pth'),map_location='cpu',weights_only=False)
   head=SelectiveStateCommitHead().eval().requires_grad_(False)
   head.load_state_dict(saved['head'],strict=True)
   for partition,(data,records) in datasets.items():
    scores=head(data['features']).flatten(1)
    keep=data['keep'].long()
    indices=torch.arange(len(keep))
    choices=choose_action(scores.reshape(-1,3,3),data['legal'],torch.zeros_like(keep),keep.bool())
    predicted=scores-scores[indices,keep,None]
    actual=data['utility']-data['utility'][indices,keep,None]
    actual_iou=data['mean_iou']-data['mean_iou'][indices,keep,None]
    nonkeep=data['legal'].flatten(1)&(torch.arange(9)[None]!=keep[:,None])
    intervention=choices!=keep
    chosen_actual=actual[indices,choices]
    chosen_predicted=predicted[indices,choices]
    weights=data['weights']
    by_mode={}
    for mode in range(3):
     valid=nonkeep&(torch.arange(9)[None]%3==mode)
     values=actual[valid].numpy()
     by_mode[str(mode)]={'legal_nonkeep_actions':len(values),'true_positive_actions':int((values>1e-8).sum()),
      'true_above_deployment_margin_actions':int((values>.03).sum()),
      'true_advantage_percentiles_5_25_50_75_95':np.percentile(values,[5,25,50,75,95]).tolist()}
    events=defaultdict(set)
    for record in records: events[record['actual_event_id']].add(record['actual_event_role'])
    label=arm+'_'+checkpoint
    online=full['variants'][label]
    rows.append({'arm':arm,'checkpoint':checkpoint,'epoch':saved['epoch'],'partition':partition,
     'queries':len(records),'distinct_events':len(events),
     'events_with_multiple_future_window_roles':sum(len(roles)>1 for roles in events.values()),
     'changed_queries':int(intervention.sum()),'chosen_modes':torch.bincount(choices%3,minlength=3).tolist(),
     'beneficial_utility_interventions':int((intervention&(chosen_actual>1e-8)).sum()),
     'harmful_utility_interventions':int((intervention&(chosen_actual< -1e-8)).sum()),
     'event_weighted_true_utility_advantage':float((chosen_actual*weights).sum()/weights.sum()),
     'event_weighted_true_mean_iou_advantage':float((actual_iou[indices,choices]*weights).sum()/weights.sum()),
     'event_weighted_predicted_utility_advantage':float((chosen_predicted*weights).sum()/weights.sum()),
     'legal_advantage_event_weighted_mse':float((((predicted-actual).square()*data['legal'].flatten(1)).sum(1)/data['legal'].flatten(1).sum(1)*weights).mean()),
     'legal_nonkeep_labels_by_mode':by_mode,
     'full98_sequence_mean_iou':online['sequence_mean_iou'],
     'full98_delta_vs_zero_parent_pp':100*(online['sequence_mean_iou']-full['variants']['gross_parent']['sequence_mean_iou']),
     'full98_actual_state_commit_interventions':online['counters']['state_commit_interventions'],
     'full98_actual_geometry_holds':online['counters']['state_geometry_holds'],
     'full98_wrong_committed_motion_localization_proxy_frames':online['counters']['failed_committed_motion_frames_localization_proxy']})
receipt={'status':'COMPLETE_CPU_CACHED_ADVANTAGE_AUDIT','rows':rows,'selected_before_native':selection,
 'CPU_cached_head_forward_evaluations':16,'new_visual_backbone_forwards':0,'new_tracking_runs':0,
 'new_optimizer_updates':0,'active_process_or_GPU_queries':0,
 'scope':'Completed TRAIN/repeated-developer cached states and eight completed full98 reports; no native TEST selection or runtime replay',
 'limitation':'CPU cached-score audit is not observed online action-score calibration; online score vectors and parent-C choice were not stored per frame.'}
(root/'cached_advantage_CPU_audit.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
print(json.dumps(receipt,allow_nan=False))
'''
compile(payload,'CPU_cached_advantage_audit','exec')
run=subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'],input=payload,
                   text=True,encoding='utf-8',capture_output=True)
(base/'cached_advantage_CPU_stdout.txt').write_text(run.stdout,encoding='utf-8')
(base/'cached_advantage_CPU_stderr.txt').write_text(run.stderr,encoding='utf-8')
run.check_returncode()
receipt=json.loads(run.stdout)
(base/'cached_advantage_CPU_audit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'rows':len(receipt['rows']),
 'summary':[{k:row[k] for k in ('arm','checkpoint','epoch','partition','changed_queries',
 'beneficial_utility_interventions','harmful_utility_interventions','event_weighted_true_utility_advantage',
 'full98_delta_vs_zero_parent_pp','full98_actual_state_commit_interventions','full98_actual_geometry_holds')} for row in receipt['rows']]}),flush=True)
