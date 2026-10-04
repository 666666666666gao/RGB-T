from pathlib import Path
import numpy as np,json,datetime,hashlib
root=Path('/data/gb/outputs/recoverability_future_policy_merged_20261004')
result={'status':'PASS_OFFLINE_GT_LABEL_DIAGNOSTIC','observed_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'902 TRAIN and128 heldout query labels only; fixed same-state candidate/action utilities; no NN or native-test tuning','formula':'0.7 current_iou +0.3 mean(future_iou) -0.1 wrong_update_fraction; extra region cost0.01','strict_oracle_policy':'Use rawGT max net only if strict net>0.03; otherwise original regular keep action. Not a deployment policy.','oracle_boundary':'All six extra regions were previously executed by collector. Max over their actions is an offline upper bound, not actual one-forward online selector performance.','initial_checker_failure':'KeyError0: jobs.json is an object with jobs list; corrected reader to exact existing schema, no source/data/model changes','arms':{}}
for arm in ['c1','own']:
 result['arms'][arm]={}
 for part in ['train','validation']:
  path=root/arm/part/'samples.npz'
  with np.load(path) as z:
   current=z['current_iou'].astype(np.float64);future=z['future_iou'].astype(np.float64);wrong=z['wrong_update_fraction'].astype(np.float64);valid=z['valid'];legal=z['action_valid'];keep=z['original_choice'].astype(int);raw=z['raw_score']
  n=len(keep);rows=np.arange(n);assert n=={'train':902,'validation':128}[part]
  utility=.7*current[...,None]+.3*future.mean(-1)-.1*wrong
  reference=utility[rows,0,keep,0]
  costs=np.zeros((1,7,1,1));costs[:,1:]=.01
  net=utility-reference[:,None,None,None]-costs
  current_keep=current[rows,0,keep]
  target_mask=(current>=.5)&valid
  local_correct=target_mask[:,0].any(-1)
  extra_correct=target_mask[:,1:].any((1,2))
  missing_found=(~local_correct)&extra_correct
  oracle_net=np.where(legal,net,-np.inf).reshape(n,-1)
  unthresholded_choice=oracle_net.argmax(-1)
  best_net=oracle_net[rows,unthresholded_choice]
  choice=np.where(best_net>.03,unthresholded_choice,keep*2);r=choice//10;c=(choice%10)//2
  selected_current=current[rows,r,c]
  qualifying_recovery=legal&(current[...,None]>=.5)&(net>.03)
  extra_qual=qualifying_recovery[:,1:].any((1,2,3))
  harmful=(current_keep[:,None,None,None]>=.5)&(current[...,None]<.2)&legal
  conflicting=harmful&(net>.05)
  useful=legal&(net>.05)
  pause_valid=legal[...,1]
  pause_gain=utility[...,1]-utility[...,0]
  d={'queries':n,'valid_candidates':int(valid.sum()),'valid_actions':int(legal.sum()),'keep_failed_queries':int((current_keep<.2).sum()),'keep_correct_queries':int((current_keep>=.5).sum()),'local_missing_queries':int((~local_correct).sum()),'local_missing_with_extra_correct_queries':int(missing_found.sum()),'among_missing_found_with_correct_extra_net_gt_003':int((missing_found&extra_qual).sum()),'positive_net_gt_005_actions':int(useful.sum()),'queries_with_positive_net_gt_005':int(useful.any((1,2,3)).sum()),'harmful_current_actions_with_positive_net_gt_005':int(conflicting.sum()),'queries_with_harmful_current_positive_net_gt_005':int(conflicting.any((1,2,3)).sum()),'strict_oracle_threshold':.03,'unthresholded_net_oracle_changed_location_queries':int(((unthresholded_choice//10!=0)|((unthresholded_choice%10)//2!=keep)).sum()),'net_oracle_changed_location_queries':int(((r!=0)|(c!=keep)).sum()),'net_oracle_rescue_queries':int(((current_keep<.2)&(selected_current>=.5)).sum()),'net_oracle_harm_queries':int(((current_keep>=.5)&(selected_current<.2)).sum()),'legal_pause_actions':int(pause_valid.sum()),'legal_pause_gain_above_003':int((pause_valid&(pause_gain>.03)).sum()),'legal_pause_gain_below_minus003':int((pause_valid&(pause_gain<-.03)).sum()),'keep_raw_above084_queries':int((raw[rows,0,keep]>.84).sum()),'max_positive_pause_gain':float(pause_gain[pause_valid].max()) if pause_valid.any() else None}
  jobs=json.loads((root/arm/part/'jobs.json').read_text())['jobs'];assert len(jobs)==n
  d['query_labels']=[{'sequence':jobs[i]['sequence'],'query_frame':jobs[i]['query_frame'],'keep_iou':float(current_keep[i]),'local_correct':bool(local_correct[i]),'extra_correct':bool(extra_correct[i]),'missing_found':bool(missing_found[i]),'net_oracle_action':int(choice[i]),'net_oracle_iou':float(selected_current[i]),'net_oracle_delta':float(oracle_net[i,choice[i]]),'positive_actions':int(useful[i].sum()),'conflicting_harm_actions':int(conflicting[i].sum())} for i in range(n)]
  result['arms'][arm][part]=d
p=Path('/data/gb/setup/future_policy_actual_label_diagnostic_20261004.json');p.write_text(json.dumps(result,indent=2))
print(json.dumps({'status':result['status'],'observed_cst':result['observed_cst'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'arms':{a:{s:{k:v for k,v in d.items() if k!='query_labels'} for s,d in q.items()} for a,q in result['arms'].items()}}))