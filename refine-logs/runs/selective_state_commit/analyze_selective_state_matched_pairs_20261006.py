"""Separate actual same-candidate write and geometry contrasts in closed labels."""
import json
from collections import Counter
from pathlib import Path

base=Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/selective_state_commit')
source=base/'training_internal_complete/collection/collection_complete.json'
records=json.loads(source.read_text())['results']
rows=[]
for record in records:
 actions=set(record['legal_actions'])
 for slot in range(3):
  for contrast,left,right in [('pause_vs_regular',slot*3+1,slot*3),
                              ('geometry_C_vs_candidate_with_both_paused',slot*3+2,slot*3+1),
                              ('geometry_C_vs_candidate_with_both_nonwriting',slot*3+2,slot*3)]:
   if contrast=='geometry_C_vs_candidate_with_both_nonwriting' and slot*3+1 in actions:
    continue  # Pause legal means raw>.84; regular then writes and would confound geometry.
   if left not in actions or right not in actions:
    continue
   for horizon in (3,32):
    outcomes=record['outcomes']
    key=str(horizon)
    if key not in outcomes[str(left)]:
     continue  # The two actual short videos supply H3 only.
    a,b=outcomes[str(left)][key],outcomes[str(right)][key]
    assert a['valid_GT_frames']==b['valid_GT_frames'] and a['unknown_GT_frames']==b['unknown_GT_frames']
    for penalty in (0.,.1):
     utility=(a['mean_iou_including_query']-b['mean_iou_including_query']
              -penalty*(a['failure_frames_including_query']-b['failure_frames_including_query'])/a['valid_GT_frames'])
     rows.append({'partition':record['partition'],'event':record['actual_event_id'],
      'sequence':record['sequence'],'query_frame':record['query_frame'],'role':record['actual_event_role'],
      'contrast':contrast,'slot':slot,'horizon':horizon,'lost_penalty':penalty,
      'utility_delta':utility,'future_mean_iou_delta':a['mean_future_iou']-b['mean_future_iou'],
      'unknown_GT_frames':a['unknown_GT_frames']})

def summary(group):
 counts=Counter(row['event'] for row in group)
 weights=[1/counts[row['event']] for row in group]
 total=sum(weights)
 values=[row['utility_delta'] for row in group]
 return {'matched_action_pairs':len(group),'distinct_actual_events':len(counts),
  'distinct_sequence_queries':len({(r['sequence'],r['query_frame']) for r in group}),
  'positive_pairs':sum(v>1e-8 for v in values),'negative_pairs':sum(v<-1e-8 for v in values),
  'above_point03_pairs':sum(v>.03 for v in values),
  'event_weighted_utility_delta':sum(w*v for w,v in zip(weights,values))/total,
  'event_weighted_future_mean_iou_delta':sum(w*r['future_mean_iou_delta'] for w,r in zip(weights,group))/total,
  'pairs_with_unknown_GT_in_window':sum(r['unknown_GT_frames']>0 for r in group)}

groups={}
for row in rows:
 key=(row['partition'],row['contrast'],row['horizon'],row['lost_penalty'])
 groups.setdefault(key,[]).append(row)
reports=[dict(zip(('partition','contrast','horizon','lost_penalty'),key),**summary(group))
         for key,group in sorted(groups.items())]
unrecovered={}
for partition in ('train','validation'):
 group=[r for r in records if r['partition']==partition and r['actual_event_role'].startswith('unrecovered')]
 with_unknown=[r for r in group if r['outcomes'][str(r['legal_actions'][0])][str(r['actual_rollout_horizon'])]['unknown_GT_frames']>0]
 unrecovered[partition]={'unrecovered_window_queries':len(group),'with_unknown_GT_frames_in_window':len(with_unknown),
  'definition':'No observed future three consecutive known frames IoU>=0.5 within the available H3/H32 window; not proof of never recoverable.'}
result={'status':'ACTUAL_MATCHED_WRITE_GEOMETRY_CONTRASTS_COMPLETE','reports':reports,
 'unrecovered_role_scope':unrecovered,'pair_weighting':'Inverse pair count per actual event within each reported group; equal event mass',
 'scope':'Same actual candidate and prestate. Pause-versus-regular changes appearance/memory commit. Mode2-versus-mode1 exists only if both legal. When mode1 illegal, mode2-versus-mode0 compares search reference plus current motion-history entry including box and quality: raw<=.84 makes both query writes false and the appearance-pause branch inactive. This is not a pure coordinate-only contrast. All later decisions use frozen parent, not deployed learned extension.',
 'limitation':'These are existing TRAIN/developer action-label contrasts, not online policy efficacy or full causal explanation of the learned decline. Pair count is not independent sample count.',
 'new_NN_forwards':0,'new_optimizer_updates':0,'active_process_GPU_queries':0}
target=base/'matched_write_geometry_CPU_contrasts_v2.json'
assert not target.exists()
target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
