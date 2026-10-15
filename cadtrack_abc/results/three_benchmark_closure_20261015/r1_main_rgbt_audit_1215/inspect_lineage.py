import ast
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'main_rgbt_audit_inputs_1150'
def j(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
state=j(ROOT/'main_benchmark_state.json')
selection=j(ROOT/'selection_after_formal.json')
obs=j(ROOT/'main_observation_1150.json')
fit=j(ROOT/'fit_complete.json')
plan=j(ROOT/'documentation/plan_snapshot.json')
weight=ROOT/'selected_checkpoint/best.pth'
source={k:{'expected':v,'actual':sha(ROOT/'project/cadtrack_abc'/k)} for k,v in state['source_sha256'].items()}
config=sha(ROOT/'project/cadtrack_abc/config.json')
result={'weight_bytes':weight.stat().st_size,'weight_sha256':sha(weight),'state_sha_matches':sha(weight)==state['checkpoint_sha256'],'observation_sha_matches':sha(weight)==obs['checkpoint_sha256'],'config_sha_matches':config==state['configuration_sha256'],'pinned_source_checks':source,'selected_epoch':selection['selected']['epoch'],'all_candidate_epochs':[v['epoch'] for v in selection['candidates']],'selection_summary_argmax':max(selection['candidates'],key=lambda x:x['sequence_mean_iou'])==selection['selected'],'selection_raw_dev_predictions_present':False,'base_pretrain_bytes_present':False,'fit_updates':fit['updates'],'fit_epochs':fit['epochs'],'seed':fit['training_hyperparameters']['seed'],'controller_phases':{k:{'return_codes':v['return_codes'],'status':v['status'],'started_utc':v['started_utc'],'completed_utc':v['completed_utc'],'jobs':[(x['pid'],x['physical_gpu']) for x in v['jobs']]} for k,v in state['phases'].items()},'controller_observation':{k:v for k,v in obs.items() if k!='reports'},'supplied_source_at_launch_omissions':[]}
for name in ['search_actions.py','counterfactual.py','ported_modules.py','vendor/CADTrack/lib/models/cadtrack/vit_care.py','vendor/CADTrack/lib/models/layers/head.py','vendor/CADTrack/experiments/cadtrack/cadtrack.yaml']:
    result['supplied_source_at_launch_omissions'].append({'file':name,'collection_sha256':sha(ROOT/'project/cadtrack_abc'/name),'pinned_at_launch':name in state['source_sha256']})
prior_path=OUT.parent/'r1_main_lasher_formal_audit_20261015/EXPERIMENT_AUDIT.json'
prior=j(prior_path)
result['prior_lasher_audit']={'path':str(prior_path),'sha256':sha(prior_path),'verdict':prior.get('verdict'),'overall_verdict':prior.get('overall_verdict'),'claims':prior.get('claims'),'scope':'Separate audit artifact; not used as a substitute for new raw RGBT recomputation.'}
(OUT/'lineage_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
for key in ['current_acceptance_20261015','r1_matched_pair_20261015','official_rgbt210_completion_20261015']:
    item=plan[key]
    print('PLAN',key,'keys',list(item))
    for k,v in item.items():
        if isinstance(v,(str,bool,int,float)) or k in ['targets','target_metrics','acceptance','selected_checkpoint','main_selected','control_selected']:
            print(k,json.dumps(v,ensure_ascii=False))
path=ROOT/'documentation/HANDOFF_snapshot.md'
lines=path.read_text(encoding='utf-8').splitlines()
print('HANDOFF total_lines',len(lines))
for n in range(max(0,len(lines)-45),len(lines)):
    print(f'{n+1}: {lines[n]}')
for relative in ['project/cadtrack_abc/score.py','project/cadtrack_abc/infer.py','project/cadtrack_abc/infer_baseline.py','project/cadtrack_abc/validate_future_actions.py','project/cadtrack_abc/select_future_actions.py','project/cadtrack_abc/model.py','project/cadtrack_abc/counterfactual.py','project/cadtrack_abc/future_actions.py','project/cadtrack_abc/search_actions.py','project/cadtrack_abc/baseline_pipeline.py','project/cadtrack_abc/pipeline.py']:
    tree=ast.parse((ROOT/relative).read_text(encoding='utf-8-sig'))
    print('DEFINITIONS',relative,[(n.name,n.lineno,n.end_lineno) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))])
