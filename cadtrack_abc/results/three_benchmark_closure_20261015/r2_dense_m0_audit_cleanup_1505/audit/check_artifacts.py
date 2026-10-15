"""Independent read-only stdlib audit of sealed artifacts; no project imports."""
import ast, datetime, hashlib, json, math, random
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'dense_m0_fresh_audit_inputs_1435'/'project'
BASE='cadtrack_abc/results/three_benchmark_closure_20261015/'
ADM=BASE+'r2_dense_admission_1425/'
checks=[]
facts={}
def read(rel): return json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))
def sha(rel): return hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
def check(name, condition, evidence=None):
    checks.append(dict(id=len(checks)+1,name=name,passed=bool(condition),evidence=evidence))
    assert condition, name
def same(name,a,b): check(name,a==b,dict(observed=a,expected=b))
def close(name,a,b): check(name,math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),dict(observed=a,expected=b))

manifest=json.loads((ROOT.parent/'input_manifest.json').read_text(encoding='utf-8-sig'))
inventory=[]
for item in manifest['files']:
    path=ROOT/item['relative']; raw=path.read_bytes(); source=raw.decode('utf-8-sig')
    check('sealed bytes '+item['relative'],len(raw)==item['bytes'])
    same('sealed hash '+item['relative'],hashlib.sha256(raw).hexdigest(),item['sha256'])
    kind='text'
    if path.suffix=='.py':
        tree=ast.parse(source,filename=item['relative']); kind='python_ast'
        check('AST parsed '+item['relative'],isinstance(tree,ast.Module))
    if path.suffix=='.json':
        value=json.loads(source); kind='json'
        def finite(v):
            if isinstance(v,float): return math.isfinite(v)
            if isinstance(v,dict): return all(finite(x) for x in v.values())
            if isinstance(v,list): return all(finite(x) for x in v)
            return True
        check('all JSON floats finite '+item['relative'],finite(value))
    inventory.append(dict(relative=item['relative'],bytes=len(raw),lines=len(source.splitlines()),kind=kind))
facts['inventory']=inventory
same('manifest declared count',len(inventory),85)

admission=read(ADM+'admission.json')
states={k:read(ADM+k+'/state.json') for k in ('batch32_64','batch128')}
probe_rows=[]
full_manifests=[]
raw_receipts=[]
for batch in (32,64,128):
    folder='batch128' if batch==128 else 'batch32_64'
    pair=[]
    for variant in ('H','HA'):
        rel=ADM+folder+f'/{variant}_batch{batch}/'
        f,h,p,m=[read(rel+name+'.json') for name in ('fit_complete','history','progress','manifest')]
        full_manifests.append(m)
        raw_receipts.append(dict(path=rel,fit=f,history=h,progress=p,
            manifest={k:v for k,v in m.items() if k not in ('train_sequences','validation_sequences')}))
        tag=f'{variant} batch{batch}'
        same(tag+' qualification status',f['status'],'DENSE_QUALIFICATION_COMPLETE')
        same(tag+' one probe epoch',f['epochs'],1)
        same(tag+' two optimizer updates',f['updates'],2)
        same(tag+' raw clips',f['raw_clips'],batch*2)
        same(tag+' query observations',f['query_frame_observations'],batch*2*8)
        same(tag+' probe flag',f['probe'],True)
        same(tag+' no test metrics',f['formal_test_metrics'],None)
        same(tag+' no full98 selection',f['full98_selection_complete'],False)
        same(tag+' candidate epochs',[x['epoch'] for x in f['candidates']],[0,1])
        same(tag+' candidate updates',[x['updates'] for x in f['candidates']],[0,2])
        same(tag+' recorded freeze equality',f['frozen_backbone_before'],f['frozen_backbone_after'])
        same(tag+' recorded freeze shared',f['frozen_backbone_before'],admission['frozen_backbone_sha256'])
        same(tag+' recorded reload',f['strict_dense_reload_passed'],True)
        same(tag+' recorded change counts',f['changed_parameter_tensors'],dict(fusion=4,head=54,**({'A':11} if variant=='HA' else {})))
        same(tag+' history epochs',len(h),1)
        same(tag+' history clips',h[0]['clips'],f['raw_clips'])
        same(tag+' history updates',h[0]['updates'],f['updates'])
        same(tag+' progress updates',p['updates'],f['updates'])
        same(tag+' progress clips',p['clips_seen'],f['raw_clips'])
        same(tag+' progress batches',p['batch'],2)
        same(tag+' gradient groups',set(p['gradients']),{'fusion','head','A'} if variant=='HA' else {'fusion','head'})
        check(tag+' all recorded gradients positive finite',all(math.isfinite(v) and v>0 for v in p['gradients'].values()))
        close(tag+' progress throughput identity',p['clips_per_second']*p['seconds_per_update'],batch)
        check(tag+' epoch end timing after last progress',h[0]['epoch_seconds']>=2*p['seconds_per_update'])
        q=admission['measured_batches'][str(batch)][variant]
        throughput=2*batch/h[0]['epoch_seconds']
        close(tag+' admission throughput recomputed',q['clips_per_second'],throughput)
        same(tag+' admission peak copied',q['peak_allocated_gib'],p['peak_allocated_gib'])
        same(tag+' admission gradients copied',q['gradients'],p['gradients'])
        same(tag+' admission order copied',q['clip_order_sha256'],h[0]['clip_order_sha256'])
        same(tag+' config batch',m['config']['batch'],batch)
        same(tag+' model seed',m['config']['seed'],42)
        same(tag+' independent data seed',m['config']['data_generator_seed'],42)
        same(tag+' configured probe',m['config']['probe'],True)
        same(tag+' memory flag',m['config']['use_memory'],variant=='HA')
        same(tag+' loaded tensor count receipt',m['pretrained_strict_tensors'],250)
        same(tag+' loaded epoch receipt',m['pretrained_epoch'],14)
        same(tag+' pretrain receipt',m['config']['pretrained_sha256'],f['pretrained_sha256'])
        same(tag+' listed trainable parameter sum',sum(m['trainable_parameters'].values()),admission['trained_parameters'][variant])
        same(tag+' train sequence uniqueness',len(set(m['train_sequences'])),881)
        same(tag+' dev sequence uniqueness',len(set(m['validation_sequences'])),98)
        check(tag+' train dev disjoint',not set(m['train_sequences'])&set(m['validation_sequences']))
        all_names=sorted(m['train_sequences']+m['validation_sequences'])
        same(tag+' split recomputed with Python seed',sorted(random.Random(20261013).sample(all_names,98)),m['validation_sequences'])
        group_counts={key:sum(n.startswith(prefix) for n in m['trainable_parameters']) for key,prefix in [('fusion','base.decode_fuse_search.'),('head','base.box_head.'),('A','prompt.')]}
        same(tag+' no backbone listed trainable',sum(n.startswith('base.backbone.') for n in m['trainable_parameters']),0)
        same(tag+' parameter tensor change count supported by names',group_counts,dict(fusion=4,head=54,A=11 if variant=='HA' else 0))
        row=dict(variant=variant,batch=batch,updates=f['updates'],raw_clips=f['raw_clips'],query_frames=f['query_frame_observations'],
                 epoch_seconds=h[0]['epoch_seconds'],clips_per_second=throughput,peak_allocated_gib=p['peak_allocated_gib'],
                 allocation_fraction_of_reported_24GiB=p['peak_allocated_gib']/24,order_hash=h[0]['clip_order_sha256'],
                 trainable_parameter_count=sum(m['trainable_parameters'].values()),gradient_groups=p['gradients'])
        pair.append(row); probe_rows.append(row)
    same(f'batch{batch} actual paired recorded order',pair[0]['order_hash'],pair[1]['order_hash'])
    gmean=math.sqrt(pair[0]['clips_per_second']*pair[1]['clips_per_second'])
    close(f'batch{batch} geometric mean',gmean,admission['measured_batches'][str(batch)]['geometric_mean_clips_per_second'])
for index,m in enumerate(full_manifests[1:],1):
    same(f'manifest{index} same train identities',m['train_sequences'],full_manifests[0]['train_sequences'])
    same(f'manifest{index} same validation identities',m['validation_sequences'],full_manifests[0]['validation_sequences'])
facts['probe_rows']=probe_rows
facts['raw_receipts_and_manifest_fields']=raw_receipts
facts['sequence_inventory']=dict(train=full_manifests[0]['train_sequences'],validation=full_manifests[0]['validation_sequences'])
same('total actual qualification optimizer updates',sum(r['updates'] for r in probe_rows),12)
same('admission total updates',admission['actual_qualification_optimizer_updates'],12)
facts['qualification_total_clips']=sum(r['raw_clips'] for r in probe_rows)
facts['qualification_total_query_observations']=sum(r['query_frames'] for r in probe_rows)
same('H parameter sum independently from layer dimensions',1536*768*9+768+2*768 + 3*((768*256*9+3*256)+(256*128*9+3*128)+(128*64*9+3*64)+(64*32*9+3*32)) + (32+1)*(1+2+2),17093253)
same('A parameter sum independently from layer dimensions',2*768+768*128+128+(128+4)*64+64+64*3+3+2*2*128+(128*3+3)*768+768,407171)
same('HA minus H parameters',admission['trained_parameters']['HA']-admission['trained_parameters']['H'],407171)
gm={b:admission['measured_batches'][str(b)]['geometric_mean_clips_per_second'] for b in (32,64,128)}
same('chosen fastest common short-probe batch',max(gm,key=gm.get),128)
facts['short_probe_relative_gmean_gain']=dict(batch128_over64=gm[128]/gm[64]-1,batch128_over32=gm[128]/gm[32]-1)

parity=read(ADM+'batch32_64/epoch0_parity.json')
same('parity24 rows',len(parity['records']),24)
same('parity declared frames',parity['actual_causal_query_frames'],24)
same('parity exactly2 train sequences',len({r['sequence'] for r in parity['records']}),2)
for i,r in enumerate(parity['records']):
    same(f'parity{i} H HA errors',r['maximum_box_errors'],[0.,0.])
    same(f'parity{i} exact outputs',r['exact_outputs_H_HA'],[True,True])
    same(f'parity{i} template values strides',r['template_values_and_strides_equal_H_HA'],[True,True])
for seq in sorted({r['sequence'] for r in parity['records']}):
    rs=[r for r in parity['records'] if r['sequence']==seq]
    same(seq+'12 query frames',len(rs),12)
    same(seq+' contiguous parity frame ids',[r['frame'] for r in rs],list(range(rs[0]['frame'],rs[0]['frame']+12)))
    check(seq+' is TRAIN',seq in full_manifests[0]['train_sequences'])
facts['parity']=dict(sequences=sorted({r['sequence'] for r in parity['records']}),frames=24,
    pure_template_update_events=sum(r['template_updated'] for r in parity['records']),records=parity['records'])
source=read(BASE+'r2_dense_dependency_fix_1405/source_manifest.json')
for rel,want in source['sources'].items(): same('M0 sealed source '+rel,sha(rel),want)
for group,state in states.items():
    same(group+' successful state',state['status'],'DENSE_M0_COMPLETE')
    same(group+' wrapper exit',read(ADM+group+'/exit.json')['actual_controller_exit_code'],0)
    same(group+' source manifest link',state['source_manifest_sha256'],sha(BASE+'r2_dense_dependency_fix_1405/source_manifest.json'))
    for phase,rec in state['phases'].items(): same(group+' actual phase exit '+phase,rec['actual_exit_code'],0)
same('batch128 base state link',states['batch128']['base_state_sha256'],sha(ADM+'batch32_64/state.json'))
same('admission orchestration hash',admission['orchestration_sha256'],sha('cadtrack_abc/dense_pipeline.py'))
launch=read(BASE+'r2_dense_full_launch_1430/launch_verified.json')
same('full launch qualification state link',launch['preflight']['qualification_state_sha256'],sha(ADM+'batch128/state.json'))
same('full launch22 source files',len(launch['preflight']['expected_sources']),22)
for rel,want in launch['preflight']['expected_sources'].items(): same('launch sealed source '+rel,sha(rel),want)
expected_nodes=ast.parse((ROOT/BASE/'r2_dense_full_launch_1430/executed_remote_launcher.py').read_text())
assigns={n.targets[0].id:n.value for n in expected_nodes.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
same('executed launcher expected source map',ast.literal_eval(assigns['expected']),launch['preflight']['expected_sources'])
payload=ast.literal_eval(assigns['payload'])
for name,target in [('driver.py','dense_full_pair_driver_remote.py'),('wrapper.py','dense_full_pair_wrapper_remote.py')]:
    same('embedded launch payload '+name,payload[name],(ROOT/ADM/target).read_text())
check('training command no probe',all('--probe' not in j['actual_command'] for j in launch['actual_neural_jobs']))
same('two recorded launch jobs',len(launch['actual_neural_jobs']),2)
same('launch physical GPU labels',[j['physical_gpu'] for j in launch['actual_neural_jobs']],[2,3])
check('launch memory only HA','--memory' not in launch['actual_neural_jobs'][0]['actual_command'] and '--memory' in launch['actual_neural_jobs'][1]['actual_command'])
first=read(BASE+'r2_dense_dependency_fix_1405/first_attempt_observation.json')
same('retained first failed controller exit',first['wrapper_exit']['actual_controller_exit_code'],1)
same('retained first failed parity exit',first['state']['phases']['epoch0_parity']['actual_exit_code'],1)
check('retained actual missing dependency stderr',any('ModuleNotFoundError' in s and 'lib.utils.tensor' in s for s in first['phase_log_tails']['epoch0_parity']))
events=[first['state']['started_utc'],first['wrapper_exit']['completed_utc'],states['batch32_64']['started_utc'],states['batch32_64']['completed_utc'],states['batch128']['started_utc'],states['batch128']['completed_utc'],launch['preflight']['observed_utc'],launch['started_utc'],launch['verified_utc']]
dt=[datetime.datetime.fromisoformat(t) for t in events]
same('chronology ordered first fail retry capacity full launch',dt,sorted(dt))
facts['chronology_utc']=events
plan=read(BASE+'r2_dense_plan_20261015.json')
same('formal planned clips',20*20000,400000)
same('formal planned updates per epoch',math.ceil(20000/128),157)
same('formal planned total updates',20*math.ceil(20000/128),3140)
same('formal planned query observations',20*20000*8,3200000)
same('formal planned validation frames',6*52200,plan['compute']['frame_pair_budgets']['six_snapshot_frames_per_variant'])
same('formal planned official frames',220703+116649+104706,plan['compute']['frame_pair_budgets']['official_frames_per_variant'])
facts['formal_results_present']=[i['relative'] for i in manifest['files'] if any(s in i['relative'] for s in ('qualified_full_report.json','selection.json','pipeline_complete.json','pair_complete.json'))]
same('no dense formal result or selection artifacts in sealed input',facts['formal_results_present'],[])
facts['stale_plan_purpose']=plan['purpose']
facts['plan_status']=plan['status']
facts['limitations']=['weights, images, GT, raw training order streams and complete process logs absent',
    'runtime receipts checked for consistency, not independently re-executed or cryptographically attested',
    'manifest enumerates trainable parameter sizes; frozen backbone parameter count has no per-tensor inventory',
    'external toolkit, baseline RGB reports and transitive native dependencies incomplete in input',
    'historical narrative references outside current R2/M0/launch scope are not re-certified']
report=dict(status='PASS',checker='check_artifacts.py',checks_count=len(checks),checks=checks,facts=facts)
(OUT/'artifact_checks.json').write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PASS',checks_count=len(checks),probe_rows=probe_rows,
    qualification_clips=facts['qualification_total_clips'],qualification_query_observations=facts['qualification_total_query_observations'],
    gains=facts['short_probe_relative_gmean_gain'],parity_template_updates=facts['parity']['pure_template_update_events']),indent=2))
