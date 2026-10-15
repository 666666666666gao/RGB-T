"""Stdlib control-flow simulation of sealed dense_pipeline.main.

Every external phase is replaced with an in-memory synthetic fixture producer.
No neural execution, imports, subprocesses, real checkpoints, or dataset reads.
Checks orchestration contracts only, not the behavior of mocked phases.
"""
import argparse, ast, copy, datetime, hashlib, json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'dense_m0_fresh_audit_inputs_1435'/'project'
SOURCE=ROOT/'cadtrack_abc/dense_pipeline.py'
checks=[]
traces=[]
def check(name,condition,details=None):
    checks.append(dict(id=len(checks)+1,name=name,passed=bool(condition),details=details))
    assert condition,name

def scenario(variant,fault=None):
    files={}; dirs=set(); commands=[]
    class MemoryPath:
        def __init__(self,p): self.p=PurePosixPath(str(p))
        def __truediv__(self,p): return MemoryPath(self.p/str(p))
        def __str__(self): return str(self.p)
        def resolve(self): return self
        @property
        def parent(self): return MemoryPath(self.p.parent)
        @property
        def stem(self): return self.p.stem
        def mkdir(self,parents=False,exist_ok=False):
            assert exist_ok or str(self) not in dirs
            dirs.add(str(self))
        def read_text(self): return files[str(self)].decode()
        def read_bytes(self): return files[str(self)]
        def write_text(self,t): files[str(self)]=t.encode()
    def write(path,obj): files[path]=json.dumps(obj).encode()
    runroot='/synthetic/'+variant
    fit=runroot+'/fit'
    write('/m0/state.json',dict(status='DENSE_M0_COMPLETE',chosen_common_batch=128))
    write('/m0/exit.json',dict(actual_controller_exit_code=0))
    write('/config.json',dict(test_root='/datasets/lasher'))
    selected=fit+'/trained_best.pth'
    payload=b'SYNTHETIC_CONTROL_FLOW_FIXTURE_NO_MODEL_WEIGHTS'
    digest=hashlib.sha256(payload).hexdigest()
    def option(c,k): return c[c.index(k)+1]
    def fake_run(name,cs,device):
        commands.append(dict(phase=name,commands=cs,device=device))
        if name=='full_training':
            candidates=[dict(epoch=e,checkpoint=fit+f'/candidate_epoch{e:02d}.pth',updates=e*157) for e in (0,1,2,5,10,20)]
            write(fit+'/fit_complete.json',dict(status='DENSE_TRAINING_COMPLETE',probe=False,epochs=20,raw_clips=400000,
                updates=2 if fault=='undertrained' else 3140,query_frame_observations=3200000,
                strict_dense_reload_passed=True,changed_parameter_tensors={'fusion':4,'head':54},
                frozen_backbone_before='synthetic',frozen_backbone_after='synthetic',candidates=candidates))
        elif name=='select_trained_checkpoint':
            files[selected]=payload
            write(fit+'/selection.json',dict(full98_development_selection_complete=True,strict_dense_reload=True,
                trained_best_checkpoint=selected,trained_selected=dict(epoch=0 if fault=='selected_epoch0' else 2,checkpoint_sha256=digest)))
        elif name.startswith('infer_'):
            dataset=name[6:]; count,frames={'lasher':(245,220703),'rgbt234':(234,116649),'rgbt210':(210,104706)}[dataset]
            for shard,c in enumerate(cs):
                write(option(c,'--output')+f'/complete_shard{shard}.json',dict(completed=True,
                    checkpoint_sha256='mismatch' if fault=='changed_checkpoint' and dataset=='rgbt234' else digest,
                    sequences=(count+1)//2 if shard==0 else count//2,frames=frames//2 if shard==0 else frames-frames//2))
        elif name.startswith('score_'):
            dataset=name[6:]; count,frames={'lasher':(245,220703),'rgbt234':(234,116649),'rgbt210':(210,104706)}[dataset]
            names={'lasher':['PR','NPR','SR'],'rgbt234':['MPR','MSR'],'rgbt210':['PR','SR']}[dataset]
            write(option(cs[0],'--output')+'/qualified_full_report.json',dict(completed=True,checkpoint_sha256=digest,
                sequences=count,frames=frames,overall_metrics_percent={n:1 for n in names},baseline_metrics_percent={n:2 for n in names},
                delta_percentage_points={n:-1 for n in names},plus_two_reached={n:False for n in names}))
    source=ast.parse(SOURCE.read_text(encoding='utf-8'))
    main=copy.deepcopy(next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
    replacement=ast.parse('def run(name, commands, device):\n    fake_run(name, commands, device)\n').body[0]
    main.body=[replacement if isinstance(n,ast.FunctionDef) and n.name=='run' else n for n in main.body]
    argv=['--config','/config.json','--output',runroot,'--m0-root','/m0','--variant',variant,'--gpu',str(2 if variant=='H' else 3),'--baseline-234','/baseline234.json','--baseline-210','/baseline210.json']
    class Parser(argparse.ArgumentParser):
        def parse_args(self): return super().parse_args(argv)
    env=dict(argparse=SimpleNamespace(ArgumentParser=Parser),Path=MemoryPath,datetime=datetime.datetime,timezone=datetime.timezone,
             hashlib=hashlib,json=json,os=SimpleNamespace(getpid=lambda:0),__file__='/project/cadtrack_abc/dense_pipeline.py',
             fake_run=fake_run,print=lambda *a,**k:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),'<sealed dense_pipeline main with synthetic external phases>','exec'),env)
    failure=None
    try:
        env['main']()
    except AssertionError as e:
        failure=type(e).__name__+': '+str(e)
    traces.append(dict(variant=variant,fault=fault,phase_trace=commands,assertion_rejection=failure,
                       synthetic_only=True,no_neural_execution=True))
    if fault:
        check('source rejects '+fault,failure is not None)
        check('source does not complete '+fault,runroot+'/pipeline_complete.json' not in files)
        return
    check(variant+' complete control flow with negative synthetic metrics',failure is None)
    state=json.loads(files[runroot+'/pipeline_complete.json'])
    check(variant+' goal remains false',state['goal_achieved'] is False and state['seven_metric_plus_two'] is False)
    check(variant+'14 serial phase calls',len(commands)==14,len(commands))
    check(variant+'23 child module command specifications',sum(len(x['commands']) for x in commands)==23)
    check(variant+' prescribed full training flags',all(v in commands[0]['commands'][0] for v in ('20','20000','8','128')) and '--probe' not in commands[0]['commands'][0])
    check(variant+' A only when enabled',('--memory' in commands[0]['commands'][0])==(variant=='HA'))
    dev=[x for x in commands if x['phase'].startswith('full98_')]
    check(variant+' all six scheduled snapshot phases',[x['phase'] for x in dev]==['full98_candidate_epoch'+str(e).zfill(2) for e in (0,1,2,5,10,20)])
    check(variant+' each dev snapshot two shards',all(len(x['commands'])==2 and [option(c,'--shard') for c in x['commands']]==['0','1'] for x in dev))
    infer=[x for x in commands if x['phase'].startswith('infer_')]
    check(variant+' same selected checkpoint all six inference commands',all(option(c,'--checkpoint')==selected for x in infer for c in x['commands']))
    check(variant+' all three datasets in order',[x['phase'] for x in infer]==['infer_lasher','infer_rgbt234','infer_rgbt210'])
    check(variant+' maximum two child command specs per GPU phase',max(len(x['commands']) for x in commands if x['device']!='')==2)
    check(variant+' all GPU phases use assigned card',all(x['device']==(2 if variant=='H' else 3) for x in commands if x['device']!=''))
    scoring=[x for x in commands if x['phase'].startswith('score_')]
    check(variant+' scalar LasHeR no baseline rerun spec','--baseline-report' not in scoring[0]['commands'][0])
    check(variant+' RGB references passed to qualified scorer',all('--baseline-report' in x['commands'][0] for x in scoring[1:]))
    check(variant+' no overall epoch0 replacement',state['selected_trained_epoch']==2)

scenario('H'); scenario('HA')
for fault in ('undertrained','selected_epoch0','changed_checkpoint'): scenario('H',fault)

# Exact pair-controller final actual-order assertions on synthetic histories.
pair=ROOT/'cadtrack_abc/results/three_benchmark_closure_20261015/r2_dense_admission_1425/dense_full_pair_driver_remote.py'
t=ast.parse(pair.read_text())
loop=next(n for n in t.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='zip')
for mismatched in (False,True):
    hist=[dict(epoch=e,clips=20000,clip_order_sha256='synthetic'+str(e)) for e in range(1,21)]
    env={'histories':{'H':hist,'HA':copy.deepcopy(hist)}}
    if mismatched: env['histories']['HA'][19]['clip_order_sha256']='different'
    rejected=False
    try:
        exec(compile(ast.Module(body=[copy.deepcopy(loop)],type_ignores=[]),'<sealed final paired-order assertions>','exec'),env)
    except AssertionError:
        rejected=True
    check('pair actual20 order loop mismatch='+str(mismatched),rejected==mismatched)

result=dict(status='PASS',checks_count=len(checks),checks=checks,traces=traces,
    scope='Synthetic in-memory control-flow only. External neural/training/scoring phases mocked. Not a real full pipeline receipt.',
    modified_node='dense_pipeline.main nested run function only; imports omitted; Path/argparse/os/print injected.')
(OUT/'orchestration_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PASS',checks_count=len(checks),scenarios=len(traces),scope=result['scope']),indent=2))
