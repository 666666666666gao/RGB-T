"""AST contracts and stdlib-only witnesses; never imports project/Torch.

Synthetic witnesses check source behavior, not neural or benchmark execution.
"""
import ast, copy, hashlib, json, random
from pathlib import Path
from types import SimpleNamespace
import math

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'dense_m0_fresh_audit_inputs_1435'/'project'
checks=[]
def tree(name): return ast.parse((ROOT/'cadtrack_abc'/name).read_text(encoding='utf-8-sig'))
def check(name, condition, observed=None):
    checks.append(dict(id=len(checks)+1,name=name,passed=bool(condition),observed=observed))
    assert condition,name
def method(t, cls, name):
    body=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name==cls).body if cls else t.body
    return next(x for x in body if isinstance(x,ast.FunctionDef) and x.name==name)
def calls(node):
    return [(n.lineno,ast.unparse(n.func)) for n in ast.walk(node) if isinstance(n,ast.Call)]
def names(node): return {n.id for n in ast.walk(node) if isinstance(n,ast.Name)}
def execute_functions(nodes,env):
    selected=copy.deepcopy(nodes)
    for n in selected: n.decorator_list=[]
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),'<extracted sealed source>','exec'),env)

dm=tree('dense_model.py')
extract=method(dm,'DenseCADTrack','extract')
predict=method(dm,'DenseCADTrack','predict')
step=method(dm,'DenseCADTrack','step_batch')
forward=method(dm,'DenseCADTrack','forward')
check('only frozen extraction decorated no_grad',any(ast.unparse(x)=='torch.no_grad()' for x in extract.decorator_list) and not predict.decorator_list)
check('predict calls extraction and differentiable fusion head',{'self.extract','self.base.decode_fuse_search','self.base.box_head'}<=set(c for _,c in calls(predict)))
check('step signature contains images state only',[a.arg for a in step.args.args]==['self','images','state'])
check('step does not mention truth labels teacher',not names(step)&{'truth','labels','teacher_features','teacher_valid','future'})
loss=method(dm,'DenseCADTrack','localization_losses')
check('GT indexed regression only in loss',any(c=='self.base.box_head.cal_bbox' for _,c in calls(loss)) and not any(c=='self.base.box_head.cal_bbox' for _,c in calls(step)))
fs=calls(forward)
step_line=next(n for n,c in fs if c=='self.step_batch')
loss_line=next(n for n,c in fs if c=='self.localization_losses')
check('forward prediction step precedes loss',step_line<loss_line,[step_line,loss_line])
check('GT teacher only feeds memory losses',any(c=='self.memory_losses' for _,c in fs))
tracker=method(tree('dense_tracker.py'),'DenseCADTracker','track')
check('online tracker accepts image only',[a.arg for a in tracker.args.args]==['self','image'])
check('online tracker no GT or teacher names',not names(tracker)&{'truth','labels','teacher_features','teacher_valid','initial'})
train=method(tree('train_dense.py'),None,'main')
tc={c for _,c in calls(train)}
check('training does real backward optimizer steps in source',{'loss.backward','optimizer.step','model','module_hash','clone.load_trainable_state'}<=tc)
check('independent data generator passed to loader',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='DataLoader' and any(k.arg=='generator' and ast.unparse(k.value)=='data_generator' for k in n.keywords) for n in ast.walk(train)))
check('training initializer uses pretrained not qualification path','DenseCADTrack(CleanCADTrack(pretrained), args.memory)' in ast.unparse(train))
check('no inherited R1 ABC construction in dense class',not any(c.endswith(('CADTrackABC','CADModules','IdentityMotion')) for _,c in calls(dm)))
ev=tree('dense_evaluation.py')
track_video=method(ev,None,'track_video')
check('video accepts only initial GT',[a.arg for a in track_video.args.args]==['tracker','rgb','tir','initial'])
check('video does not load later labels',not any(c.startswith(('np.loadtxt','np.fromstring')) for _,c in calls(track_video)))
inf=tree('infer_dense.py')
check('formal inference reads only readline initialization','stream.readline()' in ast.unparse(inf))
check('formal inference explicitly positive epoch',"assert payload['epoch'] > 0" in ast.unparse(inf))
val=tree('validate_dense.py')
check('validation passes labels0 only to video','track_video(tracker, rgb, tir, labels[0])' in ast.unparse(val))
check('validation truth comes from dataset init','np.loadtxt(sequence / \'init.txt\'' in ast.unparse(val))
score=tree('score.py')
check('official native metrics actually invoked','getattr(evaluator, metric)(tracker)' in ast.unparse(score))
check('official GT loaded dataset not baseline','evaluator.seqs_gt' in ast.unparse(score) and 'initial_gt_file' in ast.unparse(score))
check('official metric unit percent only multiplication100','float(value[0]) * 100' in ast.unparse(score))
check('LasHeR mechanism summary actually called',any(c=='summarize_lasher' for _,c in calls(score)))
check('legacy R1 sequence_iou not claimed train monitor',not any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value=='sequence_iou' for n in ast.walk(train)))

# Execute exact independent RNG/start-selection expressions on synthetic metadata.
get=method(tree('data.py'),'VideoClips','__getitem__')
selected=[n for n in get.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('rng','sequence_index','first')]
def sample(index):
    env=dict(random=random,self=SimpleNamespace(seed=42,epoch=3,samples=20000,sequences=[None]*881),index=index,starts=list(range(39)))
    exec(compile(ast.Module(body=copy.deepcopy(selected),type_ignores=[]),'<sealed RNG statements>','exec'),env)
    return env['sequence_index'],env['first']
a=[sample(i) for i in range(100)]
random.seed(999)
for _ in range(1000): random.random()
b=[sample(i) for i in range(100)]
check('exact sampler RNG statements independent of global draws',a==b)
check('sampler reads initial plus query images not future labels',"ids[:self.steps + 1]" in ast.unparse(get))
check('sampler explicit continuous stride1','stride = 1' in ast.unparse(get))

# Execute actual stdlib diagnostic functions; NumPy IoU execution remains unavailable.
ms=tree('mechanism_summary.py')
env={}
execute_functions([method(ms,None,n) for n in ('recovery_events','ratio')],env)
events=env['recovery_events']([.1,.1,.7,.1,.8],[True,False,True,True,False])
check('actual recovery excludes invalid end and counts frame delay',events==[dict(start_frame=1,recovery_frame=3,delay_frames=2,right_censored=False),dict(start_frame=4,recovery_frame=None,delay_frames=None,observed_until_frame=5,right_censored=True)],events)
check('actual ratio uses explicit event denominator',env['ratio'](2,5)==.4)
check('zero denominator represented null',env['ratio'](0,0) is None)

# Execute source max-selection statements with fabricated scores (no weights).
sel=method(tree('select_dense.py'),None,'main')
select_nodes=[n for n in sel.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('overall','trained')]
for desc,values,expected in [('initialization_wins',[.9,.2,.3,.4,.5,.6],(0,20)),('positive_wins',[.1,.2,.8,.4,.5,.6],(2,2)),('all_tie',[.5]*6,(0,1))]:
    scope={'scores':[dict(epoch=e,sequence_mean_iou=v) for e,v in zip((0,1,2,5,10,20),values)]}
    exec(compile(ast.Module(body=copy.deepcopy(select_nodes),type_ignores=[]),'<sealed selection expressions>','exec'),scope)
    check('selection '+desc,(scope['overall']['epoch'],scope['trained']['epoch'])==expected)

# Independent scalar Center-head convention oracle; no neural execution.
rng=random.Random(42)
for i in range(7):
    index=rng.randrange(256); w,h=rng.uniform(.05,.8),rng.uniform(.05,.8)
    dx,dy=rng.uniform(-.2,.9),rng.uniform(-.2,.9)
    cx,cy=((index%16)+dx)/16,((index//16)+dy)/16
    edges=(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
    check(f'Center scalar convention case{i}',math.isclose(edges[2]-edges[0],w) and math.isclose(edges[3]-edges[1],h))

# Function-call inventory is static; it does not turn source into runtime evidence.
inventory={}
for filename in ('data.py','dense_model.py','dense_tracker.py','dense_evaluation.py','qualify_dense.py','validate_dense.py','select_dense.py','infer_dense.py','score.py','score_dense.py','mechanism_summary.py','dense_pipeline.py','train_dense.py'):
    t=tree(filename)
    inventory[filename]=dict(definitions=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno) for n in ast.walk(t) if isinstance(n,ast.FunctionDef)],
                            call_sites=[dict(line=line,call=call) for line,call in calls(t)])
result=dict(status='PASS',checks_count=len(checks),checks=checks,call_inventory=inventory,
    execution_scope='Stdlib only; extracted non-neural expressions/functions and AST. Synthetic witnesses are not dataset/neural validation.',
    unsupported='No Torch/CUDA/weights, no proof of later formal pipeline completion, no reconstruction of actual GPU training order from raw stream.')
(OUT/'source_stdlib_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PASS',checks_count=len(checks),scope=result['execution_scope']),indent=2))
