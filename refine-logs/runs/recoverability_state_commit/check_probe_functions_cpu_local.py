"""Read-only CPU check of the actual Branch and new isolated probe helpers."""
import ast
import json
from pathlib import Path
import subprocess
root=Path('.')
text=(root/'research/probe_geometry_commit.py').read_text(encoding='utf-8')
tree=ast.parse(text)
functions={n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
text=(root/'research/bounded_recovery.py').read_text(encoding='utf-8')
branch=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='Branch')
code='import copy,json,datetime,math\nfrom collections import deque\nfrom dataclasses import dataclass\nfrom types import SimpleNamespace\nimport numpy as np\nimport torch\ntorch.set_num_threads(1)\n'+ast.unparse(branch)+'\n'+functions['private_state']+'\n'+functions['outcome']+"""
b=np.array([10.,20.,30.,40.])
t=torch.ones(1,6,112,112)
m=torch.ones(1,1,64,dtype=torch.bool)
branch=Branch(0,0,b.copy(),b.copy(),t,m,t,m,deque([(0,t,m)],maxlen=8),deque([.8],maxlen=8),deque([b.copy()],maxlen=8))
parent=SimpleNamespace(branch=branch,branches=[branch],stats={'writes':1},history=deque([(b.copy(),4,.8)],maxlen=8),identity_memory=torch.ones(1,4,2,128),motion_memory=torch.ones(1,4,2,128),template_source_box=b.copy())
shadow=private_state(parent)
assert shadow.branch.template is parent.branch.template
shadow.branch.box[0]=999
shadow.branch.search_box[1]=999
shadow.branch.pending.clear()
shadow.branch.boxes.append(np.zeros(4))
shadow.history[0][0][2]=999
shadow.identity_memory[0,0,0,0]=999
shadow.motion_memory[0,0,0,0]=999
shadow.template_source_box[3]=999
shadow.stats['writes']=999
assert np.array_equal(parent.branch.box,b) and np.array_equal(parent.branch.search_box,b)
assert len(parent.branch.pending)==1 and len(parent.branch.boxes)==1
assert np.array_equal(parent.history[0][0],b) and len(parent.history)==1
assert parent.identity_memory[0,0,0,0].item()==parent.motion_memory[0,0,0,0].item()==1
assert np.array_equal(parent.template_source_box,b) and parent.stats['writes']==1
values=np.array([.1,.1,.6,.7,.8,.1]);writes=np.array([True,False,False,True,False,True])
a=outcome(values,writes)
assert math.isclose(a['mean_iou_including_query'],.4) and math.isclose(a['mean_future_iou'],.46)
assert a['failure_frames_including_query']==3 and a['longest_failure_run']==2
assert a['first_future_three_correct_run_offset']==2 and a['writes']==3 and a['wrong_localization_writes']==2
assert outcome(values[:4],writes[:4])['first_future_three_correct_run_offset'] is None
print(json.dumps({'status':'PASS_CPU_STATE_COPY_AND_DELAYED_OUTCOME_CHECK','computed_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'actual_Branch_class_used':True,'mutable_state_copy_isolation':True,'shared_immutable_template_preserved':True,'delayed_three_frame_reacquisition_metric_checked':True,'GPU_queries':0,'NN_calls':0,'optimizer_updates':0,'real_video_runtime_attested':False}))
"""
p=subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'],input=code,capture_output=True,text=True)
assert p.returncode==0,p.stderr
actual=json.loads(p.stdout)
(root/'refine-logs/runs/recoverability_state_commit/actual_probe_cpu_function_checks.json').write_text(json.dumps(actual,indent=2)+'\n',encoding='utf-8')
print(json.dumps(actual))
