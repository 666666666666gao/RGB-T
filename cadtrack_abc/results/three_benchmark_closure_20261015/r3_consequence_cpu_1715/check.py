import io,json
from pathlib import Path
import numpy as np
import torch
from cadtrack_abc.dense_submission import DenseDecisionHeads
from cadtrack_abc.train_dense_consequences import evaluate_batch
from cadtrack_abc.dense_consequences import label_native_traces
assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.manual_seed(42)
records=[]
for row in range(2):
    contexts=[]
    for region,count in enumerate((5,10,10,10)):
        utility=torch.full((count,6),.4+.05*region)
        utility[1,0]+=.3
        future=torch.full_like(utility,.6)
        future[1,0]=.2
        lost=torch.zeros_like(utility)
        lost[1,0]=.7
        contexts.append(dict(features=torch.randn(1,count,275),utility=utility,
            future_mean_iou=future,future_lost_fraction=lost,current_iou=torch.full_like(utility,.5)))
    records.append(dict(contexts=contexts,search_input=torch.randn(1,295)))
model=DenseDecisionHeads()
before={name:value.detach().clone() for name,value in model.state_dict().items()}
loss,initial=evaluate_batch(model,records,torch.device('cpu'))
assert torch.isfinite(loss)
assert torch.equal(initial['delta_utility'],torch.zeros(2))
assert torch.equal(initial['extra_search'],torch.zeros(2))
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
gradients=[]
for step in range(2):
    optimizer.zero_grad(set_to_none=True)
    value,diagnostics=evaluate_batch(model,records,torch.device('cpu'))
    assert torch.isfinite(value) and all(torch.isfinite(t).all() for t in diagnostics.values())
    value.backward()
    grouped={group:float(sum((v.grad.square().sum() for name,v in model.named_parameters()
        if name.startswith(group) and v.grad is not None),torch.zeros(())).sqrt())
        for group in ('future_action.','search_action.')}
    assert all(np.isfinite(v) and v>0 for v in grouped.values())
    gradients.append(grouped)
    optimizer.step()
changed={group:sum(not torch.equal(before[name],value) for name,value in model.state_dict().items()
    if name.startswith(group)) for group in ('future_action.','search_action.')}
assert all(changed.values())
stream=io.BytesIO()
torch.save(model.state_dict(),stream)
stream.seek(0)
reloaded=DenseDecisionHeads()
reloaded.load_state_dict(torch.load(stream,map_location='cpu',weights_only=True),strict=True)
assert all(torch.equal(value,reloaded.state_dict()[name]) for name,value in model.state_dict().items())
trace=dict(box=np.array([0.,0.,10.,10.]),template_commit_rgb_tir=[True,False],
    target_memory_commit_rgb_tir=[True,False],geometry_commit=True,extra_forwards=0)
labels=np.array([[0.,0.,10.,10.],[0.,0.,0.,0.],[0.,0.,10.,10.]])
result=label_native_traces([trace,trace,trace],labels)
assert result['current_iou']==1 and result['future_mean_iou']==1
assert torch.equal(result['future_valid'],torch.tensor([False,True]))
assert result['future_lost_fraction']==0 and result['wrong_geometry_fraction']==0
assert not torch.cuda.is_initialized()
receipt=dict(status='CPU_SYNTHETIC_CONSEQUENCE_TENSOR_CHECK_PASS',updates=2,
    initial_keep_and_no_search_equal=True,actual_group_gradients=gradients,
    changed_parameter_tensors=changed,strict_in_memory_reload=True,
    native_trace_label_adapter_and_invalid_GT_mask=True,CUDA_initialized=False,
    scope='Synthetic CPU decisions and native-trace adapter only; no images/pretrained tracker/GPU/actual collected states.',
    actual_training_complete=False,formal_metrics=None,goal_achieved=False)
Path('/home/gaob/CADTrack-ABC/profile_output/dense_consequence_cpu_20261015/contracts.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)
