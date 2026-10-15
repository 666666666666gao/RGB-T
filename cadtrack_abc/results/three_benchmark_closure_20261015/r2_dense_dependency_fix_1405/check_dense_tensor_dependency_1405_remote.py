from pathlib import Path
import hashlib
import json
import os
import sys

project=Path('/home/gaob/CADTrack-ABC')
os.chdir(project)
sys.path.insert(0,str(project))
native=Path('/data/wangwj/CADTrack-main/lib/utils/tensor.py')
copied=project/'cadtrack_abc/vendor/CADTrack/lib/utils/tensor.py'
assert native.read_bytes()==copied.read_bytes()
import torch
from cadtrack_abc.dense_model import CompetitiveMemoryPrompt
torch.set_num_threads(4)
torch.manual_seed(42)
model=CompetitiveMemoryPrompt()
state=model.initialize(torch.randn(2,2,768))
features=torch.randn(2,2,768,16,16)
assert torch.equal(model(features,state),features)
candidate=dict(instance=torch.randn(2,5,2,768),raw=torch.rand(2,5))
encoded=model.encode_candidates(candidate,state)
advanced,commit=model.advance(state,encoded,torch.tensor([True,False]))
assert advanced['slots'].shape==(2,4,2,128)
assert commit.shape==(2,2)
loss=encoded['gate'].square().mean()+model(features,advanced).square().mean()
loss.backward()
assert torch.isfinite(model.output.weight.grad).all() and model.output.weight.grad.norm()>0
assert not torch.cuda.is_initialized()
result=dict(status='DENSE_IMPORT_AND_CPU_SHAPE_PASS',native_tensor_sha256=hashlib.sha256(copied.read_bytes()).hexdigest(),
    exact_native_copy=True,batch=2,candidates=5,sensors=2,slot_shape=list(advanced['slots'].shape),
    zero_prompt_exact=True,finite_nonzero_prompt_output_gradient=True,cuda_initialized=False,
    scope='Synthetic CPU tensor/import precondition only, not actual video GPU qualification, speed or tracking metric.')
print(json.dumps(result,indent=2),flush=True)
