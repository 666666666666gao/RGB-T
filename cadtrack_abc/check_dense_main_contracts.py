"""CPU synthetic component contracts, never a tracking-accuracy experiment.

Run only in the existing warm environment after source review, CUDA hidden.
No images, pretrained tensor reload, real-video parity or GPU fit is claimed.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json

import torch
from torch.nn import functional as F

from .dense_model import CompetitiveMemoryPrompt
from .dense_main_model import DenseABCModel, clone_dense_state
from .dense_main_tracker import native_candidate_boxes
from .dense_temporal import TrustedMotion, MotionGridPrompt, accept_geometry_history
from .dense_submission import DenseCandidateEvidence, DenseDecisionHeads, submission_masks, consequence_objective
from .data import xyxy
from .counterfactual import ACTION_MODES
from lib.models.layers.head import CenterPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    checks = []

    def check(name, condition):
        assert bool(condition),name
        checks.append(name)

    # Two accepted observations establish10 pixels/frame, independent of any
    # subsequently rejected output. Middle mode starts with zero residual.
    history = torch.tensor([[[0.,0.,20.,20.]]*7+[[10.,0.,30.,20.]]])
    frames = torch.zeros(1,8,dtype=torch.long)
    frames[:,-1] = 1
    valid = torch.zeros(1,8,dtype=torch.bool)
    valid[:,-2:] = True
    quality = valid.float()
    anchor = F.normalize(torch.randn(1,2,128),dim=-1)
    slots = torch.randn(1,4,2,128)
    motion = TrustedMotion().eval()
    first = motion(history,frames,valid,quality,anchor,slots,torch.tensor([2]))
    later = motion(history,frames,valid,quality,anchor,slots,torch.tensor([3]))
    center_x = 20+first['means'][0,1,0,0]*20
    later_x = 20+later['means'][0,1,0,0]*20
    check('clock advances predicted center from30 to40 while accepted history is held',
        torch.equal(torch.stack((center_x,later_x)),torch.tensor([30.,40.])))
    check('missing observation increases uncertainty',later['deviations'][0,1,0,0] > first['deviations'][0,1,0,0])
    check('three probabilities sum toone',torch.allclose(first['log_weights'].exp().sum(-1),torch.ones(1)))

    fields = dict(history=history,frames=frames,valid=valid,quality=quality)
    held = accept_geometry_history(fields,torch.tensor([[900.,900.,920.,920.]]),
        torch.tensor([2]),torch.tensor([1.]),torch.tensor([False]))
    check('rejected geometry leaves all accepted-history fields unchanged',
        all(torch.equal(held[name],value) for name,value in fields.items()))

    prior = MotionGridPrompt()
    features = torch.randn(1,768,16,16,requires_grad=True)
    transform = torch.tensor([[0.,0.,80.]])
    reference = torch.tensor([[10.,0.,20.,20.]])
    output = prior(features,first,transform,reference)
    check('zero motion residual preserves fused visual features exactly',torch.equal(output,features))
    output.square().mean().backward()
    check('motion feature insertion receives a finite nonzero gradient',
        torch.isfinite(prior.projection.weight.grad).all() and prior.projection.weight.grad.abs().sum() > 0)

    choice = torch.zeros(6,dtype=torch.long)
    mode = torch.arange(6)
    geometric,permit = submission_masks(choice,mode,torch.full((6,),3.))
    check('six actions have the declared geometry decisions',
        torch.equal(geometric,torch.tensor([True,True,False,True,True,True])))
    check('defer geometry quarantines both identity modalities',not permit[2].any())
    check('RGB-only and TIR-only restrict commits independently',
        torch.equal(permit[4:],torch.tensor([[True,False],[False,True]])))

    memory_module = CompetitiveMemoryPrompt()
    memory = dict(anchor=anchor,slots=memory_module.memory.initialize(anchor))
    appearance = dict(descriptor=F.normalize(torch.randn(1,5,2,128),dim=-1),
        gate=torch.tensor([[[[3.,0.,0.],[3.,0.,0.]]]*5]),support=torch.randn(1,5,2,3))
    template = torch.rand(128,128,6).permute(2,0,1).unsqueeze(0)
    state = dict(templates=[template,template],box=reference,age=4,memory=memory,**fields)
    packet = dict(raw=torch.full((1,5),.9),hann=torch.full((1,5),.9))
    query = dict(pair=packet,appearance=appearance,context=dict(geometry=torch.full((1,5),3.)),extra_forwards=0)
    selected = torch.tensor([[900.,900.,20.,20.]])
    propagated = torch.tensor([[30.,0.,20.,20.]])
    # Use the real state-proposal implementation and actual A module without
    # constructing a fake pretrained vision tracker.
    harness = SimpleNamespace(prompt=memory_module)
    proposal = DenseABCModel.plan_submission(harness,state,query,torch.tensor([0]),
        torch.tensor([ACTION_MODES.index('defer_geometry')]),selected,propagated)
    check('joint defer blocks a high-confidence frame5 template write',
        proposal['template_write_requested'].all() and not proposal['template_commit'].any())
    check('joint defer keeps all modality memory slots unchanged',
        torch.equal(proposal['state']['memory']['slots'],memory['slots']))
    check('joint defer preserves trusted geometry instead of inserting the selected box',
        torch.equal(proposal['state']['history'],history) and torch.equal(proposal['state']['box'],propagated))
    cloned = clone_dense_state(state)
    check('snapshot templates preserve native batch stride6',all(value.stride()[0]==6 for value in cloned['templates']))
    check('snapshot clones preserve values without aliasing mutable memory',
        torch.equal(cloned['memory']['slots'],memory['slots']) and
        cloned['memory']['slots'].data_ptr()!=memory['slots'].data_ptr())

    evidence = DenseCandidateEvidence().eval()
    candidate = dict(boxes=xyxy(reference)[:,None].expand(-1,5,-1).clone(),
        raw=torch.full((1,5),.7),hann=torch.full((1,5),.6),fused=torch.randn(1,5,768))
    encoded = evidence.encode(candidate,appearance,first,0.)
    local = evidence.contextualize(candidate,encoded)
    more = {name:torch.cat((value,value+torch.randn_like(value)),1) for name,value in candidate.items()}
    more['raw'][:,5:] = .6
    more['hann'][:,5:] = .5
    combined = {name:torch.cat((value,value+torch.randn_like(value)),1) for name,value in encoded.items()}
    joint = evidence.contextualize(more,combined)
    check('observed extra evidence can update local candidate tokens bidirectionally',
        not torch.equal(local['tokens'],joint['tokens'][:,:5]))
    decisions = DenseDecisionHeads().eval()
    selected_candidate,selected_mode,_ = decisions.choose(candidate,local,first,torch.tensor([0]))
    check('zero future heads retain the parent local keep action',int(selected_candidate[0])==0 and int(selected_mode[0])==0)

    head = SimpleNamespace(feat_sz=16)
    head.cal_bbox = lambda score,size,offset: CenterPredictor.cal_bbox(head,score,size,offset)
    maps = dict(score=torch.rand(1,1,16,16),size=torch.rand(1,2,16,16)*.3,
        offset=torch.rand(1,2,16,16))
    window = torch.ones(1,1,16,16)
    native_reference = [103.123456789,57.234567891,23.456789123,31.345678912]
    resize = 256/108.
    boxes = native_candidate_boxes(head,maps,window,native_reference,resize,(512,640,6))
    cx,cy,w,h = (head.cal_bbox(maps['score']*window,maps['size'],maps['offset']).view(-1,4).mean(0)*256/resize).tolist()
    cx += native_reference[0]+.5*native_reference[2]-.5*256/resize
    cy += native_reference[1]+.5*native_reference[3]-.5*256/resize
    x1,y1 = cx-.5*w,cy-.5*h
    x2,y2 = x1+w,y1+h
    x1,x2 = min(max(0,x1),630),min(max(10,x2),640)
    y1,y2 = min(max(0,y1),502),min(max(10,y2),512)
    check('candidate0 preserves native Python-double mapping',boxes[0]==[x1,y1,max(10,x2-x1),max(10,y2-y1)])

    future = decisions.future_action(torch.randn(1,2,275),torch.tensor([0]))
    future['harm_logits'].retain_grad()
    utility = torch.zeros(1,2,6)
    utility[0,1,0] = .5
    quality_future = torch.full((1,2,6),.8)
    quality_future[0,1,0] = .2
    lost = torch.zeros_like(quality_future)
    lost[0,1,0] = .7
    loss = consequence_objective(future,utility,quality_future,lost,torch.tensor([0]))
    loss.backward()
    check('a positive total utility does not erase future-harm supervision',future['harm_logits'].grad[0,1,0] < 0)
    check('CPU contracts did not initialize CUDA',not torch.cuda.is_initialized())
    result = dict(status='CPU_COMPONENT_CONTRACTS_PASS',checks=checks,count=len(checks),
        scope='simulation_only: synthetic components and state proposals, no real images or pretrained/GPU execution.',
        full_backbone_forward=False,real_video_parity=False,GPU_qualification=False,
        training_complete=False,formal_metrics=None,goal_achieved=False)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
