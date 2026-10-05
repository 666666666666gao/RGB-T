"""Actual frozen-parent/video parity for a zero-initialized commit policy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .collect_recoverability import InstanceExtractor
from .evaluate_online import read_pair
from .geometry_commit import GeometryCommitHead, GeometryCommitTracker
from .recoverability_modules import RecoverabilityModules
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import TemporalModules


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',required=True); p.add_argument('--fit',required=True)
    p.add_argument('--c1-head',required=True); p.add_argument('--motion-run',required=True)
    p.add_argument('--pretrained',required=True); p.add_argument('--sequence-root',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    completion=json.loads((Path(a.fit)/'completion.json').read_text())
    assert completion['complete'] and completion['sanity_only'] and completion['optimizer_updates']==4
    trained=torch.load(Path(a.fit)/'best.pth',map_location='cpu',weights_only=False)
    trained_head=GeometryCommitHead(); trained_head.load_state_dict(trained['head'],strict=True)
    device=torch.device('cuda:0'); torch.set_num_threads(4); torch.manual_seed(42)
    c1=torch.load(a.c1_head,map_location='cpu',weights_only=False)
    parent=torch.load(a.parent,map_location='cpu',weights_only=False)
    modules=RecoverabilityModules(c1).to(device)
    modules.load_state_dict(parent['model'],strict=True); modules.eval().requires_grad_(False)
    config=json.loads((Path(a.motion_run)/'config.json').read_text())
    saved=torch.load(Path(a.motion_run)/'last.pth',map_location='cpu',weights_only=False)
    assert saved['epoch']==30 and parent['epoch']==4
    motion=TemporalModules(c1,config['slots'],config['motion_history'],config['modes'],saved['horizon']).to(device)
    motion.load_state_dict(saved['model'],strict=True); motion.eval().requires_grad_(False)
    extractor=InstanceExtractor(a.pretrained,c1['args']['candidates'],.45,c1['args']['nms_iou']).to(device)
    root=Path(a.sequence_root)
    visible=sorted((root/'visible').iterdir()); thermal=sorted((root/'infrared').iterdir())
    assert len(visible)==len(thermal) and len(visible)>120
    initial=np.fromstring((root/'init.txt').read_text().splitlines()[0],sep=','); initial[2:]+=initial[:2]
    with torch.inference_mode():
        image=read_pair(visible[0],thermal[0],device)
        base=RecoverabilityTracker(extractor,modules,motion,image,initial,parent['args']['threshold'],write_verification='action')
        zero=GeometryCommitHead().to(device)
        control=GeometryCommitTracker(extractor,modules,motion,image,initial,parent['args']['threshold'],write_verification='action',commit_head=zero)
        for frame in range(1,120):
            image=read_pair(visible[frame],thermal[frame],device)
            before=base.step(image); after=control.step(image)
            assert np.array_equal(before,after) and np.array_equal(base.branch.search_box,control.branch.search_box)
            assert torch.equal(base.branch.template,control.branch.template)
            assert torch.equal(base.identity_memory,control.identity_memory) and torch.equal(base.motion_memory,control.motion_memory)
            assert bool(base.last_decision['template_updated'])==bool(control.last_decision['template_updated'])
            assert all(np.array_equal(x[0],y[0]) and x[1:]==y[1:] for x,y in zip(base.history,control.history))
        assert control.stats['geometry_commit_interventions']==control.stats['appearance_commit_overrides']==0
    out=Path(a.output); assert not out.exists(); out.mkdir()
    (out/'completion.json').write_text(json.dumps({'complete':True,'source_runtime_parity':True,
        'frames_including_initialization':120,'sequence':root.name,'fit_updates':4,
        'scope':'real zero-initialization bitwise old4 output/search/template/memory/history parity; not accuracy',
        'future_GT_input':False,'selected_sanity_head_epoch':trained['epoch']},indent=2)+'\n')
    (out/'COMPLETE').write_text('COMPLETE\n')


if __name__=='__main__':
    main()
