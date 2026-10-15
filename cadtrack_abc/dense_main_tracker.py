"""Strictly online R3 runtime; at most one additional visual region per frame.

Observation is separate from submission for matched counterfactual labels.
The real single-image crop and Python-double box mapping are retained.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch

from .counterfactual import ACTION_MODES
from .data import clip_boxes
from .dense_main_model import DenseABCModel, clone_dense_state
from .pure_baseline_tracker import PureCADTracker


def native_candidate_boxes(head, output, window, reference, resize, image_shape, search_size=256):
    response = output['score'] * window
    suppressed = response.clone()
    axis = torch.arange(16,device=response.device)
    ys,xs = torch.meshgrid(axis,axis,indexing='ij')
    predictions = []
    for _ in range(5):
        predictions.append(head.cal_bbox(suppressed,output['size'],output['offset']).view(-1,4).mean(0))
        index = suppressed.flatten(1).argmax(-1)
        near = (ys-index[0]//16).abs().maximum((xs-index[0]%16).abs()) <= 1
        suppressed = suppressed.masked_fill(near[None,None],-torch.inf)
    values = (torch.stack(predictions) * search_size / resize).tolist()
    height,width = image_shape[:2]
    half_side = .5 * search_size / resize
    result = []
    for cx,cy,w,h in values:
        cx += reference[0] + .5*reference[2] - half_side
        cy += reference[1] + .5*reference[3] - half_side
        x1,y1 = cx-.5*w,cy-.5*h
        x2,y2 = x1+w,y1+h
        x1,x2 = min(max(0,x1),width-10),min(max(10,x2),width)
        y1,y2 = min(max(0,y1),height-10),min(max(10,y2),height)
        result.append([x1,y1,max(10,x2-x1),max(10,y2-y1)])
    return result


class DenseABCTracker(PureCADTracker):
    def __init__(self, pretrained, threshold=.03):
        super().__init__(pretrained)
        self.pretrained = Path(pretrained).resolve()
        self.network = DenseABCModel(self.network,threshold).cuda().eval()
        self.state = None

    def load_payload(self, payload):
        assert payload['model_family'] == 'CADTrackDenseABC-R3'
        assert payload['config']['intervention_threshold'] == self.network.decisions.threshold
        assert Path(payload['config']['pretrained']).resolve() == self.pretrained
        assert hashlib.sha256(self.pretrained.read_bytes()).hexdigest() == payload['config']['pretrained_sha256']
        self.network.load_trainable_state(payload['dense_abc'])
        self.network.eval()
        return payload

    @torch.inference_mode()
    def initialize(self, image, initial_box):
        super().initialize(image,initial_box)
        search,_ = self.crop(image,self.box,self.search_factor,self.search_size)
        boxes = torch.tensor([self.box],dtype=torch.float32,device='cuda')
        memory = self.network.initialize_memory(self.templates,search,boxes)
        self.state = self.network.initialize_history(dict(templates=self.templates,box=boxes,age=0,memory=memory))

    @torch.inference_mode()
    def observe_local(self, image):
        search,resize = self.crop(image,self.box,self.search_factor,self.search_size)
        transform = torch.tensor([[0.,0.,self.search_size/resize]],device='cuda')
        reference = torch.tensor([self.box],dtype=torch.float32,device='cuda')
        query = self.network.observe_local_patch(search,transform,reference,self.state)
        query['native_boxes'] = native_candidate_boxes(self.network.base.box_head,
            query['local_maps'],self.window,self.box,resize,image.shape,self.search_size)
        return query

    @torch.inference_mode()
    def observe_extra(self, image, query, mode):
        mode_tensor = torch.tensor([mode],device='cuda',dtype=torch.long)
        reference = clip_boxes(self.network.forecast_box(query['distribution'],mode_tensor),[image])[0].tolist()
        search,resize = self.crop(image,reference,6.,self.search_size)
        transform = torch.tensor([[0.,0.,self.search_size/resize]],device='cuda')
        packet = self.network.observe_extra_patch(search,transform,
            torch.tensor([reference],device='cuda'),self.state,query)
        additional = native_candidate_boxes(self.network.base.box_head,packet['extra_maps'],
            self.window,reference,resize,image.shape,self.search_size)
        packet['native_boxes'] = query['native_boxes'] + additional
        packet['extra_mode'] = mode
        return packet

    @torch.inference_mode()
    def observe(self, image):
        query = self.observe_local(image)
        score,rank = query['search']['scores'].max(1)
        if bool(score[0] > self.network.decisions.threshold + .01):
            order = query['distribution']['log_weights'][0].argsort(descending=True,stable=True)
            query = self.observe_extra(image,query,int(order[rank[0]]))
        return query

    @torch.inference_mode()
    def advance(self, image, query, candidate, mode):
        """The only method that advances frame, templates, memory and geometry."""
        assert mode in ACTION_MODES
        choice = torch.tensor([candidate],device='cuda',dtype=torch.long)
        mode_tensor = torch.tensor([ACTION_MODES.index(mode)],device='cuda',dtype=torch.long)
        native_box = list(query['native_boxes'][candidate])
        selected = torch.tensor([native_box],device='cuda',dtype=torch.float32)
        dominant = query['distribution']['log_weights'].argmax(-1)
        propagated = clip_boxes(self.network.forecast_box(query['distribution'],dominant),[image])
        plan = self.network.plan_submission(self.state,query,choice,mode_tensor,selected,propagated)
        template = None
        if plan['template_commit'].any():
            template,_ = self.crop(image,native_box,self.template_factor,self.template_size)
        self.state = self.network.apply_submission(self.state,plan,template)
        if plan['template_commit'].any():
            merged = self.state['templates'][1].permute(0,2,3,1).contiguous().permute(0,3,1,2)
            self.state['templates'][1] = merged[0].unsqueeze(0)
        geometric = bool(plan['geometry_commit'][0])
        self.box = native_box if geometric else propagated[0].tolist()
        self.templates = self.state['templates']
        self.frame += 1
        assert self.frame == self.state['age']
        commit = plan['template_commit'][0].cpu().tolist()
        return dict(box=np.asarray(native_box),confidence=float(query['pair']['hann'][0,candidate]),
            choice=candidate,extra_forwards=query['extra_forwards'],geometry_commit=geometric,
            template_updated=any(commit),template_commit_rgb_tir=commit,
            target_memory_commit_rgb_tir=plan['target_memory_commit'][0].cpu().tolist(),
            template_write_requested=bool(plan['template_write_requested'][0]),
            next_geometry_reference_xywh=list(self.box),keep_box_xywh=list(query['native_boxes'][0]),
            candidate_boxes_xyxy=query['pair']['boxes'][0].cpu().tolist(),future_mode=mode,
            predicted_quality=float(query['context']['quality'][0,candidate].sigmoid()))

    @torch.inference_mode()
    def track(self, image):
        query = self.observe(image)
        age = torch.full((1,),self.state['age'],device='cuda',dtype=torch.long)
        candidate,mode,_ = self.network.decisions.choose(query['pair'],query['context'],query['distribution'],age)
        return self.advance(image,query,int(candidate[0]),ACTION_MODES[int(mode[0])])

    def snapshot(self):
        return dict(box=list(self.box),frame=self.frame,state=clone_dense_state(self.state))

    def restore(self, snapshot):
        self.box = list(snapshot['box'])
        self.frame = snapshot['frame']
        self.state = clone_dense_state(snapshot['state'])
        self.templates = self.state['templates']
