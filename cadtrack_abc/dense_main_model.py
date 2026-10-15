"""Isolated R3 A/B/C integration, not imported by the running R2 pipeline.

Raw-clip training fits localization, A, B, and current-quality proxy evidence.
Search/future-action heads are trained later on matched consequences from this
locked localizer. Old R1 feature/action caches are not its training labels.
"""
import math

import torch

from .data import crop_batch, clip_boxes, iou, xyxy
from .dense_model import DenseCADTrack, dense_candidates
from .dense_temporal import TrustedMotion, MotionGridPrompt, motion_objective, accept_geometry_history
from .dense_submission import DenseCandidateEvidence, DenseDecisionHeads, submission_masks, proxy_objective
from .ported_modules import geometry


def clone_dense_state(state):
    """One-image counterfactual snapshot; preserve native template batch stride."""
    assert len(state['box']) == 1
    templates = []
    for value in state['templates']:
        copied = value.detach().permute(0, 2, 3, 1).contiguous().clone().permute(0, 3, 1, 2)
        templates.append(copied[0].unsqueeze(0))
    return dict(templates=templates, box=state['box'].detach().clone(), age=state['age'],
        memory={name: value.detach().clone() for name, value in state['memory'].items()},
        **{name: state[name].detach().clone() for name in ('history','frames','valid','quality')})


class DenseABCModel(DenseCADTrack):
    def __init__(self, base, threshold=.03):
        super().__init__(base, use_memory=True)
        self.motion = TrustedMotion()
        self.motion_prompt = MotionGridPrompt()
        self.candidate_evidence = DenseCandidateEvidence()
        self.decisions = DenseDecisionHeads(threshold)

    def load_localizer_state(self, state):
        DenseCADTrack.load_trainable_state(self, state)

    def trainable_state(self):
        return dict(localizer=DenseCADTrack.trainable_state(self), motion=self.motion.state_dict(),
            motion_prompt=self.motion_prompt.state_dict(), candidates=self.candidate_evidence.state_dict(),
            decisions=self.decisions.state_dict())

    def load_trainable_state(self, state):
        assert set(state) == {'localizer','motion','motion_prompt','candidates','decisions'}
        self.load_localizer_state(state['localizer'])
        self.motion.load_state_dict(state['motion'], strict=True)
        self.motion_prompt.load_state_dict(state['motion_prompt'], strict=True)
        self.candidate_evidence.load_state_dict(state['candidates'], strict=True)
        self.decisions.load_state_dict(state['decisions'], strict=True)

    def initialize_history(self, state):
        batch = len(state['box'])
        history = xyxy(state['box'])[:, None].expand(-1, 8, -1).clone()
        valid = torch.zeros(batch, 8, dtype=torch.bool, device=self.device)
        valid[:, -1] = True
        return dict(state, history=history, valid=valid, quality=valid.float(),
            frames=torch.zeros(batch, 8, dtype=torch.long, device=self.device))

    def initialize_batch(self, images, boxes):
        return self.initialize_history(super().initialize_batch(images, boxes))

    def forecast(self, state):
        current = torch.full_like(state['frames'][:, -1], state['age'] + 1)
        return self.motion(state['history'],state['frames'],state['valid'],state['quality'],
            state['memory']['anchor'],state['memory']['slots'],current)

    def predict_temporal(self, templates, search, memory, distribution, transform, reference):
        # This explicit new path leaves the active R2 predict() implementation intact.
        features = self.extract(templates, search)
        prompted = self.prompt(features['modality'], memory)
        rgb = prompted[:, 0].flatten(2).transpose(1, 2)
        tir = prompted[:, 1].flatten(2).transpose(1, 2)
        fused = self.base.decode_fuse_search(torch.cat((rgb, tir), -1).transpose(1, 2).reshape(-1,1536,16,16))
        fused = self.motion_prompt(fused, distribution, transform, reference)
        score, _, size, offset = self.base.box_head(fused)
        return dict(score=score,size=size,offset=offset,fused=fused,modality=features['modality'])

    def observe_local_patch(self, patch, transform, reference, state):
        distribution = self.forecast(state)
        maps = self.predict_temporal(state['templates'],patch,state['memory'],distribution,transform,reference)
        candidate = dense_candidates(maps,transform,reference,self.window)
        appearance = self.prompt.encode_candidates(candidate,state['memory'])
        encoded = self.candidate_evidence.encode(candidate,appearance,distribution,0.)
        context = self.candidate_evidence.contextualize(candidate,encoded)
        search, summary = self.decisions.search(context,distribution)
        return dict(distribution=distribution,local_maps=maps,local_transform=transform,
            local_reference=reference,local=candidate,local_appearance=appearance,
            local_encoded=encoded,local_context=context,pair=candidate,appearance=appearance,
            context=context,search=search,search_input=summary,extra_forwards=0)

    def observe_extra_patch(self, patch, transform, reference, state, query):
        """Exactly one executed region; separate alternatives never see each other."""
        assert query['extra_forwards'] == 0
        maps = self.predict_temporal(state['templates'],patch,state['memory'],query['distribution'],transform,reference)
        extra = dense_candidates(maps,transform,reference,self.window)
        appearance = self.prompt.encode_candidates(extra,state['memory'])
        encoded = self.candidate_evidence.encode(extra,appearance,query['distribution'],1.)
        pair = {key:torch.cat((query['local'][key],extra[key]),1) for key in extra}
        combined_appearance = {key:torch.cat((query['local_appearance'][key],appearance[key]),1) for key in appearance}
        combined_encoded = {key:torch.cat((query['local_encoded'][key],encoded[key]),1) for key in encoded}
        context = self.candidate_evidence.contextualize(pair,combined_encoded)
        return dict(query,pair=pair,appearance=combined_appearance,context=context,
            extra_maps=maps,extra_transform=transform,extra_reference=reference,extra_forwards=1)

    @staticmethod
    def forecast_box(distribution, mode):
        rows = torch.arange(len(mode),device=mode.device)
        target = distribution['means'][rows,mode,0]
        center, size = geometry(distribution['reference'])
        center = center + target[:, :2] * size
        size = size * target[:, 2:].clamp(-math.log(4),math.log(4)).exp()
        return torch.cat((center-size/2,center+size/2),-1)

    def plan_submission(self, state, query, choice, mode, selected_box, propagated_box):
        """Pure state proposal; only apply_submission installs it once."""
        rows = torch.arange(len(choice),device=choice.device)
        context, pair = query['context'],query['pair']
        geometric, permit = submission_masks(choice,mode,context['geometry'][rows,choice])
        order = torch.arange(pair['raw'].shape[1],device=choice.device)[None].expand(len(choice),-1).clone()
        order[:,0] = choice
        order[rows,choice] = 0
        appearance = {name:value[rows[:,None],order] for name,value in query['appearance'].items()}
        age = state['age'] + 1
        requested = (age % 5 == 0) & (pair['hann'][rows,choice] > .65)
        memory, commit = self.prompt.advance(state['memory'],appearance,requested)
        memory = dict(anchor=state['memory']['anchor'],
            slots=torch.where(permit[:,None,:,None],memory['slots'],state['memory']['slots']))
        commit = commit & permit
        reference = torch.where(geometric[:,None],selected_box,propagated_box).detach()
        frames = torch.full_like(state['frames'][:,-1],age)
        accepted = accept_geometry_history(state,xyxy(selected_box),frames,pair['hann'][rows,choice],geometric)
        updated = dict(state,age=age,box=reference,memory=memory,**accepted)
        return dict(state=updated,box=selected_box.detach(),choice=choice,mode=mode,
            geometry_commit=geometric,template_commit=commit,target_memory_commit=commit,
            template_write_requested=requested,extra_forwards=query['extra_forwards'])

    def apply_submission(self, prior, plan, template):
        templates = list(prior['templates'])
        if plan['template_commit'].any():
            assert template is not None
            mask = plan['template_commit'].repeat_interleave(3,-1)[:,:,None,None]
            templates[1] = torch.where(mask,template,templates[1])
        return dict(plan['state'],templates=templates)

    def raw_training_step(self, images, state):
        """Localizer stage only; action-head training uses later real consequences."""
        patch, transform = crop_batch(images,state['box'],4.,256,self.device)
        query = self.observe_local_patch(patch,transform,state['box'],state)
        batch = len(images)
        choice = torch.zeros(batch,dtype=torch.long,device=self.device)
        mode = torch.zeros_like(choice)
        selected = clip_boxes(query['local']['boxes'][:,0],images).detach()
        dominant = query['distribution']['log_weights'].argmax(-1)
        propagated = clip_boxes(self.forecast_box(query['distribution'],dominant),images)
        plan = self.plan_submission(state,query,choice,mode,selected,propagated)
        template = None
        if plan['template_commit'].any():
            template,_ = crop_batch(images,selected,2.,128,self.device)
        updated = self.apply_submission(state,plan,template)
        observation = dict(maps=query['local_maps'],candidate=query['local'],encoded=query['local_appearance'],
            memory_before=state['memory'],reference=state['box'],transform=query['local_transform'],
            box=selected,template_commit=plan['template_commit'],distribution=query['distribution'],
            context=query['local_context'])
        return observation,updated

    def forward(self, clips):
        """Raw A/B localization stage, not final-policy video evaluation."""
        labels = torch.tensor([clip['boxes'].tolist() for clip in clips],device=self.device)
        state = self.initialize_batch([clip['images'][0] for clip in clips],labels[:,0])
        teacher = [state['memory']['anchor'][:,None].detach()]
        teacher_valid = [torch.ones(len(clips),1,dtype=torch.bool,device=self.device)]
        steps = len(clips[0]['images']) - 1
        losses,predictions,parts = [],[],{}
        for time in range(1,steps+1):
            observation,state = self.raw_training_step([clip['images'][time] for clip in clips],state)
            predictions.append(observation['box'])
            current,valid = self.localization_losses(observation,labels[:,time])
            memory,quality = self.memory_losses(observation,labels[:,time],valid,teacher,teacher_valid)
            future = labels[:,time:time+3]
            motion = motion_objective(observation['distribution'],xyxy(future),(future[...,2:] > 0).all(-1))
            candidate_quality = iou(observation['candidate']['boxes'].detach(),xyxy(labels[:,time])[:,None])
            proxy = proxy_objective(observation['context'],candidate_quality,valid)
            loss = current['focal'] + 5*current['l1'] + 2*current['giou']
            loss += .2*memory['gate'] + .1*(memory['identity']+memory['preservation']+motion) + .2*proxy
            losses.append(loss)
            for name,value in dict(current,**memory,motion=motion,proxy=proxy).items():
                parts[name] = parts.get(name,0.) + value.detach()/steps
            teacher.append(observation['encoded']['descriptor'][:,:1].detach())
            teacher_valid.append(((quality >= .5) & valid)[:,None])
        return dict(loss=torch.stack(losses).mean(),parts=parts,predictions=torch.stack(predictions,1),
            stage='RAW_DENSE_MOTION_ONLY',future_action_training_complete=False)
