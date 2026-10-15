"""Fixed-weight R1 image-template controls; other state commits stay native."""
import torch

from .data import clip_boxes, crop_batch

TEMPLATE_POLICIES = ('current', 'static', 'cad_selected', 'cad_local')


class R1TemplateUpdateControl:
    """Wrap the R1 search-actions policy without changing its current decision.

    The next frame sees the controlled dynamic image template. Anchor, feature
    memory, geometry and motion history retain the parent's updates. A different
    template can naturally change those decisions on later frames.
    """
    def __init__(self, policy, template_policy, interval=5, threshold=.65):
        assert template_policy in TEMPLATE_POLICIES
        assert interval > 0
        self.policy = policy
        self.template_policy = template_policy
        self.interval = interval
        self.threshold = threshold

    def initialize(self, images, boxes, frames):
        return self.policy.initialize(images, boxes, frames)

    @torch.inference_mode()
    def step(self, images, frames, state):
        # R1 search-actions clones the pre-decision state in advance_candidate.
        previous_dynamic = state['templates'][1]
        observation, advanced = self.policy.step(images, frames, state)
        parent_request = observation['write'].clone()
        parent_commit = observation['template_commit'].clone()
        choice = int(observation['choice'][0])
        source_index, source_box = choice, observation['box']
        source_score = observation['pair']['hann'][:, choice]
        if self.template_policy != 'current':
            advanced['templates'][1] = previous_dynamic
            request = torch.zeros_like(parent_request)
            if self.template_policy == 'static':
                source_index, source_box, source_score = -1, None, None
            else:
                source_index = choice if self.template_policy == 'cad_selected' else 0
                candidates = observation['pair'] if self.template_policy == 'cad_selected' else observation['local']
                source_box = clip_boxes(candidates['boxes'][:, source_index], images)
                source_score = candidates['hann'][:, source_index]
                request = (advanced['age'] % self.interval == 0) & (source_score > self.threshold)
                if bool(request[0]):
                    template, _ = crop_batch(images, source_box, 2., 128, self.policy.device)
                    advanced['templates'][1] = template
            observation['write'] = request
            observation['template_commit'] = request[:, None].expand(-1, 2).clone()
        observation['template_control'] = {
            'policy': self.template_policy,
            'parent_request': bool(parent_request[0]),
            'parent_commit_rgb_tir': parent_commit[0].cpu().tolist(),
            'source_index': source_index,
            'source_box_xywh': None if source_box is None else source_box[0].cpu().tolist(),
            'source_hann_score': None if source_score is None else float(source_score[0]),
            'rule': 'parent raw/risk/action/modal gates' if self.template_policy == 'current'
                    else 'no dynamic image writes' if self.template_policy == 'static'
                    else f'age modulo {self.interval} == 0 and own Hann score > {self.threshold}',
        }
        return observation, advanced
