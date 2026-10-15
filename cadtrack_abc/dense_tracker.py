"""Dense R2 inference preserves the original CAD crop and Python-double mapping."""
import hashlib
from pathlib import Path
import numpy as np
import torch

from .dense_model import DenseCADTrack, dense_candidates
from .pure_baseline_tracker import PureCADTracker


class DenseCADTracker(PureCADTracker):
    def __init__(self, pretrained, use_memory):
        super().__init__(pretrained)
        self.pretrained = Path(pretrained).resolve()
        self.network = DenseCADTrack(self.network, use_memory).cuda().eval()
        self.memory = None

    def load_payload(self, payload):
        assert payload['model_family'] == 'CADTrackDense-R2'
        assert payload['use_memory'] == (self.network.prompt is not None)
        assert Path(payload['config']['pretrained']).resolve() == self.pretrained
        assert hashlib.sha256(self.pretrained.read_bytes()).hexdigest() == payload['config']['pretrained_sha256']
        self.network.load_trainable_state(payload['dense'])
        self.network.eval()
        return payload

    @torch.inference_mode()
    def initialize(self, image, initial_box):
        super().initialize(image, initial_box)
        if self.network.prompt is not None:
            search, _ = self.crop(image, self.box, self.search_factor, self.search_size)
            boxes = torch.tensor([self.box], device='cuda', dtype=torch.float32)
            self.memory = self.network.initialize_memory(self.templates, search, boxes)

    @torch.inference_mode()
    def track(self, image):
        self.frame += 1
        reference = list(self.box)
        search, resize_factor = self.crop(image, self.box, self.search_factor, self.search_size)
        output = self.network.predict(self.templates, search, self.memory)
        response = self.window * output['score']
        boxes = self.network.base.box_head.cal_bbox(response, output['size'], output['offset']).view(-1, 4)
        cx, cy, width, height = (boxes.mean(0) * self.search_size / resize_factor).tolist()
        half_side = .5 * self.search_size / resize_factor
        cx += self.box[0] + .5 * self.box[2] - half_side
        cy += self.box[1] + .5 * self.box[3] - half_side
        x1, y1 = cx - .5 * width, cy - .5 * height
        x2, y2 = x1 + width, y1 + height
        image_height, image_width = image.shape[:2]
        x1, x2 = min(max(0, x1), image_width - 10), min(max(10, x2), image_width)
        y1, y2 = min(max(0, y1), image_height - 10), min(max(10, y2), image_height)
        self.box = [x1, y1, max(10, x2 - x1), max(10, y2 - y1)]
        confidence = response.flatten(1).max(1, keepdim=True).values
        write = self.frame % self.update_interval == 0 and bool(confidence > self.update_threshold)
        transform = torch.tensor([[0., 0., self.search_size / resize_factor]], device='cuda')
        candidate = dense_candidates(output, transform, torch.tensor([reference], device='cuda'), self.window)
        if self.network.prompt is not None:
            encoded = self.network.prompt.encode_candidates(candidate, self.memory)
            self.memory, commit = self.network.prompt.advance(self.memory, encoded, torch.tensor([write], device='cuda'))
        else:
            commit = torch.full((1, 2), write, device='cuda', dtype=torch.bool)
        if write:
            template, _ = self.crop(image, self.box, self.template_factor, self.template_size)
            if self.network.prompt is None:
                self.templates[1] = template
            else:
                mask = commit.repeat_interleave(3, -1)[:, :, None, None]
                merged = torch.where(mask, template, self.templates[1])
                merged = merged.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
                self.templates[1] = merged[0].unsqueeze(0)
        template_commit = commit[0].cpu().tolist()
        return dict(box=np.asarray(self.box), confidence=float(confidence),
            template_updated=any(template_commit), choice=0, extra_forwards=0,
            geometry_commit=True, template_write_requested=write,
            template_commit_rgb_tir=template_commit,
            target_memory_commit_rgb_tir=template_commit if self.network.prompt is not None else [False, False],
            next_geometry_reference_xywh=list(self.box), keep_box_xywh=list(self.box),
            candidate_boxes_xyxy=candidate['boxes'][0].cpu().tolist(),
            predicted_quality=float(confidence))
