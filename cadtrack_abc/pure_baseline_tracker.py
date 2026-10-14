"""Pure supplied CAD network with the original CAD test-time policy."""
import math

import numpy as np
import torch

from .baseline import CleanCADTrack, cfg
from .data import crop_batch


class PureCADTracker:
    def __init__(self, checkpoint):
        self.network = CleanCADTrack(checkpoint).cuda().eval()
        self.template_factor = cfg.TEST.TEMPLATE_FACTOR
        self.template_size = cfg.TEST.TEMPLATE_SIZE
        self.search_factor = cfg.TEST.SEARCH_FACTOR
        self.search_size = cfg.TEST.SEARCH_SIZE
        self.update_interval = cfg.TEST.UPDATE_INTERVALS
        self.update_threshold = cfg.TEST.UPDATE_THRESHOLD
        self.num_templates = cfg.DATA.TEMPLATE.NUMBER
        assert self.num_templates == 2
        feature_size = self.search_size // cfg.MODEL.BACKBONE.STRIDE
        window = .5 * (1 - torch.cos((2 * math.pi / (feature_size + 1)) *
                                    torch.arange(1, feature_size + 1).float()))
        self.window = (window.reshape(1, 1, -1, 1) * window.reshape(1, 1, 1, -1)).cuda()

    def crop(self, image, box, factor, size):
        # Preserve the original Python-double geometry before integer cropping.
        tensor, transform = crop_batch([image], torch.tensor([box], dtype=torch.float64),
                                       factor, size, 'cuda')
        # The original single-image preprocessor has batch stride 6, not H*W*6.
        return tensor[0].unsqueeze(0), size / float(transform[0, 2])

    @torch.inference_mode()
    def initialize(self, image, initial_box):
        self.box = [float(value) for value in initial_box]
        self.frame = 0
        template, _ = self.crop(image, self.box, self.template_factor, self.template_size)
        self.templates = [template] * self.num_templates

    @torch.inference_mode()
    def track(self, image):
        self.frame += 1
        search, resize_factor = self.crop(image, self.box, self.search_factor, self.search_size)
        output = self.network(self.templates, search)
        response = self.window * output['score']
        boxes = self.network.box_head.cal_bbox(response, output['size'], output['offset']).view(-1, 4)
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
        updated = self.frame % self.update_interval == 0 and bool(confidence > self.update_threshold)
        if updated:
            template, _ = self.crop(image, self.box, self.template_factor, self.template_size)
            self.templates.append(template)
            self.templates.pop(1)
        return {'box': np.asarray(self.box), 'confidence': float(confidence),
                'template_updated': updated}
