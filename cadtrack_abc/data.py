from pathlib import Path
import math
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = torch.tensor([.485, .456, .406, .449, .449, .449])[None, :, None, None]
STD = torch.tensor([.229, .224, .225, .226, .226, .226])[None, :, None, None]


def read_pair(rgb, tir):
    return np.concatenate((cv2.cvtColor(cv2.imread(str(rgb)), cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(cv2.imread(str(tir)), cv2.COLOR_BGR2RGB)), -1)


def crop_batch(images, boxes, factor, size, device):
    # Same square extraction and zero padding as native CADTrack sample_target.
    patches, transforms = [], []
    for image, box in zip(images, boxes.detach().cpu().numpy()):
        x, y, w, h = box
        side = math.ceil(math.sqrt(float(w * h)) * factor)
        x0, y0 = round(float(x + w / 2 - side / 2)), round(float(y + h / 2 - side / 2))
        x1p, y1p = max(0, -x0), max(0, -y0)
        x2p, y2p = max(x0 + side - image.shape[1] + 1, 0), max(y0 + side - image.shape[0] + 1, 0)
        patch = image[y0 + y1p:y0 + side - y2p, x0 + x1p:x0 + side - x2p]
        patch = cv2.copyMakeBorder(patch, y1p, y2p, x1p, x2p, cv2.BORDER_CONSTANT)
        patches.append(cv2.resize(patch, (size, size)))
        transforms.append([x0, y0, side])
    tensor = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.
    return (tensor - MEAN.to(device)) / STD.to(device), torch.tensor(transforms, device=device, dtype=torch.float32)


def xyxy(boxes):
    return torch.cat((boxes[..., :2], boxes[..., :2] + boxes[..., 2:]), -1)


def xywh(boxes):
    return torch.cat((boxes[..., :2], boxes[..., 2:] - boxes[..., :2]), -1)


def clip_boxes(boxes, images):
    limits = boxes.new_tensor([[im.shape[1], im.shape[0]] for im in images])
    start = boxes[..., :2].maximum(torch.zeros_like(boxes[..., :2])).minimum(limits - 10)
    end = boxes[..., 2:].maximum(torch.full_like(boxes[..., 2:], 10)).minimum(limits)
    return torch.cat((start, (end - start).clamp(min=10)), -1)


def iou(boxes, target):
    end = torch.minimum(boxes[..., 2:], target[..., 2:])
    start = torch.maximum(boxes[..., :2], target[..., :2])
    inter = (end - start).clamp(min=0).prod(-1)
    area = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0).prod(-1)
    target_area = (target[..., 2:] - target[..., :2]).clamp(min=0).prod(-1)
    return inter / (area + target_area - inter).clamp(min=1e-6)


class VideoClips(Dataset):
    """Raw sequential clips; only the initial annotation initializes state.

    Later annotations are loss labels, never search centers or memory inputs.
    A small filename/annotation inventory is read once; images are read on demand.
    """
    def __init__(self, root, names, samples, seed, steps=4, horizon=3):
        self.root, self.names = Path(root), names
        self.samples, self.seed, self.epoch = samples, seed, 0
        self.steps, self.horizon = steps, horizon
        self.sequences = []
        for name in names:
            directory = self.root / name
            labels = np.loadtxt(directory / 'init.txt', delimiter=',', dtype=np.float32).reshape(-1, 4)
            rgb, tir = sorted((directory / 'visible').iterdir()), sorted((directory / 'infrared').iterdir())
            assert len(rgb) == len(tir) == len(labels), (name, len(rgb), len(tir), len(labels))
            valid = np.flatnonzero((labels[:, 2:] > 0).all(-1))
            self.sequences.append((name, rgb, tir, labels, valid))

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = random.Random(self.seed + self.epoch * self.samples + index)
        sequence_index = (index + self.epoch) % len(self.sequences)
        name, rgb, tir, labels, valid = self.sequences[sequence_index]
        count = len(labels)
        stride = 1  # Motion horizon is in actual frames, matching deployment.
        starts = valid[valid + (self.steps + self.horizon) * stride < count]
        assert len(starts), name
        first = int(starts[rng.randrange(len(starts))])
        ids = first + np.arange(self.steps + self.horizon + 1) * stride
        images = [read_pair(rgb[i], tir[i]) for i in ids[:self.steps + 1]]
        return {'name': name, 'images': images, 'boxes': labels[ids].copy(), 'frames': ids.copy()}


def collate_clips(items):
    return items
