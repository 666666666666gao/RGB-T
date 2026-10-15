"""The same causal video runner serves full98 selection and official inference."""
import time
import numpy as np
import torch

from .data import read_pair
from .dense_tracker import DenseCADTracker


def load_dense_tracker(checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    tracker = DenseCADTracker(payload['config']['pretrained'], payload['use_memory'])
    tracker.load_payload(payload)
    return tracker, payload


@torch.inference_mode()
def track_video(tracker, rgb, tir, initial):
    assert len(rgb) == len(tir)
    assert initial.shape == (4,) and (initial[2:] > 0).all()
    tracker.initialize(read_pair(rgb[0], tir[0]), initial)
    boxes, latencies, diagnostics = [initial], [], []
    for frame in range(1, len(rgb)):
        torch.cuda.synchronize()
        started = time.perf_counter()
        observation = tracker.track(read_pair(rgb[frame], tir[frame]))
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)
        boxes.append(observation['box'])
        diagnostics.append({key:value for key,value in observation.items() if key != 'box'})
    return np.stack(boxes), np.asarray(latencies), diagnostics
