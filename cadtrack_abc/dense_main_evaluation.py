"""Strict R3 loading with the existing unchanged causal video runner."""
import torch

from .dense_evaluation import track_video
from .dense_main_tracker import DenseABCTracker


def load_dense_tracker(checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['model_family'] == 'CADTrackDenseABC-R3'
    tracker = DenseABCTracker(payload['config']['pretrained'], payload['config']['intervention_threshold'])
    tracker.load_payload(payload)
    return tracker, payload
