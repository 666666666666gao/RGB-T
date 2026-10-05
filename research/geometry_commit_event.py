"""Fixed-head control matching the frozen-old4 continuation used by its labels."""
import torch

from .geometry_commit import GeometryCommitTracker
from .recoverability_tracker import RecoverabilityTracker


class GeometryCommitEventTracker(GeometryCommitTracker):
    def __init__(self, *args, continuation_horizon, **kwargs):
        super().__init__(*args, **kwargs)
        assert continuation_horizon in (3, 32)
        self.continuation_horizon = continuation_horizon
        self.old4_until_frame = 0
        self.stats['old4_continuation_frames'] = 0
        self.stats['commit_events'] = 0

    def accept_observation(self, image, observation, pause):
        if self.frame <= self.old4_until_frame:
            self.stats['old4_continuation_frames'] += 1
            return RecoverabilityTracker.accept_observation(self, image, observation, pause)
        result = super().accept_observation(image, observation, pause)
        if self.commit_action is not None and self.commit_action != int(pause):
            self.old4_until_frame = self.frame + self.continuation_horizon
            self.stats['commit_events'] += 1
        return result

    @torch.inference_mode()
    def step(self, image):
        before = self.stats['old4_continuation_frames']
        box = super().step(image)
        self.last_decision['old4_continuation'] = torch.tensor(
            self.stats['old4_continuation_frames'] > before, device=self.device)
        self.last_decision['old4_until_frame'] = torch.tensor(self.old4_until_frame, device=self.device)
        return box
