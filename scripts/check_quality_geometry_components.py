"""CPU checks of the actual query choice/geometry helpers; no model forward."""
import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from research.probe_quality_geometry_commit import quality_extra_choice, use_baseline_geometry


def main():
    valid = torch.zeros((1, 7, 5), dtype=torch.bool)
    valid[:, :2] = True
    quality = torch.full((1, 7, 5), .1)
    quality[:, 0, 0] = .3
    quality[:, 0, 1] = .99  # Must not reorder local C/Hann choices.
    quality[:, 1, 2] = .8
    data = {'valid': valid}
    assert quality_extra_choice(data, torch.logit(quality), torch.tensor([0])).item() == 7
    data['valid'][:, 1] = False
    assert quality_extra_choice(data, torch.logit(quality), torch.tensor([0])).item() == 0
    old = np.array([10., 10., 20., 20.])
    current = old + np.array([5., 0., 5., 0.])
    replacement = np.array([70., 10., 80., 20.])
    template = torch.arange(4.)
    memory = torch.ones((1, 4, 2, 128))
    tracker = SimpleNamespace(frame=2, branch=SimpleNamespace(box=replacement.copy(),
                              search_box=replacement.copy(), template=template),
                              history=deque([(old.copy(), 1, .9), (replacement.copy(), 2, .8)]),
                              identity_memory=memory, motion_memory=memory.clone())
    baseline = SimpleNamespace(frame=2, branch=SimpleNamespace(search_box=current.copy()),
                               history=deque([(old.copy(), 1, .9), (current.copy(), 2, .4)]))
    before_memory = tracker.motion_memory.clone()
    use_baseline_geometry(tracker, baseline)
    assert np.array_equal(tracker.branch.box, replacement)
    assert np.array_equal(tracker.branch.search_box, current)
    assert not np.array_equal(tracker.branch.search_box, old)
    assert tracker.history[-1][1:] == (2, .4)
    assert np.array_equal(tracker.history[-1][0], current)
    assert tracker.branch.template is template and tracker.identity_memory is memory
    assert torch.equal(tracker.motion_memory, before_memory)
    print(json.dumps({'status': 'PASS', 'cases': ['extra-only choice', 'no-extra baseline',
                      'output/appearance unchanged', 'current-time geometry advances', 'history-quality provenance'],
                      'execution': {'model_forward_calls': 0, 'optimizer_steps': 0, 'GPU_queries': 0},
                      'limitation': 'Helpers only; full query lifecycle/raw parity and NN rollouts not executed.'}))


if __name__ == '__main__':
    main()
