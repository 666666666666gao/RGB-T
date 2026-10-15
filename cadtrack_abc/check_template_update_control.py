"""CPU synthetic state/source checks, without a visual model or optimizer."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from .counterfactual import assert_same_state, clone_state
from .data import crop_batch
from .score_template_update_control import summarize_sources
from .template_update_control import R1TemplateUpdateControl


class ParentFixture:
    device = 'cpu'

    def __init__(self, local=.9, selected=.7):
        self.local, self.selected = local, selected

    def step(self, images, frames, state):
        advanced = clone_state(state)
        for key in ('box', 'anchor', 'memory', 'history', 'frames', 'quality', 'age'):
            advanced[key] += 1
        advanced['templates'][1] = torch.full_like(state['templates'][1], 9.)
        boxes = torch.tensor([[[12., 16., 32., 36.], [2., 4., 22., 24.], [30., 2., 50., 22.],
                               [2., 40., 22., 60.], [30., 30., 50., 50.], [70., 2., 90., 22.],
                               [90., 2., 110., 22.], [48., 50., 68., 70.], [70., 90., 90., 110.],
                               [90., 90., 110., 110.]]])
        hann = torch.tensor([[self.local, .1, .2, .3, .4, .1, .2, self.selected, .3, .4]])
        observation = dict(box=torch.tensor([[48., 50., 20., 20.]]), choice=torch.tensor([7]),
                           pair=dict(boxes=boxes, hann=hann),
                           local=dict(boxes=boxes[:, :5], hann=hann[:, :5]),
                           write=torch.tensor([True]), template_commit=torch.tensor([[True, False]]))
        return observation, advanced


def fixture(age=4):
    template = torch.full((1, 6, 128, 128), -2.)
    return dict(templates=[template, template], box=torch.tensor([[12., 16., 20., 20.]]),
                anchor=torch.zeros(1), memory=torch.zeros(1), history=torch.zeros(1),
                frames=torch.zeros(1, dtype=torch.long), valid=torch.ones(1, dtype=torch.bool),
                quality=torch.zeros(1), age=torch.tensor([age]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    image = np.arange(128 * 128 * 6, dtype=np.uint32).reshape(128, 128, 6).astype(np.uint8)
    images, frames, state, checks = [image], torch.tensor([5]), fixture(), []
    prior = clone_state(state)
    parent = ParentFixture()
    native_observation, native_state = parent.step(images, frames, state)
    observed, advanced = R1TemplateUpdateControl(parent, 'current').step(images, frames, state)
    assert_same_state(advanced, native_state)
    assert torch.equal(observed['box'], native_observation['box'])
    assert torch.equal(observed['template_commit'], native_observation['template_commit'])
    checks.append('current preserves exact native state, output and modal writes')
    for variant in ('static', 'cad_selected', 'cad_local'):
        observed, advanced = R1TemplateUpdateControl(parent, variant).step(images, frames, state)
        assert torch.equal(advanced['templates'][0], prior['templates'][0])
        assert torch.equal(observed['box'], native_observation['box'])
        for key in state.keys() - {'templates'}:
            assert torch.equal(advanced[key], native_state[key]), (variant, key)
        if variant == 'static':
            assert torch.equal(advanced['templates'][1], prior['templates'][1])
            assert not observed['write'].any() and not observed['template_commit'].any()
        else:
            assert observed['template_commit'].all()
            box = torch.tensor([[48., 50., 20., 20.]]) if variant == 'cad_selected' else torch.tensor([[12., 16., 20., 20.]])
            expected, _ = crop_batch(images, box, 2., 128, 'cpu')
            assert torch.equal(advanced['templates'][1], expected)
            assert observed['template_control']['source_box_xywh'] == box[0].tolist()
        checks.append(variant + ' changes only dynamic image template and preserves matched score/box source')
    assert_same_state(state, prior)
    checks.append('input state and shared initial/dynamic template alias remain unchanged')
    for score, age in ((.65, 4), (.64, 4), (.9, 3)):
        observed, advanced = R1TemplateUpdateControl(ParentFixture(selected=score), 'cad_selected').step(images, frames, fixture(age))
        assert not observed['write'].any() and torch.equal(advanced['templates'][1], prior['templates'][1])
    checks.append('strict threshold and every-five-frame schedule')
    for selected_score, local_score, selected_write, local_write in ((.64, .9, False, True), (.9, .64, True, False)):
        parent = ParentFixture(local_score, selected_score)
        selected, _ = R1TemplateUpdateControl(parent, 'cad_selected').step(images, frames, state)
        local, _ = R1TemplateUpdateControl(parent, 'cad_local').step(images, frames, state)
        assert bool(selected['write'][0]) == selected_write and bool(local['write'][0]) == local_write
    checks.append('a high score from one candidate never certifies another candidate')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 's').mkdir()
        np.savetxt(root / 's/init.txt', [[0, 0, 20, 20], [12, 16, 20, 20]], delimiter=',')
        record = dict(template_commit_rgb_tir=[True, True], template_write_requested=True,
                      template_control=dict(source_box_xywh=[12, 16, 20, 20]))
        (root / 's_mechanisms.json').write_text(json.dumps([record]))
        result = summarize_sources(root, root, ['s'], 'lasher')
        assert result['counts']['incorrect_source_commit_frames'] == 0
        record['template_control']['source_box_xywh'] = [80, 80, 20, 20]
        (root / 's_mechanisms.json').write_text(json.dumps([record]))
        result = summarize_sources(root, root, ['s'], 'lasher')
        assert result['counts']['incorrect_source_commit_frames'] == 1
    checks.append('template pollution uses template-source GT overlap, independently of output box')
    assert not torch.cuda.is_initialized()
    result = dict(status='CPU_SYNTHETIC_CHECKS_PASS', checks=checks, count=len(checks),
                  visual_model_loaded=False, optimizer_updates=0, real_video_frames=0,
                  cuda_initialized=False, formal_metrics=None)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
