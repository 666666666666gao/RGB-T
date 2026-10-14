"""Run once only after current four label shards and original drivers finish."""
from pathlib import Path
import json
import os
import subprocess
import sys

project = Path('/home/gaob/CADTrack-ABC')
sys.path.insert(0, str(project))
from cadtrack_abc.future_pipeline import validate_collection

collection = project / 'outputs/native_action_collection_20261014'
sampling = project / 'outputs/native_future_sampling_20261014'
launch = json.loads((collection / 'launch.json').read_text())
receipts = []
for split in ['train', 'validation']:
    folders = [collection / ('collection_' + split) / f'shard{shard}' for shard in range(2)]
    receipts.extend(validate_collection(folders, sampling / ('queries_' + split), launch['teacher_sha256']))
assert sum(row['queries'] for row in receipts) == 5697
assert all(not (Path('/proc') / str(job['pid'])).exists() for job in launch['jobs'])
memory = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, check=True).stdout
assert all(int(line.split(',')[1]) < 500 for line in memory.splitlines()), memory
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torch
from cadtrack_abc.counterfactual import advance_candidate, assert_same_state, clone_state, observe_query
from cadtrack_abc.data import read_pair
from cadtrack_abc.future_actions import context_baseline
from cadtrack_abc.model import CADTrackABC
from cadtrack_abc.search_actions import CADTrackSearchActionPolicy

torch.set_num_threads(4)
config = json.loads((project / 'cadtrack_abc/config.json').read_text())
payload = torch.load(collection / 'teacher.pth', map_location='cpu', weights_only=False)
model = CADTrackSearchActionPolicy(config['pretrained'], config['threshold']).cuda().eval().requires_grad_(False)
model.abc.load_state_dict(payload['abc'], strict=True)
directory = Path(config['train_root']) / '10crosswhite'
rgb, tir = sorted((directory / 'visible').iterdir()), sorted((directory / 'infrared').iterdir())
initial = np.fromstring((directory / 'init.txt').read_text().splitlines()[0], sep=',', dtype=np.float32)
with torch.inference_mode():
    state = model.initialize([read_pair(rgb[0], tir[0])], torch.tensor(initial[None]), torch.tensor([0]))
    for frame in range(1, 4):
        _, state = CADTrackABC.step(model, [read_pair(rgb[frame], tir[frame])], torch.tensor([frame]), state)
    images, times = [read_pair(rgb[4], tir[4])], torch.tensor([4])
    native, native_state = CADTrackABC.step(model, images, times, clone_state(state))
    neutral, neutral_state = model.step(images, times, clone_state(state))
    assert native['extra_forwards'] == neutral['extra_forwards'] == 0
    assert torch.equal(native['box'], neutral['box']) and torch.equal(native['choice'], neutral['choice'])
    assert_same_state(native_state, neutral_state)
    query = observe_query(model, images, clone_state(state))
    rank = 1
    model.search_action.network[-1].bias[:3].copy_(torch.tensor([-1., .5, -1.], device=model.device))
    forwards = []
    hook = model.base.register_forward_hook(lambda module, inputs, output: forwards.append(1))
    searched, searched_state = model.step(images, times, clone_state(state))
    hook.remove()
    context = rank + 1
    baseline = int(context_baseline(dict(query['candidates'][context], **query['scored'][context]), .03)[0])
    expected, expected_state = advance_candidate(model, images, times, clone_state(state), query, context, baseline, 'policy')
    assert searched['extra_forwards'] == 1 and len(forwards) == 2
    assert searched['search_region_rank'] == rank and searched['search_region_mode'] == query['motion_mode_order'][rank]
    assert torch.equal(searched['pair']['boxes'], query['candidates'][context]['boxes'])
    assert torch.equal(searched['box'], expected['box']) and int(searched['choice'][0]) == baseline
    assert_same_state(searched_state, expected_state)
out = project / 'outputs/search_action_visual_sanity_20261014'
out.mkdir(exist_ok=False)
result = dict(status='ACTUAL_GPU_NEUTRAL_AND_PROBABILITY_RANK1_VISUAL_STATE_SANITY_PASS',
              physical_gpu=2, teacher_sha256=launch['teacher_sha256'], sequence='10crosswhite', frame=4,
              zero_heads_native_box_choice_and_all9_state_fields_equal=True,
              probability_rank_test=rank, actual_motion_mode=searched['search_region_mode'],
              actual_visual_forwards=2, at_most_one_extra_visual_forward=True,
              searched_candidate_box_and_all9_state_fields_equal=True, optimizer_updates=0,
              full_training_started=False, formal_metrics=None,
              scope='One real TRAIN state; probability-rank1 forced only as routing fixture. Not trained search evidence or performance.')
(out / 'sanity_complete.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
