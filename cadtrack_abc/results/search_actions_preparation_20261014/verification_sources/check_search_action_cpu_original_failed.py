"""Bounded real-packet head checks; mocked vision only tests runtime routing."""
from pathlib import Path
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from torch import nn

project = Path('/home/gaob/CADTrack-ABC')
sys.path.insert(0, str(project))
from cadtrack_abc.future_actions import FutureActionHead, objective, context_baseline
from cadtrack_abc.future_data import FuturePackets, collate_packets
from cadtrack_abc.train_future_actions import batch_loss
from cadtrack_abc import search_actions as sa

torch.set_num_threads(4)
torch.manual_seed(42)
source = project / 'outputs/actual_batch64_collector_20261014/packets'
teacher = json.loads((source / 'collection_complete.json').read_text())['teacher_sha256']
manifest = json.loads((project / 'outputs/cad_abc_20261013/manifest.json').read_text())
dataset = FuturePackets(source / 'collection_complete.json', manifest, 'train', teacher, .03, 'native')
assert len(dataset) == 2
batch = collate_packets([dataset[0], dataset[1]])
assert batch['search_input'].shape == (2, 295)
packet = torch.load(source / dataset.records[0]['file'], map_location='cpu', weights_only=False)
decision = packet['decision']
local = decision['contexts'][0]
summary = torch.cat((local['tokens'].mean(1), local['tokens'].max(1).values,
                     decision['motion']['means'].flatten(1), decision['motion']['log_weights']), -1)
assert torch.equal(summary, decision['search_input'])
head, search = FutureActionHead(), sa.SearchActionHead()
assert bool((search(batch['search_input'])['scores'] <= .04).all())
reference_losses, achieved = [], []
for features, baseline, utility in batch['contexts']:
    output = head(features, baseline)
    action = sa.selected_action(output, baseline, .03)
    assert torch.equal(action, baseline * 6)
    achieved.append(utility.flatten(1).gather(1, action[:, None]).squeeze(1))
    reference_losses.append(objective(output, utility, baseline)[0])
assert torch.equal(batch_loss(head, batch, torch.device('cpu')), torch.stack(reference_losses).mean())
search_loss = sa.search_objective(search(batch['search_input']), torch.stack(achieved, 1))
assert all(g is None for g in torch.autograd.grad(search_loss, tuple(head.parameters()), allow_unused=True))
modules = nn.ModuleDict({'future_action': head, 'search_action': search})
before = {name: value.detach().clone() for name, value in modules.named_parameters()}
optimizer = torch.optim.AdamW(modules.parameters(), lr=1e-4)
losses = []
for _ in range(2):
    optimizer.zero_grad(set_to_none=True)
    loss = batch_loss(head, batch, torch.device('cpu'), search, .03)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in modules.parameters())
    optimizer.step()
    losses.append(float(loss.detach()))
changed = [name for name, value in modules.named_parameters() if not torch.equal(before[name], value)]
assert any(name.startswith('future_action.') for name in changed)
assert any(name.startswith('search_action.') for name in changed)

# Only route/candidate/context boundaries are mocked. This is not visual parity.
calls = []
rank_to_execute = 1
chosen_context = decision['contexts'][rank_to_execute + 1]
keys = ['boxes', 'raw', 'hann', 'instance', 'fused']
local_candidate = {key: local[key] for key in keys}
extra_candidate = {key: chosen_context[key][:, 5:] for key in keys}
encoded_keys = ['tokens', 'descriptors', 'support', 'gate', 'evidence']

class MockABC(nn.Module):
    def __init__(self):
        super().__init__()
        self.device_parameter = nn.Parameter(torch.zeros(()))

    def motion(self, *args):
        return decision['motion']

    def encode(self, candidate, state, motion, source):
        return {key: local[key] if source == 0. else chosen_context[key][:, 5:] for key in encoded_keys}

    def score(self, encoded, candidate):
        context = local if candidate['boxes'].shape[1] == 5 else chosen_context
        return {key: value for key, value in context.items() if key not in keys}

class MockBase(nn.Module):
    def forward(self, templates, patch):
        calls.append('visual')
        return local_candidate if len([c for c in calls if c == 'visual']) == 1 else extra_candidate

class FixedSearch(nn.Module):
    def forward(self, features):
        assert torch.equal(features, summary)
        calls.append('local_search_decision')
        return {'scores': features.new_tensor([[-1., .5, -1.]])}

def crop(images, boxes, factor, size, device):
    calls.append(['crop', factor, size, boxes.clone()])
    return boxes, boxes

def advance(model, images, frames, state, query, context, candidate, mode):
    calls.append(['commit', context, candidate, mode])
    assert context == 1
    assert torch.equal(query['candidates'][1]['boxes'], chosen_context['boxes'])
    chosen = query['candidates'][1]['boxes'][:, candidate]
    trace = dict(box=chosen, template_commit=torch.zeros(1, 2, dtype=torch.bool),
                 target_memory_commit=torch.zeros(1, 2, dtype=torch.bool),
                 geometry_commit=torch.ones(1, dtype=torch.bool), extra_forwards=1)
    return trace, dict(state, box=chosen, age=state['age'] + 1)

sa.crop_batch = crop
sa.candidates = lambda features, transform, reference: features
sa.clip_boxes = lambda boxes, images: boxes
sa.advance_candidate = advance
policy = sa.CADTrackSearchActionPolicy.__new__(sa.CADTrackSearchActionPolicy)
nn.Module.__init__(policy)
policy.abc, policy.base = MockABC(), MockBase()
policy.future_action, policy.search_action = FutureActionHead(), FixedSearch()
policy.threshold = .03
state = {key: decision[key] for key in ['history', 'frames', 'valid', 'quality', 'anchor', 'memory']}
state.update(box=decision['motion']['reference'], age=torch.tensor([packet['metadata']['frame'] - 1]), templates=[])
observation, advanced = policy.step([None], torch.tensor([packet['metadata']['frame']]), state)
assert calls.count('visual') == 2
assert calls.index('local_search_decision') < [i for i, c in enumerate(calls) if c == 'visual'][1]
assert len([c for c in calls if isinstance(c, list) and c[0] == 'commit']) == 1
assert int(advanced['age']) == int(state['age']) + 1
assert observation['search_region_rank'] == rank_to_execute
assert observation['search_region_mode'] == decision['motion_mode_order'][rank_to_execute]
assert int(observation['choice'][0]) == int(context_baseline(chosen_context, .03)[0])
result = dict(status='BOUNDED_REAL_PACKET_CPU_HEAD_AND_MOCKED_VISION_ROUTE_CHECK_PASS',
              real_queries=2, action_labels=420, search_input_dim=295,
              zero_initial_searches=0, unchanged_control_loss_exact=True,
              detached_selector_targets=True, optimizer_updates=2, losses=losses,
              changed_tensors=changed, explicit_rank_test=rank_to_execute,
              mocked_visual_forwards=2, mocked_state_commits=1,
              trainable_parameters=sum(p.numel() for p in modules.parameters()),
              actual_gpu_visual_runtime_verified=False, full_fit_complete=False, formal_metrics=None,
              cuda_initialized=torch.cuda.is_initialized(),
              limitations='Head and routing checks only. Vision mocked from actual packets; no base model or full video was run.')
print(json.dumps(result, indent=2))
