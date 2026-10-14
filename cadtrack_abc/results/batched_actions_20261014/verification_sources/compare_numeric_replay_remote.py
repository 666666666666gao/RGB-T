from pathlib import Path
import json
import sys

project = Path('/home/gaob/CADTrack-ABC')
sys.path.insert(0, str(project))
import torch
from cadtrack_abc.data import read_pair, crop_batch

torch.set_num_threads(4)
root = project / 'outputs/batched_actions_numeric_replay_20261014'
left = torch.load(root / 'batch1/traced_replay.pt', map_location='cpu', weights_only=False)
right = torch.load(root / 'batch64/traced_replay.pt', map_location='cpu', weights_only=False)
receipts = [json.loads((root / f'batch{batch}/replay_complete.json').read_text()) for batch in (1, 64)]
assert left['teacher_sha256'] == right['teacher_sha256'] and left['teacher_epoch'] == right['teacher_epoch'] == 2
assert len(left['packets']) == len(right['packets']) == 2
original = project / 'outputs/batched_actions_probe_20261014'
serial_receipt = json.loads((original / 'serial/collection_complete.json').read_text())
assert serial_receipt['queries'] == 2 and serial_receipt['teacher_sha256'] == left['teacher_sha256']
assert not (original / 'initial_commit_parity.json').exists()
assert not (original / 'batch64/collection_complete.json').exists()
config = json.loads((project / 'cadtrack_abc/config.json').read_text())
sequence = Path(config['train_root']) / receipts[0]['sequence']
rgb, tir = sorted((sequence / 'visible').iterdir()), sorted((sequence / 'infrared').iterdir())
images = {frame: read_pair(rgb[frame], tir[frame]) for frame in range(5, 9)}
max_errors, failures, crop_counts = {}, [], dict(local=0, extra=0, template=0)
crop_mismatches = dict(local=0, extra=0, template=0)
decision_count, checked_actions = 0, 0

def numeric(a, b, key, identity):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    error = float((a - b).abs().max())
    max_errors[key] = max(max_errors.get(key, 0.), error)
    if not torch.allclose(a, b, rtol=1e-4, atol=1e-4):
        failures.append(dict(type='numeric', key=key, identity=identity, max_abs=error))

def crop(a, b, factor, size, kind, image, identity):
    first, first_transform = crop_batch([image], a, factor, size, 'cpu')
    second, second_transform = crop_batch([image], b, factor, size, 'cpu')
    crop_counts[kind] += 1
    if not (torch.equal(first_transform, second_transform) and torch.equal(first, second)):
        crop_mismatches[kind] += 1
        failures.append(dict(type='crop', kind=kind, identity=identity,
                             transforms=[first_transform.tolist(), second_transform.tolist()],
                             max_pixel_error=float((first-second).abs().max())))

for index, (a, b) in enumerate(zip(left['packets'], right['packets'])):
    assert a['frame'] == b['frame'] == [4, 5][index]
    assert a['actions'] == b['actions'] and len(a['actions']) == 210
    assert len(a['labels']) == len(b['labels']) == len(a['future_traces']) == len(b['future_traces']) == 210
    record = serial_receipt['records'][index]
    old = torch.load(original / 'serial' / record['file'], map_location='cpu', weights_only=False)
    assert old['metadata']['frame'] == a['frame']
    reference = tuple(old['metadata']['reference_action'])
    reference_index = a['actions'].index(reference)
    base_a, base_b = a['labels'][reference_index]['utility'], b['labels'][reference_index]['utility']
    for action, label_a, label_b, trace_a, trace_b in zip(a['actions'], a['labels'], b['labels'], a['future_traces'], b['future_traces']):
        identity = [a['frame'], *action]
        assert (label_a['context'], label_a['candidate'], label_a['mode']) == action
        assert (label_b['context'], label_b['candidate'], label_b['mode']) == action
        assert set(label_a) == set(label_b)
        assert torch.equal(label_a['future_valid'], label_b['future_valid'])
        for key in set(label_a) - {'context', 'candidate', 'mode', 'future_valid'}:
            numeric(label_a[key], label_b[key], key, identity)
        advantage_a, advantage_b = label_a['utility'] - base_a, label_b['utility'] - base_b
        numeric(advantage_a, advantage_b, 'advantage', identity)
        if bool(advantage_a < -.05) != bool(advantage_b < -.05):
            failures.append(dict(type='harm', identity=identity))
        if action == reference:
            assert float(advantage_a) == float(advantage_b) == 0.
        assert len(trace_a) == len(trace_b) == 3
        for step_a, step_b in zip(trace_a, trace_b):
            assert step_a['frame'] == step_b['frame']
            step_id = [*identity, step_a['frame']]
            for key in ('choice', 'write', 'template_commit', 'target_memory_commit', 'geometry_commit', 'extra_forwards'):
                decision_count += 1
                if not torch.equal(step_a[key], step_b[key]):
                    failures.append(dict(type='decision', key=key, identity=step_id,
                                         values=[step_a[key].tolist(),step_b[key].tolist()]))
            numeric(step_a['box'], step_b['box'], 'future_output_box', step_id)
            crop(step_a['local_crop_box'], step_b['local_crop_box'], 4., 256, 'local', images[step_a['frame']], step_id)
            if bool(step_a['extra_forwards']) and bool(step_b['extra_forwards']):
                crop(step_a['extra_crop_box'], step_b['extra_crop_box'], 6., 256, 'extra', images[step_a['frame']], step_id)
            if bool(step_a['template_commit'].any()) and bool(step_b['template_commit'].any()):
                crop(step_a['box'], step_b['box'], 2., 128, 'template', images[step_a['frame']], step_id)
        checked_actions += 1
assert checked_actions == 420 and decision_count == 7560
assert not torch.cuda.is_initialized()
result = dict(status='BOUNDED_LABEL_AND_DECISION_PARITY_PASS' if not failures else 'BOUNDED_PARITY_FAILED',
              queries=2, actions=420, horizon=3, future_steps=1260, decision_fields_checked=decision_count,
              maximum_error=max_errors, crop_counts=crop_counts, crop_mismatches=crop_mismatches,
              failures_count=len(failures), first_failures=failures[:30],
              teacher_sha256=left['teacher_sha256'], initial_commit_checks=receipts[1]['initial_commit_checks'],
              mixed_diagnostics=receipts[1]['mixed_diagnostics'], original_1e5_check_status='FAILED_PRESERVED',
              comparison_rtol=1e-4, comparison_atol=1e-4, updates=0, cuda_initialized=False,
              speedup_claim=False, peak_cuda_allocated_bytes=[r['peak_cuda_allocated_bytes'] for r in receipts],
              scope='Bounded two TRAIN queries, all210 actions and H3; not bitwise NN state equality or a global guarantee.')
(root / 'comparison_harm05.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('mixed_diagnostics','initial_commit_checks')}, indent=2))
