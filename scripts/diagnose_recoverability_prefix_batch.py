"""Locate batch-prefix divergence and test serial C1 without production edits."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from research import collect_recoverability as collector
from research.bounded_recovery import BoundedRecoveryTracker
from research.collect_rollouts import observe_actions
from research.recoverability_modules import DECISION_FIELDS, RecoverabilityModules
from research.temporal_modules import TemporalModules
from research.train_recoverability import LABEL_FIELDS
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


def differences(a, b):
    return {'bitwise_equal': bool(np.array_equal(a, b)), 'different_values': int((a != b).sum()),
            'max_absolute_difference': float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))}


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    previous = json.loads(Path('/data/gb/outputs/recoverability_prefix_batch_m0_20261003/receipt.json').read_text())
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load('/data/wangwj/dataset/LasHeR',
        'trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    indices = {dataset[i].get_name(): i for i in range(len(dataset))}
    jobs = [(indices[j['sequence']], j['query_frame']) for j in previous['jobs']]
    assert len(jobs) == 16 and all(q == 17 for _, q in jobs)
    config = json.loads(Path('/data/gb/outputs/recoverability_train_s42_20261003/config.json').read_text())
    c1 = torch.load(config['head'], map_location='cpu', weights_only=False)
    zero = RecoverabilityModules(c1).to(device).eval().requires_grad_(False)
    extractor = collector.InstanceExtractor(config['pretrained'], 5, .45, c1['args']['nms_iou']).to(device)
    old = torch.load(Path(config['motion_run']) / 'last.pth', map_location='cpu', weights_only=False)
    old_config = json.loads((Path(config['motion_run']) / 'config.json').read_text())
    motion = TemporalModules(c1, old_config['slots'], old_config['motion_history'], old_config['modes'], old['horizon']).to(device)
    motion.load_state_dict(old['model'], strict=True)
    motion.eval().requires_grad_(False)
    items = []
    for index, _ in jobs:
        sequence = dataset[index]
        tracker = BoundedRecoveryTracker(extractor, zero.c1, collector.read_pair(*collector.paths(sequence, 0), device),
                                         sequence[0].get_bounding_box().copy(), torch.float16)
        items.append((tracker, tracker.branches[0], collector.read_pair(*collector.paths(sequence, 1), device)))
    inputs, outputs = [], []
    def capture_input(module, args):
        inputs.append({key: value.cpu().numpy().copy() for key, value in args[0].items()})
    def capture_output(module, args, output):
        outputs.append({key: value.float().cpu().numpy().copy() for key, value in output.items()})
    hooks = [extractor.register_forward_pre_hook(capture_input), extractor.base.head.register_forward_hook(capture_output)]
    started = time.perf_counter()
    batch_observations = observe_actions(items, extractor, zero.c1)
    single_observations = [observe_actions([item], extractor, zero.c1)[0] for item in items]
    for hook in hooks:
        hook.remove()
    input_checks = {key: differences(inputs[0][key], np.concatenate([x[key] for x in inputs[1:]])) for key in inputs[0]}
    output_checks = {key: differences(outputs[0][key], np.concatenate([x[key] for x in outputs[1:]])) for key in outputs[0]}
    choice_changes = sum(a[3] != b[3] for a, b in zip(batch_observations, single_observations))
    first_boxes = differences(np.stack([a[2] for a in batch_observations]), np.stack([a[2] for a in single_observations]))
    raw_arrays = {f'batch_input_{key}': value for key, value in inputs[0].items()}
    raw_arrays.update({f'single_input_{key}': np.concatenate([x[key] for x in inputs[1:]]) for key in inputs[0]})
    raw_arrays.update({f'batch_output_{key}': value for key, value in outputs[0].items()})
    raw_arrays.update({f'single_output_{key}': np.concatenate([x[key] for x in outputs[1:]]) for key in outputs[0]})
    calls = 0
    original_observe = collector.observe_actions
    def serial_prefix(items, extractor, head):
        nonlocal calls
        calls += 1
        return [original_observe([item], extractor, head)[0] for item in items] if calls <= 16 else original_observe(items, extractor, head)
    collector.observe_actions = serial_prefix
    args = SimpleNamespace(max_prefix=64, forward_batch=64, prefix_threshold=.03, serial_c1_prefix=False)
    baseline = collector.collect_batch(jobs, dataset, extractor, zero.c1, motion, device, args)
    collector.observe_actions = original_observe
    alternative = collector.collect_batch(jobs, dataset, extractor, zero.c1, motion, device, args, zero)
    shared = {key: differences(np.stack([r[key] for r in baseline]), np.stack([r[key] for r in alternative]))
              for key in DECISION_FIELDS + LABEL_FIELDS}
    counts = np.stack([row['prefix_counts'] for row in alternative]).sum(0).tolist()
    result = {'completed': True, 'optimization_updates': 0, 'official_accuracy': False,
              'jobs': previous['jobs'], 'first_frame_input_checks': input_checks, 'first_frame_output_checks': output_checks,
              'first_frame_c1_choice_changes': choice_changes, 'first_frame_decoded_boxes': first_boxes,
              'serial_prefix_fields': shared, 'serial_prefix_all_shared_fields_bitwise_equal': all(x['bitwise_equal'] for x in shared.values()),
              'zero_prefix_intervention_counts': counts, 'elapsed_seconds': time.perf_counter()-started,
              'peak_cuda_mib': torch.cuda.max_memory_allocated(device)/2**20,
              'scope': 'Same16TRAIN witness; serializes only first16 prefix observation calls, query/extra/future batching unchanged; production code untouched'}
    out = Path('/data/gb/outputs/recoverability_prefix_batch_diagnosis_20261003')
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / 'first_frame_raw_arrays.npz', **raw_arrays)
    (out / 'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
