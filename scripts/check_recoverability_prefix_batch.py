"""Real batch-16 C1 versus zero-ABC prefix parity; no optimization."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from research import collect_recoverability as collector
from research.recoverability_modules import DECISION_FIELDS, RecoverabilityModules
from research.temporal_modules import TemporalModules
from research.train_recoverability import LABEL_FIELDS
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial-c1-prefix', action='store_true')
    parser.add_argument('--output', default='/data/gb/outputs/recoverability_prefix_batch_m0_20261003')
    options = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load('/data/wangwj/dataset/LasHeR',
        'trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    indices = {dataset[i].get_name(): i for i in range(len(dataset))}
    config = json.loads(Path('/data/gb/outputs/recoverability_train_s42_20261003/config.json').read_text())
    jobs, names = [], set()
    for job in config['jobs']:
        name = job['sequence']
        labels = dataset[indices[name]].get_all_bounding_boxes()[[0, 17, 18, 19, 20]]
        if name not in names and np.isfinite(labels).all() and (labels[:, 2:] > labels[:, :2]).all():
            jobs.append((indices[name], 17))
            names.add(name)
        if len(jobs) == 16:
            break
    assert len(jobs) == 16
    c1 = torch.load(config['head'], map_location='cpu', weights_only=False)
    zero = RecoverabilityModules(c1).to(device).eval().requires_grad_(False)
    extractor = collector.InstanceExtractor(config['pretrained'], 5, .45, c1['args']['nms_iou']).to(device)
    old = torch.load(Path(config['motion_run']) / 'last.pth', map_location='cpu', weights_only=False)
    old_config = json.loads((Path(config['motion_run']) / 'config.json').read_text())
    motion = TemporalModules(c1, old_config['slots'], old_config['motion_history'], old_config['modes'], old['horizon']).to(device)
    motion.load_state_dict(old['model'], strict=True)
    motion.eval().requires_grad_(False)
    args = SimpleNamespace(max_prefix=64, forward_batch=64, prefix_threshold=.03,
                           serial_c1_prefix=options.serial_c1_prefix)
    started = time.perf_counter()
    baseline = collector.collect_batch(jobs, dataset, extractor, zero.c1, motion, device, args)
    alternative = collector.collect_batch(jobs, dataset, extractor, zero.c1, motion, device, args, zero)
    fields = {}
    for key in DECISION_FIELDS + LABEL_FIELDS:
        a = np.stack([row[key] for row in baseline])
        b = np.stack([row[key] for row in alternative])
        fields[key] = {'bitwise_equal': bool(np.array_equal(a, b)),
                       'different_values': int((a != b).sum()),
                       'max_absolute_difference': float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))}
    receipt = {'completed': True, 'optimization_updates': 0, 'official_accuracy': False,
               'jobs': [{'sequence': dataset[i].get_name(), 'query_frame': q} for i, q in jobs],
               'batch_clips': 16, 'forward_batch': 64,
               'serial_c1_prefix': options.serial_c1_prefix,
               'all_shared_fields_bitwise_equal': all(v['bitwise_equal'] for v in fields.values()),
               'fields': fields, 'elapsed_seconds': time.perf_counter() - started,
               'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20}
    out = Path(options.output)
    out.mkdir(exist_ok=True)
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
