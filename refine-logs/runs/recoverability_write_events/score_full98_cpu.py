"""Score full continuous developer videos against actual TRAIN GT; no native-test selection."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from research.collect_candidate_metrics import ground_truth
from research.collect_core_metrics import localization_quality

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--predictions', required=True)
p.add_argument('--model', required=True)
args = p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
root = Path(args.predictions)
cfg = json.loads((root / 'inference_config.json').read_text())
done = json.loads((root / 'inference_completion.json').read_text())
split = json.loads(Path('/data/gb/outputs/c1_initial_seed42/split.json').read_text())
names = sorted(split['validation'])
assert len(names) == 98 and not set(names) & set(split['train'])
assert done['completed'] and done['sequences'] == 98 and done['smoke_only']
assert cfg['model'] == args.model and cfg['validation_split'] == '/data/gb/outputs/c1_initial_seed42/split.json'
assert cfg['root'] == '/data/wangwj/dataset/LasHeR/traingset' and cfg['max_frames'] == cfg['limit_sequences'] == cfg['sequence_offset'] == 0
assert {p.stem for p in root.glob('*.txt')} == set(names)
rows = []
for name in names:
    gt = ground_truth(cfg['root'], name, 'lasher')
    prediction = np.loadtxt(root / (name + '.txt'), ndmin=2)
    assert prediction.shape == (len(gt), 4) and np.isfinite(prediction).all()
    actual = next(r for r in done['records'] if r['sequence'] == name)
    assert actual['frames'] == len(gt)
    q, known = localization_quality(prediction, gt, 'lasher')
    rows.append({'sequence': name, 'frames': len(gt), 'mean_iou': float(q[known].mean())})
assert sum(r['frames'] for r in rows) == done['frames'] == 49418
record = {'completed': True, 'model': args.model, 'epoch': cfg['head_epoch'], 'sequences': 98,
          'frames': 49418, 'sequence_mean_iou': float(np.mean([r['mean_iou'] for r in rows])),
          'per_sequence': rows, 'actual_TRAIN_GT_scored': True,
          'native_test_data_read': False, 'official_metrics_completed': False,
          'limitation': 'Repeatedly used developer videos; not an untouched validation holdout or LasHeR SR.'}
(root.parent / 'full98_selection_score.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k != 'per_sequence'}), flush=True)
