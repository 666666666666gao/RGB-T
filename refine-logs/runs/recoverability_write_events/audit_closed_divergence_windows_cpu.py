"""Read six sealed internal timelines and actual TRAIN GT, preserving unknown windows."""
import json
from pathlib import Path

import numpy as np

from research.collect_candidate_metrics import ground_truth
from research.collect_core_metrics import localization_quality

folder = Path('/data/gb/GOLA/refine-logs/runs/recoverability_write_events')
analysis = json.loads((folder / 'actual_all16_full98_paired_analysis.json').read_text())
names = [row['sequence'] for row in analysis['selected_worst10_vs_old4'][:3] + analysis['selected_best10_vs_old4'][:3]]
newroot = Path(analysis['selected']['model']).parent / 'continuous_epoch_004/predictions'
oldroot = Path('/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/predictions')
records = []
for name in names:
    gt = ground_truth('/data/wangwj/dataset/LasHeR/traingset', name, 'lasher')
    npred = np.loadtxt(newroot / (name + '.txt'), ndmin=2)
    opred = np.loadtxt(oldroot / (name + '.txt'), ndmin=2)
    assert npred.shape == opred.shape == gt.shape
    nq, known = localization_quality(npred, gt, 'lasher')
    oq, oknown = localization_quality(opred, gt, 'lasher')
    assert np.array_equal(known, oknown)
    first = int(np.flatnonzero(np.any(npred != opred, axis=1))[0])
    assert first > 0
    traces = {}
    for label, path in [('new', newroot), ('old4', oldroot)]:
        with np.load(path / (name + '_recoverability_decisions.npz')) as timeline:
            trace = []
            for frame in range(max(1, first - 2), min(len(gt), first + 4)):
                i = frame - 1
                choice = int(timeline['choice'][i])
                region, candidate = divmod(choice, 5)
                trace.append({'frame_zero_based': frame, 'choice': choice,
                              'original_choice': int(timeline['original_choice'][i]),
                              'pause': bool(timeline['pause'][i]),
                              'template_updated': bool(timeline['template_updated'][i]),
                              'template_source_frame': int(timeline['template_source_frame'][i]),
                              'forecast_reference': timeline['forecast_reference'][i].tolist(),
                              'selected_current_quality': float(timeline['current_quality'][i, region, candidate]),
                              'selected_predicted_advantage': timeline['predicted_advantage'][i, region, candidate].tolist()})
            traces[label] = trace
    windows = []
    for horizon in (0, 3, 32):
        end = min(len(gt), first + horizon + 1)
        valid = known[first:end]
        count = int(valid.sum())
        windows.append({'horizon_requested': horizon, 'observed_frames': end - first,
                        'valid_GT_frames': count, 'quality_status': 'KNOWN' if count else 'UNKNOWN_NO_VALID_GT',
                        'new_mean_iou': float(nq[first:end][valid].mean()) if count else None,
                        'old4_mean_iou': float(oq[first:end][valid].mean()) if count else None,
                        'new_failed_frames': int(((nq[first:end] < .2) & valid).sum()),
                        'old4_failed_frames': int(((oq[first:end] < .2) & valid).sum())})
    records.append({'sequence': name, 'first_output_difference_frame_zero_based': first, 'frames': len(gt),
                    'new_sequence_mean_iou': float(nq[known].mean()), 'old4_sequence_mean_iou': float(oq[known].mean()),
                    'windows_from_first_output_difference': windows, 'nearby_actual_decisions': traces})
print(json.dumps({'completed': True, 'neural_forward_calls': 0, 'optimizer_steps': 0, 'native_TEST_data_read': False,
                  'scope': 'Six cases selected from complete internal98. Distinct policy trajectories, not matched-state causal evidence. GT-unknown windows are null, never zero or a labeled failure; valid later GT only provides retrospective diagnostics.',
                  'records': records}, allow_nan=False))
