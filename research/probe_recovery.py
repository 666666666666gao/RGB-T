"""C2 functionality/online diagnostic on LasHeR TRAIN-held-out sequences only.

Full test datasets and future-utility training are deliberately separate stages.
Truncation is explicit; these IoUs are not official benchmark PR/SR.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .bounded_recovery import BoundedRecoveryTracker
from .candidate_learning import FrozenCandidateExtractor, CandidateQualityHead, box_iou
from .evaluate_online import read_pair, track_sequence
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    p.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    p.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json')
    p.add_argument('--pretrained', default='pretrained_models/gola_b224.bin')
    p.add_argument('--head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--output', required=True)
    p.add_argument('--sequences', type=int, default=2)
    p.add_argument('--sequence-offset', type=int, default=0, help='Disjoint held-out shards for parallel GPU jobs.')
    p.add_argument('--max-frames', type=int, default=128)
    p.add_argument('--branches', type=int, default=3)
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--switch-patience', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def run_recovery(visible, infrared, init_box, extractor, head, device, args, restore):
    image = read_pair(visible[0], infrared[0], device)
    tracker = BoundedRecoveryTracker(extractor, head, image, init_box, torch.float16,
                                    args.branches, args.window, args.switch_patience, restore)
    anchor_before = tracker.anchor.clone()
    predictions, latency, events = [init_box.copy()], [], []
    for frame in range(1, len(visible)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions.append(tracker.step(read_pair(visible[frame], infrared[frame], device)))
        torch.cuda.synchronize(device)
        latency.append(time.perf_counter() - started)
        events.append(tracker.last_event)
    assert torch.equal(tracker.anchor, anchor_before), 'First-frame anchor was changed'
    return np.array(predictions), np.array(latency), tracker.stats, events


def summarize(predictions, ground_truth):
    valid = np.isfinite(ground_truth).all(1) & (ground_truth[:, 2:] > ground_truth[:, :2]).all(1)
    valid[0] = False  # Initialization does not count as learned tracking quality.
    assert valid.any()
    quality = box_iou(torch.from_numpy(predictions), torch.from_numpy(ground_truth)).numpy()
    assert np.isfinite(predictions).all() and np.isfinite(quality[valid]).all()
    return quality, valid


def switch_diagnostics(events, ground_truth, valid):
    result = dict(valid_switch_events=0, harmful_switch_events=0, beneficial_switch_events=0)
    for event in events:
        t = event['frame']
        if not event['switched'] or not valid[t]:
            continue
        boxes = torch.tensor([branch['box'] for branch in event['branches']], dtype=torch.float64)
        quality = box_iou(boxes, torch.from_numpy(ground_truth[t])).numpy()
        by_id = {branch['id']: q for branch, q in zip(event['branches'], quality)}
        before, after = by_id[event['previous_main_id']], by_id[event['main_id']]
        result['valid_switch_events'] += 1
        result['harmful_switch_events'] += int(before >= .5 and after < .2)
        result['beneficial_switch_events'] += int(before < .2 and after >= .5)
    return result


def main():
    args = arguments()
    assert args.sequences > 0 and args.sequence_offset >= 0 and args.max_frames >= 2
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    assert not set(split['train']) & set(split['validation'])
    names = sorted(split['validation'])[args.sequence_offset:args.sequence_offset + args.sequences]
    assert len(names) == args.sequences
    data = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    by_name = {data[i].get_name(): i for i in range(len(data))}
    assert all(name in by_name and name not in split['train'] for name in names)
    checkpoint = torch.load(args.head, map_location='cpu', weights_only=False)
    assert checkpoint['module'] == 'C1_candidate_quality'
    settings = checkpoint['args']
    extractor = FrozenCandidateExtractor(args.pretrained, settings['candidates'], .45, settings['nms_iou']).to(device)
    head = CandidateQualityHead(settings['hidden']).to(device)
    head.load_state_dict(checkpoint['head'], strict=True)
    head.eval().requires_grad_(False)
    config = vars(args) | {'scope': 'C2 bounded-path functionality / train-held-out online diagnostic, not official test',
                           'sequences_selected': names, 'amp_dtype': 'float16', 'head_epoch': checkpoint['epoch'],
                           'gt_input': 'first-frame bbox only; later labels used after tracking for diagnostics',
                           'scorer': 'frozen C1 quality head; C3 utility supervision not implemented',
                           'branch_forward_batched': True,
                           'initialization_and_timing_excluded_from_latency': True,
                           'timing_order': 'C1 then C2 then box-only; cache order prevents reliable speed comparison'}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    print('INITIALIZED', json.dumps(config), flush=True)
    totals = {v: {'iou_sum': 0., 'valid_frames': 0, 'failed_frames': 0, 'switches': 0} for v in ('c1', 'c2', 'box_only')}
    records = []
    for name in names:
        sequence = data[by_name[name]]
        n = min(len(sequence), args.max_frames)
        pairs = [sequence[t].get_image_path() for t in range(n)]
        visible = [Path(pair[0]) for pair in pairs]
        infrared = [Path(pair[1]) for pair in pairs]
        ground_truth = sequence.get_all_bounding_boxes()[:n].copy()
        init_box = ground_truth[0].copy()
        assert np.isfinite(init_box).all() and (init_box[2:] > init_box[:2]).all()
        # Both trackers are causal. Full GT is kept in this outer diagnostic only.
        c1_xywh, c1_latency, updates, selections = track_sequence(visible, infrared, init_box, extractor, head, device, torch.float16)
        c1_boxes = c1_xywh.copy()
        c1_boxes[:, 2:] += c1_boxes[:, :2]
        c2_boxes, c2_latency, c2_stats, events = run_recovery(visible, infrared, init_box, extractor, head, device, args, True)
        box_boxes, box_latency, box_stats, box_events = run_recovery(visible, infrared, init_box, extractor, head, device, args, False)
        output = {'sequence': name, 'frames': n, 'variants': {}}
        qualities = {}
        for variant, boxes, latency, stats in (
            ('c1', c1_boxes, c1_latency, {'template_updates': updates, 'alternative_selections': selections, 'switches': 0}),
            ('c2', c2_boxes, c2_latency, c2_stats), ('box_only', box_boxes, box_latency, box_stats)):
            assert boxes.shape == (n, 4) and len(latency) == n - 1
            quality, valid = summarize(boxes, ground_truth)
            qualities[variant] = quality
            totals[variant]['iou_sum'] += float(quality[valid].sum())
            totals[variant]['valid_frames'] += int(valid.sum())
            totals[variant]['failed_frames'] += int((quality[valid] < .2).sum())
            totals[variant]['switches'] += stats['switches']
            output['variants'][variant] = {'mean_iou': float(quality[valid].mean()), 'valid_frames': int(valid.sum()),
                                           'failed_frames': int((quality[valid] < .2).sum()), 'stats': stats}
            np.savez(out / f'{name}_{variant}.npz', boxes=boxes, quality=quality, valid=valid, latency=latency)
        output['paired_frame_diagnostics'] = {}
        for variant, variant_events in (('c2', events), ('box_only', box_events)):
            baseline_failed = valid & (qualities['c1'] < .2)
            baseline_correct = valid & (qualities['c1'] >= .5)
            output['paired_frame_diagnostics'][variant] = {
                'c1_failed_frames': int(baseline_failed.sum()),
                'c1_correct_frames': int(baseline_correct.sum()),
                'c1_failed_but_variant_correct_frames': int((baseline_failed & (qualities[variant] >= .5)).sum()),
                'c1_correct_but_variant_failed_frames': int((baseline_correct & (qualities[variant] < .2)).sum()),
                **switch_diagnostics(variant_events, ground_truth, valid)}
        (out / f'{name}_c2_events.json').write_text(json.dumps(events))
        (out / f'{name}_box_only_events.json').write_text(json.dumps(box_events))
        records.append(output)
        (out / 'progress.json').write_text(json.dumps(records, indent=2))
        print('SEQUENCE', json.dumps(output), flush=True)
    for summary in totals.values():
        summary['mean_iou'] = summary['iou_sum'] / summary['valid_frames']
    receipt = {'completed': True, 'official_tracking_accuracy': False, 'scope': config['scope'],
               'sequences': len(names), 'variants': totals, 'records': records,
               'new_parameters_trained': False, 'base_and_c1_frozen': True,
               'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2**20}
    (out / 'completion.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
