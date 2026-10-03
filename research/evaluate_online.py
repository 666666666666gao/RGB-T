"""Strictly online GOLA/C1 inference without constructing dataset metadata.

Only the first GT row initializes a tracker. Later GT is accessed exclusively
by offline evaluators after predictions/candidate timelines have been saved.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .candidate_learning import FrozenCandidateExtractor, CandidateQualityHead, foreground_mask, selection_scores
from trackit.miscellanies.image.io import decode_image
from trackit.core.utils.siamfc_cropping import (
    get_siamfc_cropping_params, apply_siamfc_cropping, apply_siamfc_cropping_to_boxes,
    reverse_siamfc_cropping_params,
)
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.runner.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider
from trackit.runner.evaluation.distributed.tracker_evaluator.components.template_updater.simple import SimpleTemplateUpdater
from trackit.runner.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import PostProcessing_BoxWithScoreMap


def read_pair(visible, infrared, device):
    pair = [torch.from_numpy(decode_image(p.read_bytes())).permute(2, 0, 1) for p in (visible, infrared)]
    return torch.cat(pair).to(device=device, dtype=torch.float32)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['lasher', 'rgbt234'], required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--variant', choices=['baseline', 'c1'], required=True)
    p.add_argument('--pretrained', default='pretrained_models/gola_b224.bin')
    p.add_argument('--head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--output', required=True, help='Direct folder containing sequence.txt for official rgbt evaluator')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit-sequences', type=int, default=0, help='0 means all sequences; positive values are smoke ONLY')
    p.add_argument('--max-frames', type=int, default=0, help='0 means all frames; positive values are smoke ONLY')
    p.add_argument('--amp-dtype', choices=['float16', 'bfloat16'], default='float16')
    p.add_argument('--record-mechanisms', action='store_true', help='Save every candidate and template write for full offline GT diagnostics.')
    p.add_argument('--validation-split', help='Original TRAIN-held-out split; internal diagnostics only.')
    p.add_argument('--pause-after-correction', type=int, default=0,
                   help='C1 write control: pause this many frames including each non-Hann selection; 0 preserves original updates.')
    p.add_argument('--candidate-policy', choices=['peaks', 'dense_hann5', 'dense_raw5'], default='peaks',
                   help='C1 candidate-completion control: same five slots/Hann winner/spacing/NMS and frozen scorer; no extra visual crop.')
    return p.parse_args()


@torch.inference_mode()
def track_sequence(paths_v, paths_i, init_box, extractor, head, device, amp_dtype, timeline=None, pause_after_correction=0):
    assert len(paths_v) == len(paths_i)
    normalization = get_dataset_norm_stats_transform('mm', inplace=True)
    template_size, search_size = np.array((112, 112)), np.array((224, 224))
    image = read_pair(paths_v[0], paths_i[0], device)
    z_params = get_siamfc_cropping_params(init_box, 2., template_size)
    z, image_mean, z_params = apply_siamfc_cropping(image, template_size, z_params, 'bilinear', False)
    z = normalization(z / 255.)
    z_mask = foreground_mask(init_box, z_params).to(device).unsqueeze(0)
    d_mask = z_mask.clone()
    provider = SiamFCCroppingParameterSimpleProvider(4., 10.)
    provider.initialize(init_box)
    updater = SimpleTemplateUpdater(.84, 2., (112, 112), 'mm', 'bilinear', False, device)
    updater.start(1, (6, 112, 112))
    updater.initialize(0, z)
    post_process = PostProcessing_BoxWithScoreMap(device, (16, 16), (224, 224), .45)
    post_process.start()
    predictions = [init_box.copy()]
    latencies = []
    updates, alternative_selections = 0, 0
    pause_until = 0
    for frame in range(1, len(paths_v)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        image = read_pair(paths_v[frame], paths_i[frame], device)
        size = np.array((image.shape[-1], image.shape[-2]))
        x, _, x_params = apply_siamfc_cropping(image, search_size, provider.get(search_size),
                                              'bilinear', False, image_mean)
        x = normalization(x / 255.)
        batch = {'z': z.unsqueeze(0), 'd': updater.get(0).unsqueeze(0), 'x': x.unsqueeze(0),
                 'z_feat_mask': z_mask, 'd_feat_mask': d_mask}
        with torch.autocast('cuda', dtype=amp_dtype):
            if head is None:
                if timeline is None:
                    output = extractor.base(**batch)
                else:
                    candidates, output = extractor.extract_with_output(batch)
                    choice = 0
                    predicted_quality = candidates['raw_score'][0]
                result = post_process(output)
                crop_box = result['box'][0].double().cpu().numpy()
                confidence = float(result['confidence'][0])
            else:
                candidates = extractor(batch)
                logits = head(candidates).float()
                predicted_quality = logits.sigmoid()[0]
                scores = selection_scores(logits, candidates, extractor.window_penalty)
                choice = int(scores.masked_fill(~candidates['valid'], -torch.inf).argmax(1)[0])
                # Match the original postprocessor: scale float32 before double.
                crop_box = (candidates['boxes'][0, choice] * 224.).double().cpu().numpy()
                # Preserve the original confidence definition and update threshold.
                confidence = float(candidates['raw_score'][0, choice])
                alternative_selections += int(choice != 0)
        box = apply_siamfc_cropping_to_boxes(crop_box, reverse_siamfc_cropping_params(x_params))
        bbox_clip_to_image_boundary_(box, size)
        assert np.isfinite(box).all() and np.isfinite(confidence)
        provider.update(confidence, box, size)
        # Past predictions are appended once and never rewritten.
        predictions.append(box.copy())
        if pause_after_correction and head is not None and choice != 0:
            pause_until = frame + pause_after_correction - 1
        write_permitted = frame > pause_until
        if write_permitted:
            updater.update(0, confidence, image, box)
        if confidence > .84 and write_permitted:
            # Match the official online-mask plugin, which uses unadjusted params.
            d_params = get_siamfc_cropping_params(box, 2., template_size)
            d_mask = foreground_mask(box, d_params).to(device).unsqueeze(0)
            updates += 1
        torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - started)
        if timeline is not None:
            candidate_boxes = candidates['boxes'][0].double().cpu().numpy() * 224.
            candidate_boxes = apply_siamfc_cropping_to_boxes(candidate_boxes, reverse_siamfc_cropping_params(x_params))
            for candidate_box in candidate_boxes:
                bbox_clip_to_image_boundary_(candidate_box, size)
            # Baseline postprocessing scales in float32; record its actual box.
            candidate_boxes[choice] = box.copy()
            timeline.append({'boxes_xyxy': candidate_boxes.copy(),
                             'evidence': candidates['evidence'][0].float().cpu().numpy().copy(),
                             'valid': candidates['valid'][0].cpu().numpy().copy(),
                             'predicted_quality': predicted_quality.float().cpu().numpy().copy(),
                             'choice': np.array(choice, dtype=np.int64),
                             'template_updated': np.array(confidence > .84 and write_permitted),
                             'template_write_permitted': np.array(write_permitted)})
    updater.delete(0)
    updater.stop()
    post_process.stop()
    predictions = np.array(predictions)
    predictions[:, 2:] -= predictions[:, :2]
    return predictions, np.array(latencies), updates, alternative_selections


def main():
    args = arguments()
    assert args.pause_after_correction >= 0 and (args.pause_after_correction == 0 or args.variant == 'c1')
    assert args.variant == 'c1' or args.candidate_policy == 'peaks'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    head = None
    if args.variant == 'c1':
        checkpoint = torch.load(args.head, map_location='cpu', weights_only=False)
        assert checkpoint['module'] == 'C1_candidate_quality'
        head = CandidateQualityHead(checkpoint['args']['hidden'])
        head.load_state_dict(checkpoint['head'], strict=True)
        head = head.to(device).eval().requires_grad_(False)
        settings = checkpoint['args']
        assert settings['window_penalty'] == .45, 'Evaluation preserves the original GOLA window'
        extractor = FrozenCandidateExtractor(args.pretrained, settings['candidates'], .45, settings['nms_iou'],
                                             proposal_policy=args.candidate_policy).to(device)
    else:
        extractor = FrozenCandidateExtractor(args.pretrained).to(device)
    dtype = getattr(torch, args.amp_dtype)
    sequences = sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    if args.validation_split:
        split = json.loads(Path(args.validation_split).read_text())
        assert args.dataset == 'lasher' and not set(split['train']) & set(split['validation'])
        sequences = [p for p in sequences if p.name in split['validation']]
        assert {p.name for p in sequences} == set(split['validation'])
    if args.limit_sequences:
        sequences = sequences[:args.limit_sequences]
    assert sequences
    config = vars(args) | {'scope': 'strict online baseline or C1 reranking; no bounded branches/state repair/A/B',
                           'update_threshold': .84, 'search_area_factor': 4., 'min_search_object_size': 10.,
                           'template_area_factor': 2., 'pretrained_load': extractor.load_receipt,
                           'gpu': torch.cuda.get_device_name(device),
                           'head_training_amp_dtype': 'bfloat16' if head is not None else None,
                           'backbone_attention': 'none',
                           'candidate_box_scaling': 'float32_multiply_then_double',
                           'timing_excludes_first_frame_initialization': True,
                           'smoke_only': bool(args.limit_sequences or args.max_frames or args.validation_split),
                           'head_epoch': checkpoint['epoch'] if head is not None else None,
                           'initialization': 'init.txt first row' if args.dataset == 'lasher' else 'visible.txt first row'}
    config['template_write_policy'] = 'raw confidence > .84 AND outside a causal pause after C1 non-Hann selection; no GT validation'
    (out / 'inference_config.json').write_text(json.dumps(config, indent=2))
    torch.cuda.reset_peak_memory_stats(device)
    full_started = time.perf_counter()
    records, all_latency = [], []
    with (out / 'progress.jsonl').open('w') as progress:
        for index, sequence in enumerate(sequences, 1):
            visible = sorted(p for p in (sequence / 'visible').iterdir() if p.is_file())
            infrared = sorted(p for p in (sequence / 'infrared').iterdir() if p.is_file())
            assert len(visible) == len(infrared)
            label_path = sequence / ('init.txt' if args.dataset == 'lasher' else 'visible.txt')
            with label_path.open() as labels:
                init_box = np.fromstring(labels.readline().strip(), sep=',')
            assert init_box.shape == (4,)
            init_box[2:] += init_box[:2]
            if args.max_frames:
                visible, infrared = visible[:args.max_frames], infrared[:args.max_frames]
            timeline = [] if args.record_mechanisms else None
            predictions, latencies, updates, switches = track_sequence(visible, infrared, init_box, extractor, head, device, dtype,
                                                                      timeline, args.pause_after_correction)
            np.savetxt(out / f'{sequence.name}.txt', predictions, delimiter='\t', fmt='%.3f')
            np.save(out / f'{sequence.name}_latency.npy', latencies)
            if timeline is not None:
                assert len(timeline) == len(predictions) - 1
                np.savez_compressed(out / f'{sequence.name}_candidates.npz',
                                    **{key: np.stack([row[key] for row in timeline]) for key in timeline[0]})
            all_latency.extend(latencies.tolist())
            record = {'sequence': sequence.name, 'sequence_index': index, 'sequences': len(sequences),
                      'frames': len(predictions), 'elapsed_seconds': float(latencies.sum()),
                      'template_updates': updates, 'alternative_selections': switches}
            records.append(record)
            progress.write(json.dumps(record) + '\n')
            progress.flush()
            print('SEQUENCE', json.dumps(record), flush=True)
    latency = np.array(all_latency)
    receipt = {'completed': True, 'sequences': len(records), 'frames': sum(r['frames'] for r in records),
               'smoke_only': config['smoke_only'], 'fps_including_decode_crop_update': len(latency) / latency.sum(),
               'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
               'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
               'timing_excludes_first_frame_initialization': True,
               'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
               'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
               'wall_seconds_including_initialization_and_diagnostic_serialization': time.perf_counter() - full_started,
               'candidate_timeline_recorded': args.record_mechanisms,
               'records': records, 'official_accuracy': 'not computed here; run evaluation.py against actual GT'}
    (out / 'inference_completion.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETED', json.dumps({k: v for k, v in receipt.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
