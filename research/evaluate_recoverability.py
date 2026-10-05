"""Strict online one-extra-region evaluation; later GT is offline-only."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .collect_recoverability import InstanceExtractor
from .evaluate_online import read_pair, track_sequence
from .recoverability_modules import RecoverabilityModules
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import TemporalModules


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['lasher', 'rgbt234'], required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--pretrained', default='pretrained_models/gola_b224.bin')
    parser.add_argument('--c1-head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    parser.add_argument('--motion-run', default='/data/gb/outputs/abc_joint_v1_seed42')
    parser.add_argument('--output', required=True)
    parser.add_argument('--validation-split')
    parser.add_argument('--sequence-offset', type=int, default=0)
    parser.add_argument('--limit-sequences', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--zero-init', action='store_true', help='Untrained residual initialization; M0 control only.')
    parser.add_argument('--parity-check', action='store_true', help='Real-video bitwise C1 output/update check for M0 or C1 control.')
    parser.add_argument('--policy', choices=['learned', 'c1'], default='learned')
    parser.add_argument('--disable-search', action='store_true')
    parser.add_argument('--unsafe-writes', action='store_true', help='Matched write ablation: force original raw>.84 template write after selection.')
    parser.add_argument('--write-verification', choices=('identity', 'action'), default='identity',
                        help='Fixed-weight control: action keeps learned regular/pause choice without an independent identity veto. Memory gates stay unchanged.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--commit-model', help='Optional trained query search/appearance commit head; parent model stays fixed.')
    return parser.parse_args()


@torch.inference_mode()
def track(visible, infrared, initial, extractor, modules, motion, device, args, threshold, commit_head=None):
    tracker_type, kwargs = RecoverabilityTracker, {}
    if commit_head is not None:
        from .geometry_commit import GeometryCommitTracker
        tracker_type, kwargs = GeometryCommitTracker, {'commit_head':commit_head}
    tracker = tracker_type(extractor, modules, motion, read_pair(visible[0], infrared[0], device),
                                    initial, threshold, args.policy, not args.disable_search, not args.unsafe_writes,
                                    args.write_verification, **kwargs)
    anchors = [value.clone() for value in (tracker.anchor, tracker.identity_anchor, tracker.motion_anchor)]
    predictions, latencies, decisions = [initial.copy()], [], []
    for frame in range(1, len(visible)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions.append(tracker.step(read_pair(visible[frame], infrared[frame], device)))
        torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - started)
        # Timeline transfer and file serialization are outside measured tracking latency.
        decisions.append({key: value.cpu().numpy().copy() for key, value in tracker.last_decision.items()})
    assert all(torch.equal(before, after) for before, after in zip(anchors, (tracker.anchor, tracker.identity_anchor, tracker.motion_anchor)))
    assert len(decisions) == len(latencies) == len(predictions) - 1
    for box, decision in zip(predictions[1:], decisions):
        region, candidate = divmod(int(decision['choice']), 5)
        assert np.array_equal(box, decision['boxes_xyxy'][region, candidate])
        assert int(decision['valid'].sum()) <= 10
    assert tracker.identity_memory.shape == tracker.motion_memory.shape == (1, 4, 2, 128)
    assert tracker.stats['extra_visual_forwards'] == sum(int(row['extra_executed']) for row in decisions)
    assert tracker.stats['template_updates'] == sum(int(row['template_updated']) for row in decisions)
    predictions = np.asarray(predictions)
    predictions[:, 2:] -= predictions[:, :2]
    stats = {key: tracker.stats[key] for key in ('template_updates', 'max_pending_per_branch',
             'extra_searches_requested', 'extra_visual_forwards', 'skipped_empty_extra_regions',
             'changed_candidate_indices', 'paused_query_writes', 'max_motion_history')}
    if commit_head is not None:
        stats.update({key:tracker.stats[key] for key in ('geometry_commit_interventions','appearance_commit_overrides')})
    if args.parity_check:
        extractor.proposal_policy = 'peaks'
        reference_timeline = []
        reference, _, updates, _ = track_sequence(visible, infrared, initial, extractor, modules.c1, device,
                                                torch.float16, reference_timeline)
        assert np.array_equal(predictions, reference), f'C1 bitwise output parity failed: max error {np.abs(predictions-reference).max()}'
        for actual, baseline in zip(decisions, reference_timeline):
            choice = int(baseline['choice'])
            assert int(actual['choice']) == choice == int(actual['original_choice'])
            assert bool(actual['template_updated']) == bool(baseline['template_updated'])
            assert actual['raw_score'][0, choice] == baseline['evidence'][choice, 0]
        assert stats['template_updates'] == updates and stats['extra_visual_forwards'] == 0
        assert stats['paused_query_writes'] == stats['changed_candidate_indices'] == 0
        stats['real_video_c1_bitwise_parity'] = True
    return predictions, np.asarray(latencies), decisions, stats


def main():
    args = arguments()
    assert not args.parity_check or args.zero_init or args.policy == 'c1'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    c1 = torch.load(args.c1_head, map_location='cpu', weights_only=False)
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    assert checkpoint['module'] == 'ABC_recoverability' and checkpoint['args']['c1_head'] == args.c1_head
    threshold = checkpoint['args']['threshold']
    modules = RecoverabilityModules(c1).to(device)
    if not args.zero_init:
        modules.load_state_dict(checkpoint['model'], strict=True)
    modules.eval().requires_grad_(False)
    commit_head = None
    if args.commit_model:
        from .geometry_commit import GeometryCommitHead
        commit = torch.load(args.commit_model,map_location='cpu',weights_only=False)
        assert commit['module']=='geometry_commit' and commit['parent_model']==args.model
        assert commit['features']==463 and commit['threshold']==.03
        assert not args.parity_check and not args.zero_init and not args.unsafe_writes
        assert args.policy=='learned' and args.write_verification=='action'
        commit_head=GeometryCommitHead().to(device)
        commit_head.load_state_dict(commit['head'],strict=True)
        commit_head.eval().requires_grad_(False)
    motion_config = json.loads((Path(args.motion_run) / 'config.json').read_text())
    motion_checkpoint = torch.load(Path(args.motion_run) / 'last.pth', map_location='cpu', weights_only=False)
    assert motion_checkpoint['epoch'] == 30 and motion_config['c1_head'] == args.c1_head
    motion = TemporalModules(c1, motion_config['slots'], motion_config['motion_history'],
                             motion_config['modes'], motion_checkpoint['horizon']).to(device)
    motion.load_state_dict(motion_checkpoint['model'], strict=True)
    motion.eval().requires_grad_(False)
    assert motion_config['motion_history'] == 8 and motion_config['modes'] == motion_checkpoint['horizon'] == 3
    extractor = InstanceExtractor(args.pretrained, c1['args']['candidates'], .45, c1['args']['nms_iou']).to(device)
    assert c1['args']['candidates'] == 5
    sequences = sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    if args.validation_split:
        split = json.loads(Path(args.validation_split).read_text())
        assert args.dataset == 'lasher' and not set(split['train']) & set(split['validation'])
        sequences = [p for p in sequences if p.name in split['validation']]
        assert {p.name for p in sequences} == set(split['validation'])
    sequences = sequences[args.sequence_offset:]
    if args.limit_sequences:
        sequences = sequences[:args.limit_sequences]
    assert sequences
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    smoke = bool(args.validation_split or args.sequence_offset or args.limit_sequences or args.max_frames or args.zero_init or args.parity_check)
    config = vars(args) | {'threshold': threshold, 'head_epoch': 0 if args.zero_init else checkpoint['epoch'],
                           'model_training_seed': checkpoint['args']['seed'],
                           'scope': 'causal single path, bounded instance memory, learned one-extra-region search and query-only safe write',
                           'smoke_only': smoke, 'pretrained_load': extractor.load_receipt,
                           'new_abc_parameters': sum(p.numel() for name, p in modules.named_parameters() if not name.startswith('c1.')),
                           'frozen_motion_memory_and_predictor_parameters': sum(p.numel() for group in (motion.memory, motion.motion) for p in group.parameters()),
                           'total_loaded_parameters_including_unused_frozen_motion_heads': sum(p.numel() for group in (extractor, modules, motion) for p in group.parameters()),
                           'persistent_memory_slots_per_sensor': {'identity': 4, 'frozen_motion': 4},
                           'motion_history_capacity': 8, 'max_extra_visual_forwards_per_frame': 1,
                           'max_actual_candidates_per_frame': 10, 'amp_dtype': 'float16',
                           'gt_input': 'only first annotation initializes; later GT only offline scoring',
                           'past_outputs_never_rewritten': True, 'arbitrary_historical_state_restoration': False,
                           'timing_excludes_initialization': True, 'timing_excludes_diagnostic_transfer_and_serialization': True,
                           'bootstrap_training_future_policy': 'frozen continuation policy recorded in training source_configs; not recomputed by this evaluator',
                           'initialization': 'init.txt first row' if args.dataset == 'lasher' else 'visible.txt first row'}
    if commit_head is not None:
        config.update(commit_head_epoch=commit['epoch'],
                      new_commit_parameters=sum(p.numel() for p in commit_head.parameters()),
                      commit_scope='Current output and motion history unchanged; learned query search-reference and appearance commits')
        config['total_loaded_parameters_including_unused_frozen_motion_heads'] += config['new_commit_parameters']
    (out / 'inference_config.json').write_text(json.dumps(config, indent=2))
    records, all_latency, started = [], [], time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with (out / 'progress.jsonl').open('w') as stream:
        for index, sequence in enumerate(sequences, 1):
            visible = sorted(p for p in (sequence / 'visible').iterdir() if p.is_file())
            infrared = sorted(p for p in (sequence / 'infrared').iterdir() if p.is_file())
            assert len(visible) == len(infrared)
            with (sequence / ('init.txt' if args.dataset == 'lasher' else 'visible.txt')).open() as labels:
                initial = np.fromstring(labels.readline().strip(), sep=',')
            assert initial.shape == (4,)
            initial[2:] += initial[:2]
            if args.max_frames:
                visible, infrared = visible[:args.max_frames], infrared[:args.max_frames]
            assert len(visible) > 1
            prediction, latency, decisions, stats = track(visible, infrared, initial, extractor, modules, motion, device, args, threshold, commit_head)
            assert np.isfinite(prediction).all() and np.isfinite(latency).all() and (latency > 0).all()
            np.savetxt(out / (sequence.name + '.txt'), prediction, delimiter='\t', fmt='%.3f')
            if args.parity_check:
                np.save(out / (sequence.name + '_unrounded.npy'), prediction)
            np.save(out / (sequence.name + '_latency.npy'), latency)
            np.savez_compressed(out / (sequence.name + '_recoverability_decisions.npz'),
                                **{key: np.stack([row[key] for row in decisions]) for key in decisions[0]})
            record = {'sequence': sequence.name, 'sequence_index': index, 'sequences': len(sequences),
                      'frames': len(prediction), 'elapsed_seconds': float(latency.sum()), 'stats': stats}
            records.append(record)
            all_latency.extend(latency.tolist())
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            print('SEQUENCE', json.dumps(record), flush=True)
    latency = np.asarray(all_latency)
    receipt = {'completed': True, 'sequences': len(records), 'frames': sum(row['frames'] for row in records),
               'smoke_only': smoke, 'records': records, 'gt_scoring_completed': False,
               'fps_including_decode_crop_update': len(latency) / latency.sum(),
               'latency_p50_ms': float(np.percentile(latency, 50) * 1000),
               'latency_p95_ms': float(np.percentile(latency, 95) * 1000),
               'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
               'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
               'wall_seconds_including_initialization_and_diagnostic_serialization': time.perf_counter() - started,
               'official_accuracy': 'not computed here; native actual-GT collector required'}
    (out / 'inference_completion.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETED', json.dumps({key: value for key, value in receipt.items() if key != 'records'}), flush=True)


if __name__ == '__main__':
    main()
