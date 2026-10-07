"""Collect current-policy event rollouts with one exact prefix replay per video.

Each video stores its causal history once. Query inputs precede any future
decode; future images/GT are confined to the existing private action rollout.
This creates supervision, not optimization or native benchmark accuracy.
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .candidate_learning import CandidateQualityHead
from .collect_recoverability import InstanceExtractor, REGIONS, history_arrays, label_contexts
from .collect_rollouts import observe_actions
from .collect_temporal import paths
from .evaluate_online import read_pair
from .recoverability_modules import RecoverabilityModules
from .recoverability_tracker import RecoverabilityTracker
from .temporal_modules import TemporalModules
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped


HISTORY_FIELDS = ('history_descriptors', 'history_instance_descriptors', 'history_evidence',
                  'history_quality', 'history_boxes', 'history_frames', 'history_write',
                  'history_valid', 'history_iou')


def models(args, device):
    c1 = torch.load(args.head, map_location='cpu', weights_only=False)
    assert c1['module'] == 'C1_candidate_quality' and c1['args']['candidates'] == 5
    extractor = InstanceExtractor(args.pretrained, 5, .45, c1['args']['nms_iou']).to(device)
    head = CandidateQualityHead(c1['args']['hidden']).to(device)
    head.load_state_dict(c1['head'], strict=True)
    head.eval().requires_grad_(False)
    motion_config = json.loads((Path(args.motion_run) / 'config.json').read_text())
    motion_checkpoint = torch.load(Path(args.motion_run) / 'last.pth', map_location='cpu', weights_only=False)
    assert motion_checkpoint['epoch'] == 30 and motion_config['c1_head'] == args.head
    motion = TemporalModules(c1, motion_config['slots'], motion_config['motion_history'], motion_config['modes'], motion_checkpoint['horizon']).to(device)
    motion.load_state_dict(motion_checkpoint['model'], strict=True)
    motion.eval().requires_grad_(False)
    checkpoint = torch.load(args.prefix_model, map_location='cpu', weights_only=False)
    assert checkpoint['module'] == 'ABC_candidate_relations' and checkpoint['epoch'] == 5
    prefix = RecoverabilityModules(c1, candidate_relations=True).to(device)
    prefix.load_state_dict(checkpoint['model'], strict=True)
    prefix.eval().requires_grad_(False)
    args.prefix_threshold = checkpoint['args']['threshold']
    return extractor, head, motion, prefix


@torch.inference_mode()
def collect_sequence(sequence, jobs, reference_file, extractor, head, motion, prefix, device, args, output):
    # Initialize exactly as the locked native policy; the training cache clips
    # rightgreen's first box by one pixel at the image boundary.
    with (Path(args.root) / 'traingset' / sequence.get_name() / 'init.txt').open() as labels:
        initial = np.fromstring(labels.readline().strip(), sep=',')
    assert initial.shape == (4,)
    initial[2:] += initial[:2]
    image = read_pair(*paths(sequence, 0), device)
    tracker = RecoverabilityTracker(extractor, prefix, motion, image, initial, args.prefix_threshold,
                                   write_verification='action', search_value='gross')
    evidence = np.zeros(10, dtype=np.float32)
    evidence[:6] = 1
    context = dict(sequence=sequence, initial=initial, tracker=tracker, branch=tracker.branch,
                   history=deque([dict(descriptor=None, instance=None, evidence=evidence,
                                       quality=1., box=initial, frame=0, write=False)]))
    wanted = {job['query_frame'] for job in jobs}
    with np.load(reference_file) as reference:
        choices = reference['choice'].astype(int)
        boxes = reference['boxes_xyxy'].reshape(len(choices), 35, 4)[np.arange(len(choices)), choices]
        pauses = reference['pause'].copy()
        writes = reference['template_updated'].copy()
    # Three TRAIN videos have unannotated image suffixes absent from this
    # annotation cache. Their cached paths match the native prefix exactly.
    assert max(wanted) + 3 < len(sequence) and len(boxes) >= len(sequence) - 1
    rows, queries, prefix_history = [], [], None
    for frame in range(1, max(wanted) + 1):
        if frame in wanted:
            context['query'] = frame
            # Storage capacity covers this exact query, not a global padded
            # maximum. The history is moved to one shared per-video artifact.
            args.max_prefix = max(8, frame)
            query_image = read_pair(*paths(sequence, frame), device)
            original = observe_actions([(tracker, tracker.branch, query_image)], extractor, head)
            row = history_arrays(context, original[0][0]['anchor_features'][0].cpu().numpy(), args.max_prefix)
            row['prefix_counts'] = np.asarray([tracker.stats[key] for key in
                ('extra_visual_forwards', 'changed_candidate_indices', 'paused_query_writes', 'template_updates')], dtype=np.int64)
            row['online_identity_anchor'] = tracker.identity_anchor.cpu().numpy()[0].copy()
            row['online_identity_memory'] = tracker.identity_memory.cpu().numpy()[0].copy()
            history = tracker.history_inputs()
            distribution = motion.motion(history['history_boxes'], history['history_frames'], history['history_valid'],
                history['history_quality'], tracker.motion_anchor, tracker.motion_memory)
            row = label_contexts([context], [query_image], original, [row], distribution, extractor, head, device, args, 'own')[0]
            known_history = row['history_valid'].copy()
            assert np.array_equal(row['history_frames'][known_history], np.arange(frame))
            prefix_history = {key: row.pop(key)[known_history] for key in HISTORY_FIELDS}
            rows.append(row)
            queries.append(frame)
        # Counterfactuals must leave this actual prefix unchanged. Every real
        # output and decision is checked against the locked full trajectory.
        image = read_pair(*paths(sequence, frame), device)
        box = tracker.step(image)
        assert np.array_equal(box, boxes[frame - 1]), (sequence.get_name(), frame, 'Actual prefix changed')
        assert int(tracker.last_decision['choice']) == int(choices[frame - 1])
        assert bool(tracker.last_decision['pause']) == bool(pauses[frame - 1])
        assert bool(tracker.last_decision['template_updated']) == bool(writes[frame - 1])
        if frame in wanted:
            rows[-1]['parent_flat_action'] = np.int64(2 * int(tracker.last_decision['choice']) + int(tracker.last_decision['pause']))
            rows[-1]['parent_search_triggered'] = np.bool_(bool(tracker.last_decision['search_requested']))
        candidate, quality, _, choice = tracker.last_observation
        context['branch'] = tracker.branch
        context['history'].append(dict(descriptor=candidate['modality_features'][0, choice].cpu().numpy().copy(),
            instance=candidate['instance_features'][0, choice].cpu().numpy().copy(),
            evidence=candidate['evidence'][0, choice].cpu().numpy().copy(), quality=float(quality[choice]),
            box=box.copy(), frame=frame, write=bool(tracker.last_decision['template_updated'])))
    assert queries == sorted(wanted) and prefix_history is not None
    name = sequence.get_name()
    np.savez_compressed(output / (name + '_prefix.npz'), **prefix_history)
    np.savez_compressed(output / (name + '_queries.npz'), query_frames=np.asarray(queries, dtype=np.int64),
                        **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
    return dict(sequence=name, queries=len(rows), max_query=max(wanted),
                replayed_visual_prefix_frames=max(wanted), duplicate_prefix_replays=0,
                all_actual_prefix_outputs_and_decisions_exact=True,
                valid_actions=sum(int(row['action_valid'].sum()) for row in rows),
                executed_extra_label_crops=sum(int(row['region_executed'][1:].sum()) for row in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/data/wangwj/dataset/LasHeR')
    p.add_argument('--cache', default='trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
    p.add_argument('--split', default='/data/gb/outputs/c1_initial_seed42/split.json')
    p.add_argument('--partition', choices=('train', 'validation'), required=True)
    p.add_argument('--pretrained', default='pretrained_models/gola_b224.bin')
    p.add_argument('--head', default='/data/gb/outputs/c1_initial_seed42/best.pth')
    p.add_argument('--motion-run', default='/data/gb/outputs/abc_joint_v1_seed42')
    p.add_argument('--prefix-model', default='/data/gb/outputs/candidate_relation_native_recovery_20261006/reconstructed_epoch5/best.pth')
    p.add_argument('--jobs-file', required=True)
    p.add_argument('--reference-traces', required=True, help='Completed exact locked-policy timelines; one file per sequence')
    p.add_argument('--reference-model', required=True, help='Preserved checkpoint recorded by these timelines; must equal all current policy tensors')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true', help='Keep closed per-video artifacts after a proven interrupted collection')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    args.forward_batch = 1  # Exact serial deployed visual path for every region.
    split = json.loads(Path(args.split).read_text())
    assert len(split['train']) == 881 and len(split['validation']) == 98 and not set(split['train']) & set(split['validation'])
    jobs = json.loads(Path(args.jobs_file).read_text())['jobs']
    assert jobs and len({(row['sequence'], row['query_frame']) for row in jobs}) == len(jobs)
    groups = {}
    for job in jobs:
        assert job['sequence'] in split[args.partition]
        groups.setdefault(job['sequence'], []).append(job)
    references = {}
    current = torch.load(args.prefix_model, map_location='cpu', weights_only=False)
    reference_checkpoint = torch.load(args.reference_model, map_location='cpu', weights_only=False)
    assert reference_checkpoint['module'] == current['module'] == 'ABC_candidate_relations'
    assert reference_checkpoint['model'].keys() == current['model'].keys()
    assert all(torch.equal(current['model'][key], value) for key, value in reference_checkpoint['model'].items())
    for path in Path(args.reference_traces).rglob('*_recoverability_decisions.npz'):
        name = path.name.removesuffix('_recoverability_decisions.npz')
        assert name not in references
        config = json.loads((path.parent / 'inference_config.json').read_text())
        assert config['model'] == args.reference_model and config['head_epoch'] == reference_checkpoint['epoch']
        assert config['ABC_model_family'] == 'ABC_candidate_relations' and config['search_value'] == 'gross' and config['write_verification'] == 'action'
        assert config['policy'] == 'learned' and not config['disable_search'] and not config['unsafe_writes'] and not config['zero_init']
        assert all(config[key] is None for key in ('commit_model', 'state_commit_model', 'identity_projection'))
        # The completed parent-control DEV run uses a read-only native reference
        # stream; its main parent behavior was unchanged. Full replay is still
        # required here, not inferred from the configuration name.
        assert config['reference_mode'] in (None, 'parent')
        assert config['pretrained'] == args.pretrained and config['c1_head'] == args.head and config['motion_run'] == args.motion_run
        assert config['identity_weight'] == 0. and config['max_frames'] == 0
        references[name] = path
    assert set(groups) <= references.keys()
    dataset = MultiModalObjectTrackingDataset_MemoryMapped.load(args.root, args.cache)
    sequences = {dataset[i].get_name(): i for i in range(len(dataset))}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device('cuda:0')
    extractor, head, motion, prefix = models(args, device)
    config = vars(args) | dict(storage_schema='causal_sequence_prefix_v1', jobs=jobs, regions=REGIONS,
        clips=len(jobs), sequences=len(groups), prefix_model_family='ABC_candidate_relations', prefix_checkpoint_epoch=5,
        prefix_search_value='gross', prefix_write_verification='action', future_policy_mode='own', future_checkpoint_epoch=5,
        future_horizon=3, future_policy='Frozen actual E5/gross/action serial private continuations',
        prefix_policy='Locked actual E5/gross/action; one replay pervideo, every output checked against closed fulltimeline',
        decision_input_contains_future=False, history_storage='One full causal predicted prefix pervideo; query loader must slice strictly before query',
        reference_model_weights_exact_current=True, reference_checkpoint_epoch=reference_checkpoint['epoch'],
        motion_input='Exact online frozen motion anchors/memory/history at query-1, not a reconstructed quantized prefix',
        query_extra_regions='Six independent training actions; deployment still local plus at mostone observed extra',
        pretrained_load=extractor.load_receipt, new_optimizer_updates=0, native_TEST_accuracy=False)
    rows, previous_elapsed, previous_peak = [], 0., 0.
    if args.resume:
        saved = json.loads((output / 'config.json').read_text())
        assert {key: value for key, value in saved.items() if key != 'resume'} == {key: value for key, value in json.loads(json.dumps(config)).items() if key != 'resume'}
        assert not (output / 'completion.json').exists()
        rows = [json.loads(line) for line in (output / 'progress.jsonl').read_text().splitlines()]
        for row in rows:
            name = row['sequence']
            expected = sorted(job['query_frame'] for job in groups[name])
            assert row['all_actual_prefix_outputs_and_decisions_exact'] and row['queries'] == len(expected)
            with np.load(output / (name + '_queries.npz')) as archive:
                assert np.array_equal(archive['query_frames'], expected)
            with np.load(output / (name + '_prefix.npz')) as archive:
                assert np.array_equal(archive['history_frames'], np.arange(max(expected)))
        progress = json.loads((output / 'progress.json').read_text())
        assert progress['completed_sequences'] == len(rows)
        assert progress['completed_queries'] == sum(row['queries'] for row in rows)
        previous_elapsed, previous_peak = progress['elapsed_seconds'], progress['peak_cuda_mib']
    else:
        (output / 'config.json').write_text(json.dumps(config, indent=2))
    completed = {row['sequence'] for row in rows}
    assert len(completed) == len(rows) and completed <= groups.keys()
    started = time.perf_counter()
    with (output / 'progress.jsonl').open('a' if args.resume else 'w') as stream:
        for name in sorted(groups):
            if name in completed:
                continue
            sequence = dataset[sequences[name]]
            for job in groups[name]:
                q = job['query_frame']
                assert q >= 1 and q + 3 < len(sequence)
                gt = np.stack([sequence[t].get_bounding_box() for t in range(q, q + 4)])
                assert np.isfinite(gt).all() and (gt[:, 2:] > gt[:, :2]).all()
            row = collect_sequence(sequence, groups[name], references[name], extractor, head, motion, prefix, device, args, output)
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            progress = dict(completed_queries=sum(row['queries'] for row in rows), queries=len(jobs),
                completed_sequences=len(rows), sequences=len(groups), elapsed_seconds=previous_elapsed+time.perf_counter()-started,
                peak_cuda_mib=max(previous_peak, torch.cuda.max_memory_allocated(device)/2**20))
            (output / 'progress.json').write_text(json.dumps(progress))
            print(json.dumps(progress), flush=True)
    receipt = dict(completed=True, partition=args.partition, clips=len(jobs), sequences=len(rows), records=rows,
        valid_actions=sum(row['valid_actions'] for row in rows), exact_actual_prefix_parity=True,
        decision_input_contains_future=False, new_optimizer_updates=0, native_TEST_accuracy=False,
        elapsed_seconds=previous_elapsed+time.perf_counter()-started, peak_cuda_mib=max(previous_peak, torch.cuda.max_memory_allocated(device)/2**20))
    (output / 'completion.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
