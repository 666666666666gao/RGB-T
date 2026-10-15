"""Finite real-TRAIN qualification for R3; never a benchmark or fit result.

Run only after the original R2 pair closes. The locked parent is compared with
zero-residual R3, then actual native state interventions and real-label head
updates are checked. Probe heads are never used to initialize formal training.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .counterfactual import ACTION_MODES
from .data import VideoClips, iou, xyxy
from .dense_consequences import assert_same_snapshot, collect_matched_query
from .dense_main_tracker import DenseABCTracker
from .dense_submission import DenseCandidateEvidence, DenseDecisionHeads
from .dense_tracker import DenseCADTracker
from .train_dense import module_hash
from .train_dense_consequences import evaluate_batch
from .train_dense_evidence import batch_objective, score_context


def ordinary_copy(value):
    # Leave inference mode via the same serialization boundary as real caches.
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return torch.load(stream, map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--r2-pair-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    source = json.loads(Path(args.config).read_text())
    pair_root = Path(args.r2_pair_root).resolve()
    pair = json.loads((pair_root / 'pair_complete.json').read_text())
    assert pair['status'] == 'DENSE_MATCHED_PAIR_FULL_TRAINING_AND_EVALUATION_COMPLETE'
    assert pair['all20_actual_clip_orders_equal']
    selection = json.loads((pair_root / 'HA/fit/selection.json').read_text())
    assert selection['full98_development_selection_complete'] and selection['strict_dense_reload']
    parent_path = Path(selection['best_checkpoint']).resolve()
    parent_digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    assert parent_digest == selection['selected']['checkpoint_sha256']
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    assert parent['model_family'] == 'CADTrackDense-R2' and parent['use_memory']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    names = sorted(path.name for path in Path(source['train_root']).iterdir() if path.is_dir())
    assert len(names) == 979
    development = set(random.Random(20261013).sample(names, 98))
    training = sorted(set(names) - development)
    assert len(training) == 881
    dataset = VideoClips(source['train_root'], training, 2, 42, steps=12)
    reference = DenseCADTracker(source['pretrained'], True)
    reference.load_payload(parent)
    tracker = DenseABCTracker(source['pretrained'])
    tracker.network.load_localizer_state(parent['dense'])
    tracker.network.requires_grad_(False).eval()
    before = module_hash(tracker.network)
    parity = []
    event = None
    with torch.inference_mode():
        for index in range(2):
            clip = dataset[index]
            initial = clip['boxes'][0].astype(np.float64)
            image = clip['images'][0]
            reference.initialize(image, initial)
            tracker.initialize(image, initial)
            for frame in range(1, 13):
                if index == 0 and frame == 5:
                    event = (clip, tracker.snapshot())
                expected = reference.track(clip['images'][frame])
                actual = tracker.track(clip['images'][frame])
                assert np.array_equal(actual['box'], expected['box'])
                assert actual['confidence'] == expected['confidence']
                assert actual['template_commit_rgb_tir'] == expected['template_commit_rgb_tir']
                assert actual['template_write_requested'] == expected['template_write_requested']
                assert actual['extra_forwards'] == 0 and actual['choice'] == 0
                assert actual['future_mode'] == 'policy' and actual['geometry_commit']
                assert tracker.box == reference.box
                assert all(torch.equal(a, b) and a.stride() == b.stride()
                    for a, b in zip(reference.templates, tracker.templates))
                assert all(torch.equal(reference.memory[key], tracker.state['memory'][key])
                    for key in reference.memory)
                parity.append(dict(sequence=clip['name'], clip_frame=frame,
                    original_frame=int(clip['frames'][frame]), exact_box_confidence_state=True,
                    template_write_requested=actual['template_write_requested'],
                    template_commit_rgb_tir=actual['template_commit_rgb_tir']))
        assert len(parity) == 24
        assert any(record['template_write_requested'] for record in parity)
        clip, prior = event
        tracker.restore(prior)
        image = clip['images'][5]
        local = tracker.observe_local(image)
        order = local['distribution']['log_weights'][0].argsort(descending=True, stable=True)
        contexts = [local] + [tracker.observe_extra(image, local, int(mode)) for mode in order]
        assert [len(query['native_boxes']) for query in contexts] == [5, 10, 10, 10]
        assert all(query['extra_forwards'] == 1 for query in contexts[1:])
        assert_same_snapshot(prior, tracker.snapshot())
        evidence_contexts = []
        for query in contexts:
            observed = score_context(tracker.network.candidate_evidence,
                query['pair'], query['appearance'], query['distribution'])
            for key in ('tokens', 'quality', 'risk', 'geometry'):
                assert torch.equal(observed[key], query['context'][key]), key
            boxes = torch.tensor(query['native_boxes'], dtype=torch.float64)
            quality = iou(xyxy(boxes), xyxy(torch.as_tensor(clip['boxes'][5]))[None]).float()
            evidence_contexts.append(dict(
                candidate={key: value.cpu() for key, value in query['pair'].items()
                    if key in ('boxes', 'raw', 'hann', 'fused')},
                appearance={key: value.cpu() for key, value in query['appearance'].items()}, quality=quality))
        evidence = dict(contexts=evidence_contexts,
            distribution={key: value.cpu() for key, value in local['distribution'].items()})
        action_checks = []
        for mode in ACTION_MODES:
            tracker.restore(prior)
            result = tracker.advance(image, local, 0, mode)
            assert tracker.frame == tracker.state['age'] == 5
            assert np.array_equal(result['box'], np.asarray(local['native_boxes'][0]))
            if mode == 'defer_geometry':
                assert not result['geometry_commit'] and not any(result['template_commit_rgb_tir'])
                for key in ('history', 'frames', 'valid', 'quality'):
                    assert torch.equal(prior['state'][key], tracker.state[key]), key
                assert all(torch.equal(prior['state']['memory'][key], value)
                    for key, value in tracker.state['memory'].items())
                assert all(torch.equal(a, b) and a.stride() == b.stride()
                    for a, b in zip(prior['state']['templates'], tracker.templates))
                first = tracker.network.forecast(tracker.state)
                query = tracker.observe_local(clip['images'][6])
                tracker.advance(clip['images'][6], query, 0, mode)
                second = tracker.network.forecast(tracker.state)
                assert int(second['observation_gap']) == int(first['observation_gap']) + 1
                assert bool((second['deviations'] > first['deviations']).all())
            if mode == 'pause_identity':
                assert not any(result['template_commit_rgb_tir'])
            if mode == 'rgb_only':
                assert not result['template_commit_rgb_tir'][1]
            if mode == 'tir_only':
                assert not result['template_commit_rgb_tir'][0]
            tracker.restore(prior)
            assert_same_snapshot(prior, tracker.snapshot())
            assert tracker.state['history'].data_ptr() != prior['state']['history'].data_ptr()
            action_checks.append(dict(mode=mode, real_native_submission_and_restore=True))
        tracker.restore(prior)
        consequences = collect_matched_query(tracker, image, clip['images'][6:9], clip['boxes'][5:9])
        assert consequences['actions'] == 210 and consequences['actual_native_replay_equal']
        assert consequences['prefix_state_unchanged_by_counterfactuals']
    evidence, consequences = ordinary_copy(evidence), ordinary_copy(consequences)
    torch.save(dict(sequence=clip['name'], sampled_clip_frames=clip['frames'].tolist(),
        evidence=evidence, consequences=consequences, qualification_only=True), output / 'real_train_query.pt')

    reports = {}
    for name, model, objective in [
        ('candidate_evidence', DenseCandidateEvidence().cuda(),
            lambda module: batch_objective(module, [evidence], torch.device('cuda'))[0]),
        ('decisions', DenseDecisionHeads().cuda(),
            lambda module: evaluate_batch(module, [consequences], torch.device('cuda'))[0])]:
        source_model = tracker.network.candidate_evidence if name == 'candidate_evidence' else tracker.network.decisions
        model.load_state_dict(source_model.state_dict(), strict=True)
        old = {key: value.detach().cpu().clone() for key, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        rows = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model)
            assert bool(torch.isfinite(loss))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm)) and float(norm) > 0
            optimizer.step()
            rows.append(dict(step=step + 1, loss=float(loss.detach()), gradient_norm=float(norm)))
        changed = sum(not torch.equal(old[key], value.detach().cpu()) for key, value in model.named_parameters())
        assert changed > 0
        payload = ordinary_copy(model.state_dict())
        clone = DenseCandidateEvidence() if name == 'candidate_evidence' else DenseDecisionHeads()
        clone.load_state_dict(payload, strict=True)
        assert all(torch.equal(value, clone.state_dict()[key]) for key, value in payload.items())
        reports[name] = dict(real_label_updates=rows, changed_parameter_tensors=changed, strict_reload=True)
        del model, optimizer, clone
    assert module_hash(tracker.network) == before
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == parent_digest
    receipt = dict(status='DENSE_R3_NATIVE_QUALIFICATION_PASS', parent_checkpoint=str(parent_path),
        parent_checkpoint_sha256=parent_digest, parity=parity, actual_causal_query_frames=24,
        train_clips=2, action_checks=action_checks, actual_matched_actions=210,
        actual_native_replay_equal=True, reconstructed_calibration_contexts_equal=True,
        head_checks=reports, parent_tracker_parameters_unchanged=True,
        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']), elapsed_seconds=time.monotonic() - started,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
        scope='Finite sampled TRAIN clips and one matched query; not full-video validation or formal training.',
        probe_heads_used_as_formal_initialization=False, formal_metrics=None, goal_achieved=False)
    (output / 'qualification_complete.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
