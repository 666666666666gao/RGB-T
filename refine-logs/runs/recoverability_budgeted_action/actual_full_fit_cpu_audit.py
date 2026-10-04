"""Read-only actual fit/replay and LasHeR native audit. Run with CUDA_VISIBLE_DEVICES=''."""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
sys.dont_write_bytecode = True
PRIVATE = Path('/data/gb/experiments/recoverability_budgeted_action_20261004')
MAIN = Path('/data/gb/GOLA')
OUTPUTS = Path('/data/gb/outputs')
CACHE = OUTPUTS / 'recoverability_future_policy_merged_20261004'
SETUP = Path('/data/gb/setup')
sys.path[:0] = [str(PRIVATE), str(MAIN)]
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(Path(path).read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def common():
    return {'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional',
            'reviewer_agent': '/root/budgeted_full_fit_acceptance',
            'model_metadata': 'Codex/GPT-6 family; exact runtime model/effort not independently exposed',
            'audited_utc': datetime.now(timezone.utc).isoformat(), 'checker': str(Path(__file__)),
            'checker_sha256': sha(__file__), 'blocking_findings': [],
            'execution': {'CUDA_VISIBLE_DEVICES': '', 'GPU_queries': 0, 'training': False,
                          'source_model_data_modified': False, 'job_commands_or_processes_changed': False}}


def overlap(boxes, target):
    inter = np.maximum(np.minimum(boxes[..., 2:], target[..., 2:])
                       - np.maximum(boxes[..., :2], target[..., :2]), 0).prod(-1)
    area = np.maximum(boxes[..., 2:] - boxes[..., :2], 0).prod(-1)
    other = np.maximum(target[..., 2:] - target[..., :2], 0).prod(-1)
    return inter / np.maximum(area + other - inter, 1e-8)


def fit_audit():
    import torch
    import research.recoverability_modules as modules
    import research.train_recoverability as trainer
    from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped
    assert Path(modules.__file__).parent == Path(trainer.__file__).parent == PRIVATE / 'research'
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    prior_path = SETUP / 'budgeted_action_actual_m0_cpu_audit_20261004.json'
    prior = read(prior_path)
    old_cache = read(SETUP / 'future_policy_actual_m0_cpu_audit_20261004.json')
    assert prior['status'] == old_cache['status'] == 'PASS'
    assert all(sha(p) == h for p, h in prior['source_sha256'].items())
    for filename, changed, added in [('recoverability_modules.py', ['objective'], {'budgeted_winner_loss'}),
                                     ('train_recoverability.py', ['arguments'], set())]:
        main = definitions(MAIN / 'research' / filename)
        private = definitions(PRIVATE / 'research' / filename)
        assert set(private) - set(main) == added and not set(main) - set(private)
        assert [k for k in main if main[k] != private[k]] == changed
    assert not set(modules.DECISION_FIELDS) & set(trainer.LABEL_FIELDS)
    assert set(trainer.LABEL_FIELDS) == {'current_iou', 'future_iou', 'wrong_update_fraction', 'action_valid', 'history_iou'}
    c1_path = OUTPUTS / 'c1_initial_seed42/best.pth'
    assert sha(c1_path) == prior['frozen_C1_source_sha256']
    c1 = torch.load(c1_path, map_location='cpu', weights_only=False)
    cache_config = read(CACHE / 'c1/train/config.json')
    ds = MultiModalObjectTrackingDataset_MemoryMapped.load(cache_config['root'], str(MAIN / cache_config['cache']))
    sequences = {seq.get_name(): seq for seq in ds}
    assert any(name == 'DataCleaning_BoundingBox' and settings['fit_in_image_size']
               for name, settings in ds.dataset_attributes['filters'])
    cache_checks, zip_members = {}, {}
    for policy in ('c1', 'own'):
        for partition, count in [('train', 902), ('validation', 128)]:
            root = CACHE / policy / partition
            cfg, receipt = read(root / 'config.json'), read(root / 'completion.json')
            old = prior['cache_metadata_rechecks'][policy + '_' + partition]
            assert sha(root / 'config.json') == old['config_sha256']
            assert sha(root / 'completion.json') == old['completion_sha256']
            assert receipt['completed'] and not receipt['decision_input_contains_future']
            assert cfg['decision_inputs_copied_before_future_decode'] and cfg['max_prefix'] == 1024
            assert cfg['clips'] == receipt['clips'] == count and len(cfg['jobs']) == count
            for p, h in old_cache['cache_checks'][policy + '_' + partition]['source_receipt_sha256'].items():
                assert sha(p) == h
            with zipfile.ZipFile(root / 'samples.npz') as z:
                members = {i.filename: [i.CRC, i.file_size, i.compress_size] for i in z.infolist()}
            assert len(members) == 30
            assert hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest() == old['npz_archive_directory_sha256']
            zip_members[policy + '_' + partition] = members
            with np.load(root / 'samples.npz', allow_pickle=False) as z:
                small = {k: z[k] for k in ('image_boxes', 'current_iou', 'valid', 'motion_targets',
                                           'history_boxes', 'history_frames', 'history_valid', 'history_iou')}
            assert small['history_frames'].shape == (count, 1024)
            current_gap, history_gap, gt_clipped = 0., 0., []
            for i, job in enumerate(cfg['jobs']):
                seq, query = sequences[job['sequence']], job['query_frame']
                assert query <= 1024
                raw = np.loadtxt(Path(cfg['root']) / 'traingset' / job['sequence'] / 'init.txt', delimiter=',', ndmin=2)
                raw[:, 2:] += raw[:, :2]
                for offset in range(3):
                    f = query + offset
                    size = seq[f].get_image_size()[0]
                    clipped = np.minimum(np.maximum(raw[f], 0), np.tile(size, 2))
                    assert np.array_equal(clipped, seq[f].get_bounding_box()), (job, f)
                    assert np.array_equal(clipped.astype(np.float32), small['motion_targets'][i, offset])
                    if not np.array_equal(raw[f], clipped):
                        gt_clipped.append({'sequence': job['sequence'], 'query': query, 'frame': f,
                                           'raw_xyxy': raw[f].tolist(), 'loader_xyxy': clipped.tolist(),
                                           'image_size': size.tolist()})
                valid = small['valid'][i]
                err = abs(overlap(small['image_boxes'][i].astype(float), seq[query].get_bounding_box())
                          - small['current_iou'][i])
                current_gap = max(current_gap, float(err[valid].max()))
                hf, hv = small['history_frames'][i], small['history_valid'][i]
                assert (hf[hv] < query).all() and np.array_equal(hf[hv], np.arange(query))
                hgt = np.stack([seq[int(f)].get_bounding_box() for f in hf[hv]])
                known = np.isfinite(hgt).all(1) & (hgt[:, 2:] > hgt[:, :2]).all(1)
                expected = overlap(small['history_boxes'][i, hv].astype(float), hgt)
                history_gap = max(history_gap, float(abs(expected[known] - small['history_iou'][i, hv][known]).max()))
                assert (small['history_iou'][i, hv][~known] == -1).all()
            assert current_gap < 1e-5 and history_gap < 1e-5, (policy, partition, current_gap, history_gap)
            cache_checks[policy + '_' + partition] = {
                'clips': count, 'fields': 30, 'history_capacity': 1024, 'complete_causal_prefix_verified': True,
                'current_and_history_iou_actual_loader_GT_recomputed': True,
                'current_iou_max_abs_error_saved_float32_boxes': current_gap,
                'history_iou_max_abs_error': history_gap, 'motion_GT_targets_exact': True,
                'raw_GT_clipped_by_existing_loader': gt_clipped, 'unchanged_prior_cache_metadata': True,
                'future_label_boundary': 'Future rollout boxes were not saved; GT label derivation verified in collector source and prior shard-to-merge evidence, no future NN rollout repeated.'}
    for part in ('train', 'validation'):
        a, b = [zip_members[p + '_' + part] for p in ('c1', 'own')]
        assert all(a[k] == b[k] for k in a if k not in ('future_iou.npy', 'wrong_update_fraction.npy'))
    from PIL import Image
    clipped_image = Path('/data/wangwj/dataset/LasHeR/traingset/10phone_boy/visible/v018.jpg')
    with Image.open(clipped_image) as image:
        assert image.size == (960, 576)
    arms = {}
    for policy in ('c1', 'own'):
        root = OUTPUTS / ('recoverability_budgeted_action_' + policy + '_b384_full_20261004')
        cfg, receipt, metrics = [read(root / p) for p in ('config.json', 'completion.json', 'metrics.json')]
        control = OUTPUTS / ('recoverability_future_' + policy + '_b384_full_20261004')
        cc, cr = read(control / 'config.json'), read(control / 'completion.json')
        differences = [k for k in cfg if cfg[k] != cc[k]]
        assert set(cfg) == set(cc) and set(differences) == {'output', 'action_ranking'}
        assert cc['action_ranking'] == 'pairwise' and cr['completed'] and cr['optimizer_steps'] == 180
        assert cfg['epochs'] == receipt['epochs'] == 60 and cfg['batch_size'] == 384 and cfg['seed'] == 42
        assert cfg['lr'] == cfg['weight_decay'] == 1e-4 and cfg['threshold'] == .03
        assert cfg['action_ranking'] == 'budgeted' and cfg['search_supervision'] == 'oracle'
        assert cfg['write_verification'] == 'action' and not cfg['write_pair_calibration']
        assert cfg['frozen_c1'] and cfg['initial_all_choices_match_c1'] and not cfg['prefer_last_prefix']
        assert cfg['train'] == [str(CACHE / policy / 'train')] and cfg['validation'] == str(CACHE / policy / 'validation')
        assert cfg['source_configs'] == [read(CACHE / policy / p / 'config.json') for p in ('train', 'validation')]
        split = read(cfg['source_configs'][0]['split'])
        for part, n in [('train', 902), ('validation', 128)]:
            jobs = cfg[part + '_jobs']
            assert len(jobs) == len({tuple(j) for j in jobs}) == n
            assert {j[0] for j in jobs} <= set(split[part])
        assert len(split['train']) == 881 and len(split['validation']) == 98
        assert not set(split['train']) & set(split['validation'])
        logs = [json.loads(l) for l in (root / 'train.jsonl').read_text().splitlines()]
        triplets = [[r['epoch'], r['step'], r['optimizer_steps']] for r in logs]
        assert triplets == [[e, s, (e - 1) * 3 + s] for e in range(1, 61) for s in (1, 3)]
        assert receipt['completed'] and receipt['optimizer_steps'] == 60 * math.ceil(902 / 384) == 180
        assert all(np.isfinite(r['loss']) and all(np.isfinite(v) for v in r['parts'].values()) for r in logs)
        assert all(set(r['module_gradient_norms']) == {'A', 'B', 'C'} for r in logs)
        assert all(np.isfinite(v) and v > 0 for r in logs for v in r['module_gradient_norms'].values())
        assert receipt['modules_changed'] == {'A': True, 'B': True, 'C': True}
        assert all(np.isfinite(v) and v >= max(r['module_gradient_norms'][k] for r in logs)
                   for k, v in receipt['max_module_gradient_norms'].items())
        assert receipt['frozen_c1_gradients_absent'] and receipt['strict_reload_metrics_equal']
        assert receipt['retained_weights'] == ['best.pth'] and not receipt['official_tracking_accuracy']
        assert [m['epoch'] for m in metrics] == list(range(61))
        assert all(all(np.isfinite(v) for v in m.values()) for m in metrics)
        best = max(range(61), key=lambda e: metrics[e]['utility'])
        ck = torch.load(root / 'best.pth', map_location='cpu', weights_only=False)
        assert best == ck['epoch'] == receipt['best_epoch'] == 25
        assert ck['validation'] == receipt['best_validation'] == {k: v for k, v in metrics[best].items() if k != 'epoch'}
        assert receipt['last_validation'] == {k: v for k, v in metrics[-1].items() if k != 'epoch'}
        assert all(cfg[k] == v for k, v in ck['args'].items())
        torch.manual_seed(42)
        model = modules.RecoverabilityModules(c1).cpu().eval()
        fresh = model.state_dict()
        changes = {}
        for name, prefix in [('A', 'memory.'), ('B', 'search.'), ('C', 'action.')]:
            keys = [k for k in fresh if k.startswith(prefix)]
            altered = [k for k in keys if not torch.equal(fresh[k], ck['model'][k])]
            assert altered
            changes[name] = {'changed_tensors': len(altered), 'total_tensors': len(keys),
                             'changed_elements': sum(int((fresh[k] != ck['model'][k]).sum()) for k in keys),
                             'max_abs_change': max(float((fresh[k] - ck['model'][k]).abs().max()) for k in keys)}
        model.load_state_dict(ck['model'], strict=True)
        assert all(p.device.type == 'cpu' for p in model.parameters())
        assert all(torch.isfinite(t).all() for t in model.state_dict().values())
        assert all(torch.equal(ck['model']['c1.' + k], v) for k, v in c1['head'].items())
        assert all(not p.requires_grad and p.grad is None for p in model.c1.parameters())
        counts = {k: sum(p.numel() for p in m.parameters()) for k, m in [('A', model.memory), ('B', model.search), ('C', model.action)]}
        assert counts == cfg['trainable_parameters'] == {'A': 109187, 'B': 117133, 'C': 86791}
        data, _, names, jobs = trainer.load_data([cfg['validation']], 'validation', torch.device('cpu'))
        assert [list(j) for j in jobs] == cfg['validation_jobs']
        measured, values = trainer.evaluate(model, data, 384, .03, details=True, search_supervision='oracle',
                                           action_ranking='budgeted', write_verification='action')
        with np.load(root / 'best_validation.npz', allow_pickle=False) as z:
            saved = {k: z[k] for k in z.files}
        assert set(saved) == set(values) and len(saved) == 21
        errors = {}
        for k, v in values.items():
            assert v.shape == saved[k].shape == (128,)
            errors[k] = float(np.abs(v.astype(float) - saved[k].astype(float)).max())
            if v.dtype.kind in 'biu':
                assert np.array_equal(v, saved[k]), (policy, k)
            else:
                assert errors[k] <= np.finfo(np.float32).eps, (policy, k, errors[k])
        metric_errors = {k: abs(v - ck['validation'][k]) for k, v in measured.items()}
        assert set(measured) == set(ck['validation'])
        assert all(v <= 2e-6 for v in metric_errors.values()), metric_errors
        # CPU reproduction of the online memory commit rule, with the same persisted observations.
        i = int(data['history_valid'].sum(1).argmax())
        one = {k: v[i:i + 1] for k, v in data.items()}
        anchor, rebuilt, _, _ = model.memory.history(one)
        state = model.memory.initialize(anchor)
        slots = torch.where(one['history_valid'][0] & (one['history_frames'][0] > 0))[0]
        for slot in slots:
            descriptor = model.memory.encode(one['history_instance_descriptors'][:, slot].half().float())
            probability = model.memory.gate_logits(descriptor, one['history_evidence'][:, slot].float(),
                                                    one['history_quality'][:, slot].float(), anchor).softmax(-1)
            observed = torch.ones(1, dtype=torch.bool)
            commit = (probability[..., 0] >= .5) & one['history_write'][:, slot, None]
            state = model.memory.update(state, descriptor, probability, observed, commit)
        memory_gap = float((state - rebuilt).abs().max())
        assert memory_gap <= 2e-6, memory_gap
        inf = read(root / 'predictions/inference_config.json')
        assert inf['head_epoch'] == 25 and inf['model'] == str(root / 'best.pth')
        assert inf['validation_split'] == str(OUTPUTS / 'c1_initial_seed42/split.json')
        assert inf['max_frames'] == inf['limit_sequences'] == inf['sequence_offset'] == 0
        assert not inf['disable_search'] and not inf['unsafe_writes'] and inf['write_verification'] == 'action'
        progress = [json.loads(l) for l in (root / 'predictions/progress.jsonl').read_text().splitlines()]
        assert all(r['sequences'] == 98 and r['sequence'] in split['validation'] for r in progress)
        arms[policy] = {'root': str(root), 'epochs': 60, 'optimizer_steps': 180, 'logged_rows': len(logs),
            'logged_steps': 'First and third update per epoch; all 120 observed ABC gradient triplets finite and positive.',
            'best_epoch': best, 'all_61_epoch_metrics_and_selected_checkpoint_receipt_exact': True,
            'frozen_C1_tensors_exact': len(c1['head']), 'strict_CPU_load': True,
            'ABC_best25_changes_vs_fresh_seed42': changes, 'ABC_parameter_counts': counts,
            'paired_control': str(control), 'config_differences_vs_pairwise': differences,
            'deleted_control_or_M0_weights_read_or_recreated': False,
            'best_validation': receipt['best_validation'], 'CPU_replay_128_queries': True,
            'all_discrete_NPZ_decisions_exact': True, 'per_field_max_replay_abs_error': errors,
            'per_metric_max_replay_abs_error': metric_errors,
            'CPU_same_observation_incremental_memory_check': {'job': jobs[i], 'commits_observed': len(slots),
                'max_abs_error': memory_gap, 'scope': 'Same parameters and cached complete causal history; CPU arithmetic, not an independent visual or GPU trajectory replay.'},
            'inference_snapshot': {'completed_sequences': len(progress), 'expected_sequences': 98,
                'latest_sequence': progress[-1]['sequence'], 'accuracy_claimed': False,
                'completion_receipt_present': (root / 'predictions/inference_completion.json').is_file()},
            'artifacts_sha256': {p: sha(root / p) for p in ('config.json', 'completion.json', 'metrics.json', 'train.jsonl', 'best.pth', 'best_validation.npz')}}
        print('FIT_ARM_PASS', policy, best, max(metric_errors.values()), flush=True)
        del model, data, ck
    assert not torch.cuda.is_initialized()
    result = common()
    result.update(scope='Completed 60 epoch fits and actual CPU cached validation replay; full 98-sequence online accuracy remains separately pending.',
        source_sha256=prior['source_sha256'], prior_cache_audit=str(prior_path), prior_cache_audit_sha256=sha(prior_path),
        cache_checks=cache_checks, arms=arms, all_inference_classes_unchanged_private_vs_main=True,
        label_boundary='Current/history labels and query GT targets independently checked against actual loader GT. Future labels derive from GT at future frames in collector source; future trajectories were not rerun.',
        clipped_GT_actual_image={'path': str(clipped_image), 'width_height': [960, 576]},
        memory_training_deployment={'training': 'Reconstructs A memory from anchor and all 1024 cached history slots, with valid/frame/write/gate masks; every sampled prefix observed from frame0 through query-1.',
            'online': 'Tracker.step passes persistent identity_memory to modules.decide; accept_observation incrementally applies the same encode, gate, verified-write and context-update rules.',
            'motion_window': 8, 'ABC_DECISION_FIELDS_contain_persistent_memory': False,
            'difference': 'Training recomputes memory along fixed teacher observations; online observations and template writes follow the deployed model trajectory. Last8 refers to motion/search geometry, not truncated A history.',
            'beyond_1024': 'Training caches do not establish performance or numerical equivalence on prefixes longer than1024. No such long visual trajectory replay performed.',
            'performance_cause_established': False},
        checker_corrections=[{'issue': 'Initial SSH alias wording interpreted literally as alias2027', 'fix': 'Use actual configured Host2027; no remote job issue.'},
            {'issue': 'Exploratory memory-mapped dataset constructor was passed a path', 'fix': 'Used its existing .load(root,path) API; no data or production change.'},
            {'issue': 'Raw GT was initially compared before existing loader clipping', 'fix': 'Verified actual fit_in_image_size filter, loader boxes and image960x576 for10phone_boy; retained native evaluation GT unchanged.'}])
    result['execution'].update(cuda_initialized=False, neural_forward='Two128-query CPU evaluations plus two single-prefix CPU memory replays', TRAIN_embeddings_loaded=False)
    return result


def native_audit(control):
    from rgbt import LasHeR
    evaluator = LasHeR()  # Names, packaged GT and attributes only; no native metric calls.
    root = OUTPUTS / ('recoverability_write_pair_native_lasher_' + control + '_own_b384_full_20261004')
    gt_root = Path('/data/wangwj/dataset/LasHeR/testingset')
    runs = {'baseline': OUTPUTS / 'online_core/lasher_baseline',
            'c1': OUTPUTS / 'online_core_v2/lasher_c1',
            'write_pair_reference_own_b384': OUTPUTS / 'recoverability_write_pair_native_lasher_own_b384_full_20261004',
            'write_pair_' + control + '_own_b384': root}
    candidate = 'write_pair_' + control + '_own_b384'
    report = read(root / 'core_report/full_report.json')
    names = list(evaluator.ALL)
    assert len(names) == 245 and {p.name for p in gt_root.iterdir() if p.is_dir()} == set(names)
    assert report['sequences'] == 245 and report['frames'] == 220703 and set(report['variants']) == set(runs)
    gt = {n: np.loadtxt(gt_root / n / 'init.txt', delimiter=',', dtype=np.float32, ndmin=2) for n in names}
    assert sum(map(len, gt.values())) == 220703
    assert all(np.array_equal(gt[n], np.asarray(evaluator.seqs_gt[n], dtype=np.float32)) for n in names)
    with (root / 'core_report/per_sequence.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 245
    recorded = {(r['variant'], r['sequence']): r for r in rows}
    thresholds = {'NPR': np.linspace(0, .5, 51), 'PR': np.linspace(0, 50, 51), 'SR': np.linspace(0, 1, 21)}
    scores, overall, counters = {}, {}, {}
    for variant, run in runs.items():
        prediction = run / 'predictions'
        receipt = read(prediction / 'inference_completion.json')
        assert receipt['completed'] and not receipt['smoke_only'] and receipt['sequences'] == 245 and receipt['frames'] == 220703
        assert {p.stem for p in prediction.glob('*.txt')} == set(names)
        assert {r['sequence'] for r in receipt['records']} == set(names)
        assert all(r['frames'] == len(gt[r['sequence']]) for r in receipt['records'])
        curves = {m: [] for m in thresholds}
        for n in names:
            p = np.loadtxt(prediction / (n + '.txt'), dtype=np.float32, ndmin=2).round(0)
            g = gt[n]
            assert p.shape == g.shape and np.isfinite(p).all() and (p[1:, 2:] > 0).all()
            p[0] = g[0]
            pc, gc = (p[:, 2:] - 1) / 2 + p[:, :2], (g[:, 2:] - 1) / 2 + g[:, :2]
            pr = ((pc - gc) ** 2).sum(1) ** .5
            npr = ((pc / (g[:, 2:] + 1e-8) - gc / (g[:, 2:] + 1e-8)) ** 2).sum(1) ** .5
            right = np.minimum(p[:, :2] + p[:, 2:] - 1, g[:, :2] + g[:, 2:] - 1)
            left = np.maximum(p[:, :2], g[:, :2])
            intersection = np.maximum(right - left + 1, 0).prod(1)
            sr = intersection / (p[:, 2:].prod(1) + g[:, 2:].prod(1) - intersection)
            unknown = (g <= 0).any(1)
            for metric, values in [('NPR', npr), ('PR', pr), ('SR', sr)]:
                values[unknown] = -1
                curve = (values[:, None] > thresholds[metric] if metric == 'SR'
                         else values[:, None] <= thresholds[metric]).mean(0)
                curves[metric].append(curve)
                value = float(curve.mean() if metric == 'SR' else curve[20]) * 100
                assert value == float(recorded[variant, n][metric]), (variant, n, metric, value, recorded[variant, n][metric])
        curves = {m: np.array(c) for m, c in curves.items()}
        overall[variant] = {m: float(c.mean() if m == 'SR' else c.mean(0)[20]) * 100 for m, c in curves.items()}
        assert overall[variant] == report['variants'][variant]['overall_metrics_percent']
        for m, c in curves.items():
            assert np.array_equal(c.mean(0), report['variants'][variant]['mean_curves'][m]['values'])
            assert np.array_equal(thresholds[m], report['variants'][variant]['mean_curves'][m]['thresholds'])
        for attr in evaluator.get_attr_list():
            indices = [names.index(n) for n in getattr(evaluator, attr)]
            values = {m: float(c[indices].mean() if m == 'SR' else c[indices].mean(0)[20]) * 100 for m, c in curves.items()}
            assert {'sequences': len(indices), **values} == report['variants'][variant]['attributes'][attr]
        scores[variant] = {n: {m: float(c[i].mean() if m == 'SR' else c[i, 20]) * 100
                              for m, c in curves.items()} for i, n in enumerate(names)}
        if variant == candidate:
            for n in names:
                with np.load(prediction / (n + '_recoverability_decisions.npz')) as z:
                    assert len(z['region']) == len(gt[n]) - 1
                    if control == 'no_search':
                        assert not z['search_requested'].any() and not z['extra_executed'].any() and not z['region'].any()
                    else:
                        assert not z['pause'].any()
                        raw = z['raw_score'].reshape(len(gt[n]) - 1, 35)
                        chosen_raw = raw[np.arange(len(raw)), z['choice']]
                        assert np.array_equal(z['template_updated'], chosen_raw > .84)
            counters = {k: sum(r['stats'][k] for r in receipt['records']) for k in ('extra_searches_requested', 'extra_visual_forwards', 'template_updates', 'paused_query_writes', 'changed_candidate_indices')}
            if control == 'no_search':
                assert counters['extra_searches_requested'] == counters['extra_visual_forwards'] == 0
            else:
                assert counters['paused_query_writes'] == 0
        print('NATIVE_VARIANT_PASS', variant, overall[variant], flush=True)
    config = read(root / 'predictions/inference_config.json')
    full_config = read(runs['write_pair_reference_own_b384'] / 'predictions/inference_config.json')
    flag = 'disable_search' if control == 'no_search' else 'unsafe_writes'
    assert config[flag] and not full_config[flag]
    differences = [k for k in config if config[k] != full_config[k]]
    assert set(differences) == {'output', flag, 'bootstrap_training_future_policy'}
    assert config['bootstrap_training_future_policy'] == 'frozen continuation policy recorded in training source_configs; not recomputed by this evaluator'
    assert full_config['bootstrap_training_future_policy'] == 'frozen C1 continuations, not this deployed policy'
    assert config['head_epoch'] == 4 and config['motion_history_capacity'] == 8
    assert config['frozen_motion_memory_and_predictor_parameters'] == 158412
    bootstrap = {}
    for reference in ('baseline', 'c1', 'write_pair_reference_own_b384'):
        folder = root / (reference + '_paired_report')
        paired = read(folder / 'full_report.json')
        artifact = read(folder / 'paired_bootstrap.json')
        assert artifact['args']['iterations'] == 5000 and artifact['args']['seed'] == 42
        assert list(paired['variants']) == [reference, candidate]
        assert paired['variants'][reference]['overall_metrics_percent'] == overall[reference]
        assert paired['variants'][candidate]['overall_metrics_percent'] == overall[candidate]
        ordered = sorted(names)
        draws = np.random.default_rng(42).integers(245, size=(5000, 245))
        metric_results = {}
        for metric in thresholds:
            delta = np.array([scores[candidate][n][metric] - scores[reference][n][metric] for n in ordered])
            metric_results[metric] = {'mean_delta_percentage_points': float(delta.mean()),
                'percentile_95_interval_percentage_points': np.percentile(delta[draws].mean(1), (2.5, 97.5)).tolist(),
                'improved_sequences': int((delta > 0).sum()), 'worsened_sequences': int((delta < 0).sum()),
                'tied_sequences': int((delta == 0).sum())}
        assert metric_results == artifact['datasets']['lasher']['metrics']
        bootstrap[reference] = metric_results
    result = common()
    result.update(scope='Completed LasHeR ' + control + ' control; actual GT and all four variants independently rescored with NumPy.',
        root=str(root), dataset='lasher', actual_GT=str(gt_root), sequences=245, frames=220703,
        completeness='All980 predictions and all980 receipt frame counts checked; all245 candidate diagnostic NPZ control-action arrays checked.',
        packaged_GT_equals_actual_GT=True, native_arithmetic='float32 rounded predictions; GT initial row included; strict centers and inclusive IoU; GT any coordinate<=0 maps to-1; NPR/PR <= threshold, SR > threshold; equal sequence weights.',
        native_invalid_prediction_size_replacements=0, native_metric_function_calls=0,
        all_980_native_sequence_triples_exact=True, all_19_attributes_and_mean_curves_per_variant_exact=True,
        native_overall_percent=overall, paired_5000_seed42_bootstrap_exact=bootstrap,
        observed_extra_search_control_counters=counters,
        config_differences_vs_full=differences,
        control_scope=('Disables additional NN visual search only. Frozen B motion memory/predictor remains active; this is not full B-off.'
                       if control == 'no_search' else 'Forces query template writes at raw_score>.84 regardless of learned pause; does not disable learned search or A memory.'),
        descriptive_config_change='bootstrap_training_future_policy changed wording only; verified actual strings, same checkpoint and remaining execution configuration.',
        checker_corrections=[{'issue': 'First no_search audit config assertion omitted changed bootstrap_training_future_policy description',
                             'observed': 'All980 native triples/attributes/curves already passed before assertion.',
                             'fix': 'Include this exact descriptive string difference. No production change and no metric tolerance relaxation.'}],
        uncertainty_scope='Paired sequence uncertainty for fixed checkpoints, not training-seed variation.',
        artifacts_sha256={p: sha(root / p) for p in ('predictions/inference_config.json', 'predictions/inference_completion.json', 'core_report/full_report.json', 'core_report/per_sequence.csv')})
    result['execution']['neural_forward'] = False
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('section', choices=['fit', 'native', 'native-raw'])
    args = parser.parse_args()
    start = time.perf_counter()
    result = fit_audit() if args.section == 'fit' else native_audit('no_search' if args.section == 'native' else 'raw_write')
    result['execution']['elapsed_seconds'] = time.perf_counter() - start
    out = SETUP / {'fit': 'budgeted_action_actual_full_fit_cpu_audit_20261004.json',
                   'native': 'own4_native_no_search_lasher_independent_cpu_audit_20261004.json',
                   'native-raw': 'own4_native_raw_write_lasher_independent_cpu_audit_20261004.json'}[args.section]
    with out.open('w') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print('AUDIT_PASS', args.section, str(out), result['execution']['elapsed_seconds'], flush=True)
