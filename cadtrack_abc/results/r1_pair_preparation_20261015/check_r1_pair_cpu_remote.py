"""CPU head sanity and synthetic command-path checks; never formal results."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

project = Path('/home/gaob/CADTrack-ABC')
work = project / 'outputs/r1_pair_source_checks_20261015'
sys.path.insert(0, str(project))
torch.set_num_threads(4)

def load(name):
    spec = importlib.util.spec_from_file_location('cadtrack_abc.' + name, work / 'sources' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

pipeline = load('future_pipeline')
trainer = load('train_future_actions')
pair = load('r1_pair_pipeline')
from cadtrack_abc.future_actions import FutureActionHead
from cadtrack_abc.search_actions import SearchActionHead

def order(policy, isolated):
    torch.manual_seed(42)
    loader = DataLoader(torch.arange(64), batch_size=8, shuffle=True,
        generator=torch.Generator().manual_seed(42) if isolated else None)
    FutureActionHead()
    if policy == 'search_actions':
        SearchActionHead()
    return torch.cat(list(loader)).tolist()

unisolated = [order(policy, False) for policy in ['future_actions', 'search_actions']]
isolated = [order(policy, True) for policy in ['future_actions', 'search_actions']]
assert unisolated[0] != unisolated[1] and isolated[0] == isolated[1]
checks = [dict(case='actual_head_initialization_shuffle', old_orders_differ=True,
               isolated_orders_equal=True, isolated_first16=isolated[0][:16], synthetic_indices=64)]

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

with tempfile.TemporaryDirectory(dir=work) as temporary:
    temporary = Path(temporary)
    teacher_path = project / 'outputs/cad_abc_20261013/best.pth'
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    for policy in ['future_actions', 'search_actions']:
        output = temporary / ('real_cpu_' + policy)
        args = ['train', '--config', str(project / 'cadtrack_abc/config.json'), '--teacher', str(teacher_path),
            '--manifest', str(project / 'outputs/cad_abc_20261013/manifest.json'),
            '--train-index', str(project / 'outputs/actual_batch64_collector_20261014/packets/collection_complete.json'),
            '--output', str(output), '--batch', '2', '--workers', '0', '--device', 'cpu', '--policy', policy, '--sanity']
        with patch.object(sys, 'argv', args):
            trainer.main()
        receipt = json.loads((output / 'fit_complete.json').read_text())
        assert receipt['updates'] == 2 and receipt['strict_head_reload']
        assert receipt['training_hyperparameters'] == dict(batch=2, lr=1e-4, seed=42, teacher_policy='native')
        assert receipt['train_index'].endswith('actual_batch64_collector_20261014/packets/collection_complete.json')
        assert receipt['validation_index'] is None
        payload = torch.load(receipt['candidates'][-1]['checkpoint'], map_location='cpu', weights_only=False)
        assert all(torch.equal(value, teacher['abc'][key]) for key, value in payload['abc'].items())
        FutureActionHead().load_state_dict(payload['future_action'], strict=True)
        if policy == 'search_actions':
            SearchActionHead().load_state_dict(payload['search_action'], strict=True)
        checks.append(dict(case='real_two_packet_cpu_' + policy, updates=2,
            train_queries=receipt['train_queries'], strict_reload=True, frozen_abc_exact=True,
            receipt_metadata_verified=True, temporary_weights_retired_by_fixture_cleanup=True))

    inventories = {'lasher': dict(sequences=245, frames=220703),
                   'rgbt210': dict(sequences=210, frames=104706), 'rgbt234': dict(sequences=234, frames=116649)}
    for case in ['full_pair', 'failed_fit', 'wrong_reused_packet']:
        root = temporary / case
        parent = root / 'parent'
        parent.mkdir(parents=True)
        teacher_marker = parent / 'best.pth'
        teacher_marker.write_bytes(b'synthetic_teacher_not_a_model')
        teacher_sha = hashlib.sha256(teacher_marker.read_bytes()).hexdigest()
        train_names = [f'train_{i:03}' for i in range(881)]
        val_names = [f'validation_{i:03}' for i in range(98)]
        manifest = parent / 'manifest.json'
        write(manifest, dict(train_sequences=train_names, validation_sequences=val_names))
        write(parent / 'complete.json', dict(status='TRAINING_COMPLETE', epochs=20, updates=3140,
            strict_reload_passed=True, best_checkpoint=str(teacher_marker)))
        write(parent / 'pipeline_complete.json', dict(training_complete=True, best_checkpoint=str(teacher_marker)))
        write(parent / 'lasher_report/full_report.json', dict(completed=True, dataset='lasher',
            sequences=245, frames=220703, checkpoint=str(teacher_marker), overall_metrics_percent=dict(PR=1, NPR=1, SR=1)))
        config = root / 'config.json'
        write(config, dict(physical_gpus=[2, 3], output=str(parent), test_root='/fixture/lasher', seed=42))
        executed, fit_devices, rounds, scores = [], {}, [], []
        output = root / 'paired'
        pid_counter = [700000]

        class CommandFixture:
            def __init__(self, command, **kwargs):
                self.code = 0
                pid_counter[0] += 1
                self.pid = pid_counter[0]
                module, args = command[2], command[3:]
                def arg(name):
                    return args[args.index(name) + 1]
                executed.append(module)
                if module == 'cadtrack_abc.merge_action_packets':
                    split, destination = arg('--split'), Path(arg('--output'))
                    names = train_names if split == 'train' else val_names
                    write(destination, dict(completed=True, teacher_sha256=teacher_sha, sequences=len(names),
                        queries=len(names), records=[dict(sequence=name, frame=3,
                            file=str(root / 'packet_files' / (name + '.pt'))) for name in names]))
                elif module == 'cadtrack_abc.train_future_actions':
                    policy, fit = arg('--policy'), Path(arg('--output'))
                    fit_devices[policy] = kwargs['env']['CUDA_VISIBLE_DEVICES']
                    if case == 'failed_fit' and policy == 'search_actions':
                        self.code = 1
                    candidates = []
                    for epoch in [0, 1]:
                        path = fit / f'candidate_epoch{epoch:02}.pth'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'fixture_candidate_not_a_model')
                        candidates.append(dict(epoch=epoch, checkpoint=str(path)))
                    train_index = Path(arg('--train-index'))
                    if case == 'wrong_reused_packet' and policy == 'future_actions':
                        bad = json.loads(train_index.read_text())
                        bad['records'][0]['file'] = str(root / 'wrong_packet.pt')
                        train_index = root / 'bad_index.json'
                        write(train_index, bad)
                    write(fit / 'fit_complete.json', dict(status='SEARCH_ACTION_FIT_COMPLETE' if policy == 'search_actions' else 'FUTURE_HEAD_FIT_COMPLETE',
                        policy_family=policy, teacher_sha256=teacher_sha, epochs=20, updates=280, train_queries=881,
                        train_sequences=881, strict_head_reload=True, changed_head_tensors=['synthetic'], candidates=candidates,
                        train_index=str(train_index), validation_index=arg('--validation-index'),
                        training_hyperparameters=dict(batch=64, lr=1e-4, seed=42, teacher_policy='native')))
                elif module == 'cadtrack_abc.future_pipeline':
                    rounds.append(arg('--policy'))
                    with patch.object(sys, 'argv', ['pipeline', *args]):
                        pipeline.main()
                elif module == 'cadtrack_abc.validate_future_actions':
                    write(Path(arg('--output')) / f'development_shard{arg("--shard")}.json',
                        dict(completed=True, full_videos=True, sanity=False, sequences=49))
                elif module == 'cadtrack_abc.select_future_actions':
                    fit = Path(arg('--fit'))
                    (fit / 'best.pth').write_bytes(b'synthetic_best_not_a_model')
                    (fit / 'trained_best.pth').write_bytes(b'synthetic_trained_not_a_model')
                    write(fit / 'selection.json', dict(full98_development_selection_complete=True, strict_future_head_reload=True,
                        best_checkpoint=str(fit / 'best.pth'), trained_best_checkpoint=str(fit / 'trained_best.pth'), selected={'epoch': 0}))
                elif module == 'cadtrack_abc.infer':
                    dataset, shard = arg('--dataset'), int(arg('--shard'))
                    inventory = inventories[dataset]
                    write(Path(arg('--output')) / f'complete_shard{shard}.json',
                        dict(completed=True, policy_family=arg('--policy'), checkpoint=arg('--checkpoint'), dataset=dataset,
                            sequences=inventory['sequences']//2 + (inventory['sequences']%2 if shard == 0 else 0),
                            frames=inventory['frames']//2 + (inventory['frames']%2 if shard == 0 else 0)))
                elif module == 'cadtrack_abc.score':
                    dataset, report = arg('--dataset'), Path(arg('--output'))
                    receipt = json.loads((Path(arg('--predictions')) / 'complete_shard0.json').read_text())
                    metrics = {'lasher': dict(PR=1, NPR=1, SR=1), 'rgbt210': dict(PR=1, SR=1), 'rgbt234': dict(MPR=1, MSR=1)}[dataset]
                    write(report / 'full_report.json', dict(completed=True, dataset=dataset, checkpoint=receipt['checkpoint'],
                        epoch=1 if 'trained_best' in receipt['checkpoint'] else 0, **inventories[dataset],
                        overall_metrics_percent=metrics, lasher_stage_goal={'all_three_reached': False}))
                    scores.append(dict(dataset=dataset, checkpoint=receipt['checkpoint']))
                else:
                    raise AssertionError(module)
            def wait(self):
                return self.code

        arguments = ['pair', '--config', str(config), '--output', str(output),
            '--sampling-root', str(root / 'sampling'), '--collection-root', str(root / 'collection'),
            '--rgbt210-root', '/fixture/rgbt210', '--rgbt234-root', '/fixture/rgbt234']
        sample = lambda folder, names, teacher: [{'queries': len(names)//2}, {'queries': len(names)-len(names)//2}]
        collect = lambda folders, sampling, teacher: [dict(completed=True, teacher_sha256=teacher,
            model_family='CADTrackABC-native-current-quality', queries=count)
            for count in ([440, 441] if Path(sampling).name == 'queries_train' else [49, 49])]
        with patch.object(sys, 'argv', arguments), patch.object(pair.subprocess, 'Popen', CommandFixture), \
             patch.object(pair, 'validate_benchmark_inventory', lambda dataset, root: inventories[dataset]), \
             patch.object(pipeline, 'validate_benchmark_inventory', lambda dataset, root: inventories[dataset]), \
             patch.object(pair, 'validate_sampling', sample), patch.object(pipeline, 'validate_sampling', sample), \
             patch.object(pair, 'validate_collection', collect), patch.object(pipeline, 'validate_collection', collect):
            if case == 'full_pair':
                pair.main()
            else:
                with TestCase().assertRaises(AssertionError):
                    pair.main()
        assert fit_devices == {'future_actions': '2', 'search_actions': '3'}
        assert executed.count('cadtrack_abc.train_future_actions') == 2  # Reused rounds never refit.
        if case == 'full_pair':
            complete = json.loads((output / 'pipeline_complete.json').read_text())
            assert complete['status'] == 'R1_MATCHED_FITS_AND_FULL_ROUNDS_COMPLETE'
            assert rounds == ['future_actions', 'search_actions'] and len(scores) == 12
        else:
            assert not (output / 'pipeline_complete.json').exists()
            assert not scores
            assert all((output / f'fit_{policy}/candidate_epoch01.pth').exists() for policy in ['future_actions', 'search_actions'])
            if case == 'failed_fit':
                assert not rounds and json.loads((output / 'queue_state.json').read_text())['status'] == 'FAILED'
            else:
                assert 'cadtrack_abc.validate_future_actions' not in executed
        checks.append(dict(case=case, fits=2, fit_devices=fit_devices, full_rounds=rounds,
            synthetic_reports=len(scores), reused_fits_not_rerun=True, real_training=False, real_metrics=False))
assert not torch.cuda.is_initialized()
result = dict(status='R1_PAIR_CPU_HEAD_AND_COMMAND_CHECKS_PASS', checks=checks,
    actual_small_head_optimizer_updates=4, actual_packets_per_policy=2, cuda_initialized=False,
    full_main_fit_started=False, actual_rgbt210_inventory=False,
    scope='Actual CPU head updates/reloads and synthetic orchestration receipts. No vision inference or formal benchmark result; temporary fixture weights cleaned.')
(work / 'cpu_checks.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
