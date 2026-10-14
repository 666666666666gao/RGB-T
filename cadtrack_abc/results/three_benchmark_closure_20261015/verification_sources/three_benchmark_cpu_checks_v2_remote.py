"""CPU source fixtures; generated receipts are not real benchmark results."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
from unittest.mock import patch

import torch

project = Path('/home/gaob/CADTrack-ABC')
work = project / 'outputs/three_benchmark_source_checks_20261015'
sys.path.insert(0, str(project))
torch.set_num_threads(2)

def load(name, file):
    spec = importlib.util.spec_from_file_location('cadtrack_abc.' + name, work / 'sources' / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

selection = load('selection_under_test', 'select_future_actions.py')
pipeline = load('pipeline_under_test', 'future_pipeline.py')
from cadtrack_abc.future_actions import FutureActionHead
from cadtrack_abc.search_actions import SearchActionHead

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

checks = []
with tempfile.TemporaryDirectory(dir=work) as temporary:
    root = Path(temporary)
    for case, values, selected_epoch in [('epoch0_wins', [.9, .8, .7], 0), ('trained_wins', [.5, .9, .7], 1)]:
        folder = root / case
        fit, development = folder / 'fit', folder / 'dev'
        fit.mkdir(parents=True)
        names = [f'validation_{index:03}' for index in range(98)]
        manifest = folder / 'manifest.json'
        write(manifest, {'validation_sequences': names})
        candidates = []
        for epoch, score in zip([0, 1, 20], values):
            checkpoint = fit / f'candidate_epoch{epoch:02}.pth'
            head = FutureActionHead()
            with torch.no_grad():
                next(head.parameters()).add_(epoch * .001)
            torch.save(dict(policy_family='search_actions', epoch=epoch, future_action=head.state_dict(),
                            search_action=SearchActionHead().state_dict()), checkpoint)
            candidates.append(dict(epoch=epoch, checkpoint=str(checkpoint)))
            for shard in range(2):
                write(development / checkpoint.stem / f'development_shard{shard}.json',
                      dict(completed=True, full_videos=True, sanity=False, shard=shard, shards=2,
                           epoch=epoch, checkpoint=str(checkpoint), policy_family='search_actions',
                           records=[dict(sequence=name, mean_iou=score, iou_sum=score * 10, valid_frames=10)
                                    for name in names[shard::2]]))
        write(fit / 'fit_complete.json', dict(status='SEARCH_ACTION_FIT_COMPLETE', candidates=candidates))
        with patch.object(sys, 'argv', ['select', '--fit', str(fit), '--manifest', str(manifest),
                                      '--development', str(development), '--policy', 'search_actions']):
            selection.main()
        chosen = json.loads((fit / 'selection.json').read_text())
        assert chosen['selected']['epoch'] == selected_epoch and chosen['trained_selected']['epoch'] == 1
        assert all(Path(candidate['checkpoint']).is_file() for candidate in candidates)
        assert Path(chosen['trained_best_checkpoint']).is_file()
        assert (chosen['trained_best_checkpoint'] == chosen['best_checkpoint']) == (selected_epoch == 1)
        checks.append(dict(case=case, selected_epoch=selected_epoch, trained_epoch=1, candidate_weights_preserved=True))

    for case, fail in [('full_roles_complete', False), ('failed_trained210_preserves_weights', True)]:
        folder = root / case
        parent = folder / 'parent'
        parent.mkdir(parents=True)
        (parent / 'best.pth').write_bytes(b'fixture_teacher_only')
        train_names = [f'train_{i:03}' for i in range(881)]
        val_names = [f'validation_{i:03}' for i in range(98)]
        write(parent / 'manifest.json', dict(train_sequences=train_names, validation_sequences=val_names))
        write(parent / 'complete.json', dict(status='TRAINING_COMPLETE', epochs=20, updates=3140,
              strict_reload_passed=True, best_checkpoint=str(parent / 'best.pth')))
        write(parent / 'pipeline_complete.json', dict(training_complete=True, best_checkpoint=str(parent / 'best.pth')))
        write(parent / 'lasher_report/full_report.json', dict(completed=True, dataset='lasher', sequences=245,
              frames=220703, checkpoint=str(parent / 'best.pth'), overall_metrics_percent=dict(PR=1, NPR=1, SR=1)))
        config = folder / 'config.json'
        write(config, dict(physical_gpus=[2, 3], output=str(parent), test_root='/fixture/lasher'))
        output = folder / 'output'
        executed = []
        formal_done = []
        inventories = {'lasher': dict(sequences=245, frames=220703), 'rgbt210': dict(sequences=210, frames=104706),
                       'rgbt234': dict(sequences=234, frames=116649)}

        class CommandFixture:
            def __init__(self, command, **kwargs):
                command = command[2:]
                module = command[0]
                arguments = command[1:]
                def arg(name):
                    return arguments[arguments.index(name) + 1]
                executed.append(command)
                self.code = 0
                if module == 'cadtrack_abc.sample_action_queries':
                    sampling, split, shard = Path(arg('--output')), arg('--split'), int(arg('--shard'))
                    names = train_names if split == 'train' else val_names
                    rows = [dict(sequence=name, frame=3) for name in names[shard::2]]
                    write(sampling / f'queries_shard{shard}.json', rows)
                elif module == 'cadtrack_abc.merge_action_packets':
                    split = arg('--split')
                    names = train_names if split == 'train' else val_names
                    write(Path(arg('--output')), dict(completed=True, teacher_sha256=pipeline.hashlib.sha256(b'fixture_teacher_only').hexdigest(), sequences=len(names), queries=len(names)))
                elif module == 'cadtrack_abc.train_future_actions':
                    fit = Path(arg('--output'))
                    candidates = []
                    for epoch in [0, 1]:
                        checkpoint = fit / f'candidate_epoch{epoch:02}.pth'
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint.write_bytes(b'fixture_candidate_only')
                        candidates.append(dict(epoch=epoch, checkpoint=str(checkpoint)))
                    write(fit / 'fit_complete.json', dict(status='SEARCH_ACTION_FIT_COMPLETE', epochs=20, updates=280,
                          train_sequences=881, strict_head_reload=True, teacher_sha256=pipeline.hashlib.sha256(b'fixture_teacher_only').hexdigest(), candidates=candidates))
                elif module == 'cadtrack_abc.validate_future_actions':
                    write(Path(arg('--output')) / f'development_shard{arg("--shard")}.json',
                          dict(completed=True, full_videos=True, sanity=False, sequences=49))
                elif module == 'cadtrack_abc.select_future_actions':
                    fit = Path(arg('--fit'))
                    (fit / 'best.pth').write_bytes(b'fixture_selected_only')
                    (fit / 'trained_best.pth').write_bytes(b'fixture_trained_only')
                    write(fit / 'selection.json', dict(full98_development_selection_complete=True, strict_future_head_reload=True,
                          best_checkpoint=str(fit / 'best.pth'), trained_best_checkpoint=str(fit / 'trained_best.pth'), selected={'epoch': 0}))
                elif module == 'cadtrack_abc.infer':
                    dataset, shard, checkpoint = arg('--dataset'), int(arg('--shard')), arg('--checkpoint')
                    inventory = inventories[dataset]
                    write(Path(arg('--output')) / f'complete_shard{shard}.json',
                          dict(completed=True, policy_family='search_actions', checkpoint=checkpoint, dataset=dataset,
                               sequences=inventory['sequences']//2 + (inventory['sequences']%2 if shard == 0 else 0),
                               frames=inventory['frames']//2 + (inventory['frames']%2 if shard == 0 else 0)))
                elif module == 'cadtrack_abc.score':
                    dataset, report_folder = arg('--dataset'), Path(arg('--output'))
                    checkpoint = json.loads((Path(arg('--predictions')) / 'complete_shard0.json').read_text())['checkpoint']
                    assert all((output / 'fit' / f'candidate_epoch{epoch:02}.pth').exists() for epoch in [0, 1])
                    if fail and dataset == 'rgbt210' and report_folder.parent.name == 'trained':
                        self.code = 1
                    else:
                        metrics = {'lasher': dict(PR=1, NPR=1, SR=1), 'rgbt210': dict(PR=1, SR=1), 'rgbt234': dict(MPR=1, MSR=1)}[dataset]
                        write(report_folder / 'full_report.json', dict(completed=True, dataset=dataset, checkpoint=checkpoint,
                              epoch=1 if 'trained_best' in checkpoint else 0, **inventories[dataset], overall_metrics_percent=metrics,
                              lasher_stage_goal={'all_three_reached': False}))
                        formal_done.append((report_folder.parent.name, dataset, checkpoint))
                else:
                    assert module == 'cadtrack_abc.collect_actions', module
            def wait(self):
                return self.code

        arguments = ['pipeline', '--config', str(config), '--output', str(output), '--policy', 'search_actions',
                     '--rgbt210-root', '/fixture/rgbt210', '--rgbt234-root', '/fixture/rgbt234']
        sampling_receipt = lambda sampling, names, teacher: [{'queries': len(names)//2}, {'queries': len(names)-len(names)//2}]
        collection_receipt = lambda folders, sampling, teacher: [{'completed': True, 'teacher_sha256': teacher,
                           'model_family': 'CADTrackABC-native-current-quality', 'queries': 440 if Path(sampling).name == 'queries_train' else 49},
                           {'completed': True, 'teacher_sha256': teacher, 'model_family': 'CADTrackABC-native-current-quality',
                            'queries': 441 if Path(sampling).name == 'queries_train' else 49}]
        with patch.object(sys, 'argv', arguments), patch.object(pipeline.subprocess, 'Popen', CommandFixture), \
             patch.object(pipeline, 'validate_benchmark_inventory', lambda dataset, root: inventories[dataset]), \
             patch.object(pipeline, 'validate_sampling', sampling_receipt), patch.object(pipeline, 'validate_collection', collection_receipt):
            if fail:
                from unittest import TestCase
                with TestCase().assertRaises(AssertionError):
                    pipeline.main()
            else:
                pipeline.main()
        if fail:
            assert not (output / 'pipeline_complete.json').exists()
            assert all((output / 'fit' / f'candidate_epoch{epoch:02}.pth').exists() for epoch in [0, 1])
            assert (output / 'teacher.pth').exists()
            assert json.loads((output / 'queue_state.json').read_text())['status'] == 'FAILED'
        else:
            terminal = json.loads((output / 'pipeline_complete.json').read_text())
            assert terminal['full_three_benchmarks_complete'] and terminal['trained_checkpoint_formal_metrics_complete']
            assert set(terminal['formal_reports']) == {'selected', 'trained'} and len(formal_done) == 6
            assert all(len({row[2] for row in formal_done if row[0] == role}) == 1 for role in ['selected', 'trained'])
            assert not any((output / 'fit').glob('candidate_epoch*.pth'))
            assert (output / 'fit/trained_best.pth').is_file() and not (output / 'teacher.pth').exists()
        checks.append(dict(case=case, executed_commands=len(executed), completed_fixture_reports=len(formal_done)))

actual = {dataset: pipeline.validate_benchmark_inventory(dataset, Path(root)) for dataset, root in {
          'lasher': '/data/wangwj/dataset/LasHeR/testingset', 'rgbt234': str(project / 'dataset/rgbt234')}.items()}
assert not torch.cuda.is_initialized()
result = dict(status='CPU_SOURCE_FIXTURES_AND_TWO_REAL_INVENTORIES_PASS', checks=checks,
              actual_inventories=actual, cuda_initialized=False, new_optimizer_updates=0,
              real_model_inference=False, rgbt210_real_inventory_verified=False,
              scope='Selection uses actual head serialization on CPU. Pipeline jobs/receipts are command fixtures, not neural training or benchmark results.')
(work / 'cpu_checks.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
