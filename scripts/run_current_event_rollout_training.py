"""Dependency queue: exact current events, all ABC fits, full98, one native model.

Reuse the project subprocess-wave lifecycle. Dependencies require successful
closed receipts; no OOM retry, environment rebuild or foreign-process cleanup.
"""
import argparse
import json
import math
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from research.mine_current_policy_events import mine_sequence
from research.collect_candidate_metrics import ground_truth


REPO = Path(__file__).resolve().parents[1]
PYTHON = '/data/gb/envs/gola/bin/python'
TRACE = Path('/data/gb/outputs/current_policy_train_traces_20261007')
PARENT = '/data/gb/outputs/candidate_relation_native_recovery_20261006/reconstructed_epoch5/best.pth'
DEV_MODEL = '/data/gb/outputs/post_search_relation_control_20261007/fit_full/one_way_lr5/best.pth'
DEV = Path('/data/gb/outputs/protected_visual_reference_20261007/full98/parent/predictions')
SPLIT = '/data/gb/outputs/c1_initial_seed42/split.json'
DATA = '/data/wangwj/dataset/LasHeR/traingset'
ARMS = [('reference_lr5', 'reference', 1e-5), ('reference_lr3e6', 'reference', 3e-6),
        ('budgeted_lr5', 'budgeted', 1e-5), ('budgeted_lr3e6', 'budgeted', 3e-6)]


def read(path):
    return json.loads(Path(path).read_text())


def environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                LD_LIBRARY_PATH='/data/gb/envs/gola/lib', PYTHONPATH=str(REPO),
                TORCH_HOME='/data/gb/cache/torch', XDG_CACHE_HOME='/data/gb/cache',
                TMPDIR='/data/gb/cache/tmp', OMP_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')


def groups(jobs):
    videos = {}
    for job in jobs:
        videos.setdefault(job['sequence'], []).append(job)
    result, costs = [[] for _ in range(4)], [0]*4
    for name, rows in sorted(videos.items(), key=lambda item:max(row['query_frame'] for row in item[1])+240*len(item[1]), reverse=True):
        gpu = min(range(4), key=costs.__getitem__)
        result[gpu].extend(rows)
        costs[gpu] += max(row['query_frame'] for row in rows)+240*len(rows)
    assert all(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--predecessor-pid', type=int, default=2310613)
    args = p.parse_args()
    review = read(args.review)
    assert review['status'] == 'PASS' and review['scope'] == 'COMPLETE_CURRENT_EVENT_ROLLOUT_PIPELINE_SOURCE'
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def record(stage, **fields):
        value = dict(stage=stage, at_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                     elapsed_seconds=time.perf_counter()-started, **fields)
        (root/'queue_state.json').write_text(json.dumps(value, indent=2))
        (root/'progress.json').write_text(json.dumps(value, indent=2))
        with (root/'events.jsonl').open('a') as stream: stream.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)

    def cpu(module, *arguments):
        with (root/(module.rsplit('.', 1)[-1]+'.log')).open('a') as log:
            subprocess.run([PYTHON, '-u', '-m', module, *map(str, arguments)], cwd=REPO,
                           env=environment(''), stdout=log, stderr=subprocess.STDOUT, check=True)

    def wave(stage, commands):
        assert len(commands) <= 4
        while True:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, check=True)
            used = [int(value) for value in result.stdout.splitlines()]
            assert len(used) == 4
            if all(value < 500 for value in used): break
            record(stage.upper()+'_WAIT_GPU_RELEASE', used_mib=used, no_foreign_process_killed=True)
            time.sleep(240)
        children = []
        for gpu, command in enumerate(commands):
            log = (root/f'{stage}_gpu{gpu}.log').open('w')
            child = subprocess.Popen(command, cwd=REPO, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log, gpu))
        record(stage.upper(), children=[dict(gpu=gpu, pid=child.pid, command=commands[gpu]) for child,_,gpu in children])
        exits = []
        for child, log, gpu in children:
            code = child.wait(); log.close(); exits.append((gpu,code))
        assert all(code == 0 for _,code in exits), (stage, exits, 'Primary logs retained; no unchanged retry')
        record(stage.upper()+'_ALL_PROCESSES_CLOSED_PASS', exits=exits)

    split = read(SPLIT)
    assert len(split['train']) == 881 and len(split['validation']) == 98 and not set(split['train']) & set(split['validation'])
    done = read(DEV/'inference_completion.json')
    assert done['completed'] and done['sequences'] == 98 and done['frames'] == 49418
    assert {row['sequence'] for row in done['records']} == set(split['validation'])
    dev_queries, dev_counts, dev_failures = [], Counter(), []
    for row in done['records']:
        queries, counts, failures = mine_sequence(DEV, row, DATA)
        dev_queries.extend(queries); dev_counts.update(counts)
        dev_failures.extend(dict(sequence=row['sequence'], **event) for event in failures)
    dev_jobs = [dict(sequence=row['sequence'], query_frame=row['query_frame']) for row in dev_queries if row['H3_collector_eligible']]
    (root/'DEV_events.json').write_text(json.dumps(dict(jobs=dev_jobs, queries=dev_queries, counts=dict(dev_counts),
        failures=dev_failures, repeated_developer_partition=True, future_GT_for_labels_only=True), indent=2))
    record('WAIT_ORIGINAL_FULL_TRAIN_OWNER', predecessor_pid=args.predecessor_pid, DEV_event_jobs=len(dev_jobs),
           scope='CPU mining and queued commands; new NN/optimizer0, original four GPU jobs untouched')
    while read(TRACE/'progress.json')['stage'] != 'COMPLETE_CURRENT_POLICY_FULL_TRAIN_TRACES':
        assert Path('/proc',str(args.predecessor_pid)).exists(), 'Predecessor ended without successful completion; inspect its primary log'
        time.sleep(240)
    complete = read(TRACE/'progress.json')
    assert complete['train_sequences'] == 881 and complete['frames'] == 464663
    cpu('research.mine_current_policy_events', '--trace-root', TRACE, '--output', root/'TRAIN_events')
    inventory = read(root/'TRAIN_events/report.json')
    assert inventory['full_TRAIN_inventory'] and inventory['closed_sequences'] == 881 and inventory['closed_frames'] == 464663
    train_jobs = read(root/'TRAIN_events/jobs.json')['jobs']
    assert {row['sequence'] for row in train_jobs} == set(split['train']), 'Report omitted videos before claiming all881 training coverage'
    assert {row['sequence'] for row in dev_jobs} == set(split['validation'])
    shards = dict(train=groups(train_jobs), validation=groups(dev_jobs))
    (root/'plan.json').write_text(json.dumps(dict(TRAIN_jobs=len(train_jobs), DEV_jobs=len(dev_jobs),
        shard_jobs=shards, ARMS=ARMS, epochs=24, batch_size=32, seed=42, threshold=.03,
        same_architecture='ABC_candidate_relations', initialization=PARENT, storage_schema='causal_sequence_prefix_v1',
        order=['four_real_prefix_sanity','long32_full_backward','four_ABC_sanity_fits','all_event_collection',
               'full_collection_capacity','four_complete24epoch_fits','eight_full98','one_model_both_native'],
        queue_helper='Existing project subprocess waves; successful closed receipts before dependencies, no screen/output-early termination',
        untouched_confirmation=False), indent=2))

    def collect_command(partition, rows, output):
        jobs = output.parent/(output.name+'_jobs.json')
        jobs.parent.mkdir(parents=True, exist_ok=True)
        jobs.write_text(json.dumps(dict(jobs=rows)))
        return [PYTHON, '-u', '-m', 'research.collect_event_rollouts', '--partition', partition,
                '--jobs-file', str(jobs), '--reference-traces', str(TRACE/'full' if partition == 'train' else DEV),
                '--reference-model', PARENT if partition == 'train' else DEV_MODEL, '--prefix-model', PARENT, '--output', str(output)]

    # Early plus late query from each real video checks shared storage and that
    # private futures leave the main prefix unchanged. One TRAIN video carries
    # the global longest actual query for the capacity witness.
    selected = []
    for partition in ('train', 'validation'):
        videos = {}
        for job in train_jobs if partition == 'train' else dev_jobs:
            videos.setdefault(job['sequence'], []).append(job)
        ordered = sorted(videos, key=lambda name:max(row['query_frame'] for row in videos[name]), reverse=True)
        early = min(videos, key=lambda name:min(row['query_frame'] for row in videos[name]))
        short_write_query = None
        if partition == 'train':
            # q=1 has no actual memory update. Find a real early post-write
            # state so the sequential-vs-parallel full gradient check matters.
            for item in read(root/'TRAIN_events/closed_inventory.json'):
                name = item['sequence']
                if name==ordered[0]:continue
                folder = TRACE/'full'/f"gpu{item['gpu']}"/'predictions'
                gt = ground_truth(DATA,name,'lasher')
                with np.load(folder/(name+'_recoverability_decisions.npz')) as archive:
                    for index in np.flatnonzero(archive['template_updated'][:30]):
                        q = int(index)+2
                        if q+3<len(gt) and np.isfinite(gt[q:q+4]).all() and (gt[q:q+4,2:]>0).all():
                            early,short_write_query = name,dict(sequence=name,query_frame=q)
                            break
                if short_write_query is not None:break
            assert short_write_query is not None
        assert ordered[0] != early
        for name in (ordered[0], early):
            rows = sorted(videos[name], key=lambda row:row['query_frame'])
            chosen = [rows[i] for i in sorted({0,len(rows)-1})]
            if name==early and short_write_query is not None:
                chosen=sorted({row['query_frame']:row for row in [*chosen,short_write_query]}.values(),key=lambda row:row['query_frame'])
            selected.append((partition,chosen))
    sanity_dirs = [root/'collect_sanity'/f'gpu{gpu}' for gpu in range(4)]
    wave('collect_sanity', [collect_command(partition, rows, sanity_dirs[gpu]) for gpu,(partition,rows) in enumerate(selected)])
    sanity_train, sanity_val = list(map(str,sanity_dirs[:2])), list(map(str,sanity_dirs[2:]))
    cpu('scripts.validate_event_collection', '--roots', *sanity_dirs)
    wave('capacity_sanity', [[PYTHON,'-u','-m','research.check_event_training_sanity','--train',*sanity_train,
        '--validation',*sanity_val,'--model',PARENT,'--batch-size','32','--output',str(root/'capacity_sanity.json')]])

    def fit(stage, epochs, train, validation):
        commands=[]
        for arm, ranking, lr in ARMS:
            commands.append([PYTHON,'-u','-m','research.train_event_recoverability','--train',*train,
                '--validation',*validation,'--init-checkpoint',PARENT,'--epochs',str(epochs),'--batch-size','32',
                '--lr',str(lr),'--seed','42','--action-ranking',ranking,'--search-supervision','oracle',
                '--output',str(root/stage/arm)])
        wave(stage, commands)
        count=sum(read(Path(folder)/'completion.json')['clips'] for folder in train)
        for arm,_,_ in ARMS:
            folder=root/stage/arm
            receipt,config,history=read(folder/'completion.json'),read(folder/'config.json'),read(folder/'metrics.json')
            assert receipt['completed'] and receipt['epochs']==epochs and receipt['optimizer_steps']==epochs*math.ceil(count/32)
            assert receipt['modules_changed']=={'A':True,'B':True,'C':True} and all(value>0 for value in receipt['max_module_gradient_norms'].values())
            assert receipt['frozen_C1_weights_exact'] and not receipt['GOLA_and_motion_new_gradients']
            assert all(value['strict_reload'] for value in receipt['checkpoints'].values())
            assert [row['epoch'] for row in history]==list(range(epochs+1))
            assert config['train_clips']==count
        record(stage.upper()+'_ALL_ABC_GRADIENTS_UPDATES_RELOAD_PASS', updates_per_arm=epochs*math.ceil(count/32))

    fit('fit_sanity', 2, sanity_train, sanity_val)
    for partition in ('train','validation'):
        wave('collect_full_'+partition, [collect_command(partition, rows, root/'collection'/partition/f'gpu{gpu}')
                                        for gpu,rows in enumerate(shards[partition])])
    train=[str(root/'collection/train'/f'gpu{gpu}') for gpu in range(4)]
    validation=[str(root/'collection/validation'/f'gpu{gpu}') for gpu in range(4)]
    cpu('scripts.validate_event_collection','--roots',*train,*validation)
    fit('fit_full', 24, train, validation)
    candidates=[]
    for checkpoint in ('best','last'):
        commands=[]
        for arm,_,_ in ARMS:
            label=arm+'_'+checkpoint
            model=root/'fit_full'/arm/(checkpoint+'.pth')
            candidates.append((label,model))
            commands.append([PYTHON,'-u','-m','research.evaluate_recoverability','--dataset','lasher','--root',DATA,
                '--validation-split',SPLIT,'--model',str(model),'--search-value','gross','--write-verification','action',
                '--output',str(root/'full98'/label/'predictions')])
        wave('full98_'+checkpoint,commands)
    for label,_ in candidates:
        receipt=read(root/'full98'/label/'predictions/inference_completion.json')
        assert receipt['completed'] and receipt['sequences']==98 and receipt['frames']==49418
    references=['/data/gb/outputs/abc_internal_validation_v1/baseline/predictions',
                '/data/gb/outputs/abc_internal_validation_v1/c1/predictions',
                '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/predictions',
                '/data/gb/outputs/recoverability_search_gross_control_20261006/predictions',str(DEV)]
    cpu('research.collect_recoverability_metrics','--dataset','lasher','--root',DATA,'--split',SPLIT,
        '--labels',*[label for label,_ in candidates],'--runs',*[str(root/'full98'/label/'predictions') for label,_ in candidates],
        '--reference-labels','baseline','c1','old4','gross_parent','current_parent','--references',*references,
        '--output',root/'full98_report')
    report=read(root/'full98_report/full_recoverability_report.json')
    assert report['completed'] and report['sequences']==98
    # Keep the actual parent if all new trained checkpoints are worse. Its score
    # is a reference, never credited as a new training gain.
    label,model=max([('current_parent',Path(PARENT)),*candidates], key=lambda row:report['variants'][row[0]]['sequence_mean_iou'])
    selection=dict(selected_candidate=label,parent_model=str(model),state_commit_model=None,
        same_checkpoint_both_native_datasets=True,native_metrics_completed=False,search_value='gross',
        improves_current_parent=report['variants'][label]['sequence_mean_iou']>report['variants']['current_parent']['sequence_mean_iou'],
        same_parent_chosen_not_new_gain=label=='current_parent',
        sequence_mean_iou={key:value['sequence_mean_iou'] for key,value in report['variants'].items()},
        selection_scope='Eight actual full98 trained checkpoints plus protected parent, locked before same two native datasets; repeated DEV, not3seed/untouched confirmation')
    (root/'selected_model.json').write_text(json.dumps(selection,indent=2))
    record('ONE_MODEL_LOCKED_BEFORE_BOTH_NATIVE',**selection)
    cpu('scripts.run_selective_state_native','--selection',root/'selected_model.json','--review',args.review,'--output',root/'native')
    native=read(root/'native/complete_metrics.json')
    assert native['completed'] and native['same_ABC_model_both_datasets']==str(model)
    removed=[]
    for stage in ('fit_sanity','fit_full'):
        for arm,_,_ in ARMS:
            for filename in ('best.pth','last.pth','initial.pth'):
                path=root/stage/arm/filename
                if path==model:continue
                assert path.resolve().is_relative_to(root.resolve())
                removed.append(dict(path=str(path),bytes=path.stat().st_size))
                path.unlink()
    (root/'unused_own_weight_cleanup.json').write_text(json.dumps(dict(removed=removed,kept=str(model),protected_dependencies_preserved=True),indent=2))
    record('COMPLETE_CURRENT_EVENT_TRAINING_FULL98_BOTH_NATIVE',selection=selection,native=native,
           full_epochs_per_arm=24,unused_own_weights_removed=len(removed))


if __name__ == '__main__':main()
