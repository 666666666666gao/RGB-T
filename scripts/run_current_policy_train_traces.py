"""Collect all 881 TRAIN trajectories of the locked parent to locate real events.

This is collection, not optimization or native TEST evaluation. Reused DEV98
traces already exist. All four GPUs share disjoint frame-balanced TRAIN shards.
"""
import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime,timedelta,timezone
from pathlib import Path

import numpy as np

from scripts.run_candidate_relation_training import environment,REPO,PYTHON,read

MODEL='/data/gb/outputs/candidate_relation_native_recovery_20261006/reconstructed_epoch5/best.pth'
DATA=Path('/data/wangwj/dataset/LasHeR/traingset')
SPLIT='/data/gb/outputs/c1_initial_seed42/split.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();review=read(args.review)
    assert review['status']=='PASS' and review['scope']=='CURRENT_POLICY_FULL_TRAIN_TRACE_SOURCE'
    assert read('/data/gb/outputs/action_risk_calibration_probe_20261007/progress.json')['stage']=='COMPLETE_ACTION_RISK_CALIBRATION_PROBE'
    free=shutil.disk_usage('/data/gb').free;assert free>40*1024**3
    cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    values=[tuple(int(part.strip()) for part in line.split(',')) for line in cards.splitlines()]
    # The server's remote-desktop process uses GPU0 without an ML job. A
    # transient display-utilization sample must not block these small trackers.
    assert [row[0] for row in values]==[0,1,2,3] and all(row[1]<500 for row in values),values
    split=read(SPLIT);assert len(split['train'])==881 and len(split['validation'])==98 and not set(split['train']) & set(split['validation'])
    entries=[]
    for name in split['train']:
        seq=DATA/name
        frames=sum(path.is_file() for path in (seq/'visible').iterdir())
        assert frames==sum(path.is_file() for path in (seq/'infrared').iterdir()) and frames>1
        with (seq/'init.txt').open() as stream:initial=np.fromstring(stream.readline().strip(),sep=',')
        assert initial.shape==(4,) and np.isfinite(initial).all() and (initial[2:]>0).all(),name
        entries.append(dict(sequence=name,frames=frames))
    groups=[[] for _ in range(4)];totals=[0]*4
    for row in sorted(entries,key=lambda row:(-row['frames'],row['sequence'])):
        gpu=min(range(4),key=lambda gpu:totals[gpu]);groups[gpu].append(row);totals[gpu]+=row['frames']
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    for gpu,rows in enumerate(groups):
        (root/f'gpu{gpu}_sequences.json').write_text(json.dumps(dict(sequences=sorted(row['sequence'] for row in rows)),indent=2))
        longest=max(rows,key=lambda row:row['frames']);assert longest['frames']>=64
        (root/f'sanity_gpu{gpu}_sequences.json').write_text(json.dumps(dict(sequences=[longest['sequence']]),indent=2))
    plan=dict(model=MODEL,search_value='gross',write_verification='action',split=SPLIT,root=str(DATA),
        train_sequences=881,validation_sequences_excluded=98,frames=sum(totals),groups=groups,frames_per_GPU=totals,
        new_optimizer_updates=0,no_TEST_data=True,raw_full_TRAJECTORY_collection_not_training=True,
        free_disk_bytes=free,output_reserve_bytes=40*1024**3,gpu_preflight_memory_utilization=values,
        selection_labels_only_after_predictions=True,
        event_mining_next='Actual interventions, near-threshold alternatives, recoverable misses and failure turns; not old-policy strata')
    (root/'plan.json').write_text(json.dumps(plan,indent=2))
    def record(stage,**fields):
        value=dict(stage=stage,at_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),elapsed_seconds=time.perf_counter()-started,**fields)
        (root/'progress.json').write_text(json.dumps(value,indent=2))
        with (root/'events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    for stage,sanity in (('sanity',True),('full',False)):
        children=[]
        for gpu in range(4):
            out=root/stage/f'gpu{gpu}'/'predictions';log=(root/f'{stage}_gpu{gpu}.log').open('w')
            command=[PYTHON,'-u','-m','research.evaluate_recoverability','--dataset','lasher','--root',str(DATA),
                '--model',MODEL,'--train-split',SPLIT,'--sequence-list',str(root/(f'sanity_gpu{gpu}_sequences.json' if sanity else f'gpu{gpu}_sequences.json')),
                '--search-value','gross','--write-verification','action','--output',str(out)]
            if sanity:command+=['--limit-sequences','1','--max-frames','64']
            child=subprocess.Popen(command,cwd=REPO,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
            children.append((child,log,gpu,out))
        record(stage.upper(),children=[dict(gpu=gpu,pid=child.pid,output=str(out),expected_frames=64 if sanity else totals[gpu]) for child,_,gpu,out in children])
        codes=[child.wait() for child,_,_,_ in children]
        for _,log,_,_ in children:log.close()
        assert all(code==0 for code in codes),(stage,codes,'Read original traces; no unchanged retry')
        receipts=[read(out/'inference_completion.json') for _,_,_,out in children]
        for gpu,row in enumerate(receipts):
            assert row['completed'] and row['frames']==(64 if sanity else totals[gpu]) and row['sequences']==(1 if sanity else len(groups[gpu]))
            assert not row['gt_scoring_completed']
        record(stage.upper()+'_COMPLETE',receipts=receipts)
    record('COMPLETE_CURRENT_POLICY_FULL_TRAIN_TRACES',train_sequences=881,frames=sum(totals),
        new_optimizer_updates=0,official_TEST_accuracy_new=False,raw_trajectories_available_for_actual_event_sampling=True)


if __name__=='__main__':main()
