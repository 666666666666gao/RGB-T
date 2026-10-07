"""Four-GPU TRAIN query/state consistency diagnostics after closed native controls."""
import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.run_candidate_relation_training import environment, REPO, PYTHON, read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    review=read(args.review)
    assert review['status']=='PASS' and review['scope']=='CURRENT_QUERY_CONSISTENCY_SOURCE'
    assert read('/data/gb/outputs/protected_visual_reference_20261007/progress.json')['stage']=='COMPLETE_PROTECTED_REFERENCE_FULL98_AND_NATIVE'
    free=shutil.disk_usage('/data/gb').free;assert free>2*1024**3
    cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    values=[tuple(int(part.strip()) for part in line.split(',')) for line in cards.splitlines()]
    assert [row[0] for row in values]==[0,1,2,3] and all(row[1]<500 and row[2]==0 for row in values)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    sources=[f'/data/gb/outputs/current_relation_policy_collection_20261006/train/full/gpu{gpu}' for gpu in range(4)]
    for source in sources:
        config,done=read(Path(source)/'config.json'),read(Path(source)/'completion.json')
        assert config['partition']=='train' and config['batch_clips']==16 and done['completed']
    (root/'plan.json').write_text(json.dumps(dict(sources=sources,queries_per_full_shard=16,seed=42,
        sanity_queries_per_shard=1,first_original_batch_only=True,new_optimizer_updates=0,
        no_future_images=True,no_test_data=True,free_disk_bytes=free,
        scientific_scope='Numeric/state diagnostic, not a new model or proof of causal performance change'),indent=2))
    started=time.perf_counter()
    def record(stage,**fields):
        value=dict(stage=stage,at_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                   elapsed_seconds=time.perf_counter()-started,**fields)
        (root/'progress.json').write_text(json.dumps(value,indent=2))
        with (root/'events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    for stage,queries in (('sanity',1),('full',16)):
        children=[]
        for gpu,source in enumerate(sources):
            out=root/stage/f'gpu{gpu}'
            log=(root/f'{stage}_gpu{gpu}.log').open('w')
            command=[PYTHON,'-u','-m','research.probe_current_query_consistency','--source',source,
                     '--queries',str(queries),'--output',str(out)]
            child=subprocess.Popen(command,cwd=REPO,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
            children.append((child,log,gpu,out,command))
        record(stage.upper(),children=[dict(gpu=gpu,pid=child.pid,command=command) for child,_,gpu,_,command in children])
        codes=[child.wait() for child,_,_,_,_ in children]
        for _,log,_,_,_ in children:log.close()
        assert all(code==0 for code in codes),(stage,codes,'Read original traces; no unchanged retry')
        receipts=[read(out/'completion.json') for _,_,_,out,_ in children]
        assert all(row['completed'] and row['queries']==queries and row['new_optimizer_updates']==0 for row in receipts)
        record(stage.upper()+'_COMPLETE',receipts=receipts)
    record('COMPLETE_CURRENT_QUERY_CONSISTENCY',new_optimizer_updates=0,train_queries=64,
           receipts=[read(root/'full'/f'gpu{gpu}'/'completion.json') for gpu in range(4)])


if __name__=='__main__':
    main()
