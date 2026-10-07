"""Four full C calibration fits on matched policy states; no new ABC deployment."""
import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.run_candidate_relation_training import environment, PYTHON, REPO, read

ARMS=[('plain_lr3',1e-3,False),('balanced_lr3',1e-3,True),('plain_lr4',1e-4,False),('balanced_lr4',1e-4,True)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();review=read(args.review)
    assert review['status']=='PASS' and review['scope']=='ACTION_RISK_CALIBRATION_PROBE_SOURCE'
    assert read('/data/gb/outputs/current_query_consistency_20261007/progress.json')['stage']=='COMPLETE_CURRENT_QUERY_CONSISTENCY'
    free=shutil.disk_usage('/data/gb').free;assert free>2*1024**3
    cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    values=[tuple(int(part.strip()) for part in line.split(',')) for line in cards.splitlines()]
    assert [row[0] for row in values]==[0,1,2,3] and all(row[1]<500 and row[2]==0 for row in values)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    (root/'plan.json').write_text(json.dumps(dict(arms=ARMS,epochs=128,batch_size=64,
        optimizer_steps_per_full_arm=512,TRAIN_queries=256,DEV_queries=196,committee_heads_per_model=3,
        bootstrap_unit='TRAIN sequence, not three separately accepted seeds',parent_ABC_frozen=True,
        risk_probe_not_online_tracking=True,free_disk_bytes=free,seed=42),indent=2))
    def record(stage,**fields):
        value=dict(stage=stage,at_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),elapsed_seconds=time.perf_counter()-started,**fields)
        (root/'progress.json').write_text(json.dumps(value,indent=2))
        with (root/'events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    for stage,epochs in (('sanity',2),('full',128)):
        children=[]
        for gpu,(name,lr,balanced) in enumerate(ARMS):
            out=root/stage/name;log=(root/f'{stage}_{name}.log').open('w')
            command=[PYTHON,'-u','-m','research.probe_action_risk_calibration','--output',str(out),'--lr',str(lr),'--epochs',str(epochs),'--batch-size','64']
            if balanced:command.append('--balanced')
            child=subprocess.Popen(command,cwd=REPO,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
            children.append((child,log,gpu,name,out))
        record(stage.upper(),children=[dict(gpu=gpu,pid=child.pid,arm=name,output=str(out)) for child,_,gpu,name,out in children])
        codes=[child.wait() for child,_,_,_,_ in children]
        for _,log,_,_,_ in children:log.close()
        assert all(code==0 for code in codes),(stage,codes,'Read original traces; no unchanged retry')
        receipts={name:read(out/'completion.json') for _,_,_,name,out in children}
        assert all(row['completed'] and row['optimizer_steps']==epochs*4 and row['parent_parameters_exact'] and row['strict_best_last_reload_equal'] for row in receipts.values())
        record(stage.upper()+'_COMPLETE',receipts=receipts)
    record('COMPLETE_ACTION_RISK_CALIBRATION_PROBE',receipts=receipts,
           new_C_calibrator_full_optimizer_steps=2048,new_main_ABC_optimizer_steps=0,
           native_tracking_metrics_new=False,equal_count_cache_comparison_is_not_online_policy=True)


if __name__=='__main__':main()
