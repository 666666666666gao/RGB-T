"""Use the existing reviewed collector on new actual write-episode queries."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/data/gb/GOLA')
FOLDER=ROOT/'refine-logs/runs/recoverability_write_events'
OUT=Path('/data/gb/outputs/recoverability_write_events_collection_20261005')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['sanity','full'])
    p.add_argument('--gpu',type=int,choices=range(4),required=True)
    args=p.parse_args()
    assert read(FOLDER/'source_review.json')['status']=='PASS'
    plan=read(FOLDER/'prepared_write_episode_queries.json')
    cfg=read(Path(plan['base_train'])/'config.json')
    assert plan['teacher']==cfg['prefix_model'] and plan['new_episode_queries']==98
    assert plan['GT_ineligible_queries']==0 and not plan['collector_source_changed']
    if args.phase=='full':
        for gpu in range(4):
            assert read(OUT/f'sanity_gpu{gpu}_completion.json')['complete']
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    row=next(line.split(', ') for line in devices.splitlines() if int(line.split(', ')[0])==args.gpu)
    assert float(row[2])>20000 and not any(line.split(', ')[0]==row[1] and 'python' in line.lower() for line in apps.splitlines())
    free=shutil.disk_usage('/data/gb').free
    assert free>10*2**30
    suffix='sanity_jobs' if args.phase=='sanity' else 'jobs'
    jobs_path=FOLDER/f'gpu{args.gpu}_{suffix}.json'
    jobs=read(jobs_path)['jobs']
    output=OUT/f'{args.phase}_gpu{args.gpu}'
    receipt=OUT/f'{args.phase}_gpu{args.gpu}_launch.json'
    assert not output.exists() and not receipt.exists()
    OUT.mkdir(exist_ok=True)
    command=['bash','scripts/run_temporal.sh',str(args.gpu),'collect_recoverability','--partition','train',
             '--prefix-model',plan['teacher'],'--future-policy','own','--prefix-write-verification','action',
             '--jobs-file',str(jobs_path),'--output',str(output),'--clips',str(len(jobs)),
             '--batch-clips','1' if args.phase=='sanity' else '16']
    for key in ('root','cache','split','pretrained','head','motion_run','forward_batch','max_prefix','seed'):
        command+=['--'+key.replace('_','-'),str(cfg[key])]
    with (OUT/f'{args.phase}_gpu{args.gpu}.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launched={'pid':child.pid,'gpu':args.gpu,'phase':args.phase,'clips':len(jobs),'command':command,
              'created_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'disk_free_bytes':free,'collector_source_changed':False,'optimization_started':False}
    receipt.write_text(json.dumps(launched,indent=2)+'\n')
    print(json.dumps(launched),flush=True)
    if args.phase=='sanity':
        assert child.wait()==0,(OUT/f'{args.phase}_gpu{args.gpu}.log').read_text()[-3000:]
        complete=read(output/'completion.json');collected=read(output/'config.json')
        assert complete['completed'] and complete['clips']==collected['clips']==1
        assert not complete['decision_input_contains_future']
        assert collected['prefix_model']==plan['teacher'] and collected['future_policy_mode']=='own'
        assert collected['prefix_checkpoint_epoch']==collected['future_checkpoint_epoch']==4
        (OUT/f'sanity_gpu{args.gpu}_completion.json').write_text(json.dumps({'complete':True,'child_exit_code':0,
            'configuration':collected,'receipt':complete,'new_weights':0,'scope':'Real collector sanity, not training or native accuracy.'},indent=2)+'\n')
        print(json.dumps({'sanity_complete':True,'gpu':args.gpu}),flush=True)


if __name__=='__main__':main()
