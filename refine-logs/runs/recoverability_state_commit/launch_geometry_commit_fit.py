"""Fit one small causal submission head per available GPU; no backbone retrain."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/data/gb/GOLA')
FOLDER=ROOT/'refine-logs/runs/recoverability_state_commit'
BASE=Path('/data/gb/outputs/recoverability_geometry_commit_fit_20261005')
PARENT='/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'
PROBE='/data/gb/outputs/recoverability_velocity_search_commit_train88_20261005'
C1='/data/gb/outputs/c1_initial_seed42/best.pth'


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['sanity','full'],required=True)
    p.add_argument('--gpu',type=int,choices=range(4),required=True); a=p.parse_args()
    review=json.loads((FOLDER/'geometry_commit_fit_source_review.json').read_text())
    assert review['status']=='PASS' and review['runtime_attested'] is False
    assert (Path(PROBE)/'merged88_event_analysis.json').is_file()
    for part in ['sanity','gpu0','gpu1','gpu2','gpu3']:
        assert (Path(PROBE)/part/'COMPLETE').is_file()
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    device=next(line.split(', ') for line in devices.splitlines() if int(line.split(', ')[0])==a.gpu)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    assert float(device[2])>20000 and not any(line.split(', ')[0]==device[1] and 'python' in line.lower() for line in apps.splitlines())
    BASE.mkdir(exist_ok=True)
    name='sanity' if a.stage=='sanity' else 'gpu'+str(a.gpu)
    output=BASE/name; receipt=BASE/(name+'_launch.json')
    assert not output.exists() and not receipt.exists()
    if a.stage=='sanity':
        assert a.gpu==0; horizon, penalty, epochs=32,.1,4
    else:
        assert (BASE/'parity/COMPLETE').is_file()
        completed=json.loads((BASE/'sanity/completion.json').read_text())
        assert completed['complete'] and completed['optimizer_updates']==4 and completed['frozen_parent_exact']
        horizon,penalty=[(3,0.),(3,.1),(32,0.),(32,.1)][a.gpu]; epochs=60
    command=['bash',str(ROOT/'scripts/run_temporal.sh'),str(a.gpu),'train_geometry_commit',
        '--probe-root',PROBE,'--jobs',str(FOLDER/'prepared_velocity_train88_jobs.json'),
        '--parent',PARENT,'--c1-head',C1,'--output',str(output),'--horizon',str(horizon),
        '--lost-penalty',str(penalty),'--epochs',str(epochs),'--seed','42']
    if a.stage=='sanity': command+=['--sanity']
    with (BASE/(name+'.log')).open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=os.environ|{'CUDA_VISIBLE_DEVICES':str(a.gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':child.pid,'gpu':a.gpu,'stage':a.stage,'command':command,'output':str(output),
        'epochs':epochs,'horizon':horizon,'lost_penalty':penalty,
        'launched_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    receipt.write_text(json.dumps(record,indent=2)+'\n'); print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
