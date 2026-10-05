"""Evaluate one locked commit checkpoint on all98 reused developer videos."""
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


def main():
    p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,choices=range(4),required=True)
    a=p.parse_args()
    review=json.loads((FOLDER/'geometry_commit_full98_source_review.json').read_text())
    assert review['status']=='PASS' and review['runtime_attested'] is False
    parity=json.loads((BASE/'parity/completion.json').read_text())
    assert parity['complete'] and parity['source_runtime_parity']
    fit=BASE/('gpu'+str(a.gpu)); completed=json.loads((fit/'completion.json').read_text())
    assert (fit/'COMPLETE').is_file() and completed['complete'] and not completed['sanity_only']
    assert completed['epochs']==completed['optimizer_updates']==60 and completed['frozen_parent_exact']
    cfg=json.loads((fit/'config.json').read_text())
    assert cfg['parent']==PARENT and cfg['seed']==42
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    device=next(line.split(', ') for line in devices.splitlines() if int(line.split(', ')[0])==a.gpu)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    assert float(device[2])>20000 and not any(line.split(', ')[0]==device[1] and 'python' in line.lower() for line in apps.splitlines())
    receipt=fit/'full98_launch.json'; output=fit/'continuous/predictions'
    assert not receipt.exists() and not output.exists()
    command=['bash',str(ROOT/'scripts/run_temporal.sh'),str(a.gpu),'evaluate_recoverability',
        '--dataset','lasher','--root','/data/wangwj/dataset/LasHeR/traingset',
        '--model',PARENT,'--commit-model',str(fit/'best.pth'),
        '--pretrained','/data/gb/GOLA/pretrained_models/gola_b224.bin',
        '--c1-head','/data/gb/outputs/c1_initial_seed42/best.pth',
        '--motion-run','/data/gb/outputs/abc_joint_v1_seed42',
        '--validation-split','/data/gb/outputs/c1_initial_seed42/split.json',
        '--max-frames','0','--write-verification','action','--seed','42','--output',str(output)]
    with (fit/'full98.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=os.environ|{'CUDA_VISIBLE_DEVICES':str(a.gpu)},
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':child.pid,'gpu':a.gpu,'stage':'COMPLETE98_CONTINUOUS_DEVELOPER_VIDEOS',
        'command':command,'output':str(output),'parent':PARENT,'commit_model':str(fit/'best.pth'),
        'head_epoch':completed['selected_epoch'],'all_fit_updates':60,
        'native_test_data_read':False,'validation_scope':'98 repeatedly used developer videos, not unseen confirmation',
        'launched_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    receipt.write_text(json.dumps(record,indent=2)+'\n'); print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
