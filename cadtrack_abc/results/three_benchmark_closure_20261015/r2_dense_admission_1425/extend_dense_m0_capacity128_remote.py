from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess

project=Path('/home/gaob/CADTrack-ABC')
base=project/'profile_output/r2_dense_m0_20261015_attempt2'
output=project/'profile_output/r2_dense_m0_capacity128_20261015'
python='/data/wangwj/envs/CADTrack/bin/python'
original=json.loads((base/'state.json').read_text())
assert original['status']=='DENSE_M0_COMPLETE'
assert json.loads((base/'exit.json').read_text())['actual_controller_exit_code']==0
state=dict(original,status='DENSE_CAPACITY128_RUNNING',controller_pid=os.getpid(),
    started_utc=datetime.now(timezone.utc).isoformat(),base_qualification=str(base),
    base_state_sha256=hashlib.sha256((base/'state.json').read_bytes()).hexdigest(),
    phases={},original_phases=original['phases'],original_chosen_common_batch=original['chosen_common_batch'],
    rationale='Measured HA batch64 peak6.8712GiB on24GiB; one final128 test before formal training.')


def save():
    (output/'state.json').write_text(json.dumps(state,indent=2))


save()
jobs=[]
for label,gpu,memory in [('H',2,False),('HA',3,True)]:
    command=[python,'-m','cadtrack_abc.train_dense','--config',str(project/'cadtrack_abc/config.json'),
        '--output',str(output/f'{label}_batch128'),'--batch','128','--probe']
    if memory:
        command.append('--memory')
    stream=(output/f'{label}_batch128.log').open('w')
    child=subprocess.Popen(command,cwd=project,stdout=stream,stderr=subprocess.STDOUT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    state['phases'][label]=dict(command=command,pid=child.pid,physical_gpu=gpu,status='RUNNING')
    jobs.append((label,child,stream))
    save()
codes=[]
for label,child,stream in jobs:
    code=child.wait()
    stream.close()
    state['phases'][label].update(actual_exit_code=code,status='EXITED')
    codes.append(code)
    save()
assert codes==[0,0],codes
pair={}
orders=[]
for label in ('H','HA'):
    folder=output/f'{label}_batch128'
    terminal=json.loads((folder/'fit_complete.json').read_text())
    history=json.loads((folder/'history.json').read_text())
    progress=json.loads((folder/'progress.json').read_text())
    assert terminal['status']=='DENSE_QUALIFICATION_COMPLETE' and terminal['updates']==2 and terminal['raw_clips']==256
    assert terminal['strict_dense_reload_passed'] and all(terminal['changed_parameter_tensors'].values())
    assert terminal['frozen_backbone_before']==terminal['frozen_backbone_after']
    assert all(value>0 for value in progress['gradients'].values())
    orders.append(history[0]['clip_order_sha256'])
    pair[label]=dict(clips_per_second=256/history[0]['epoch_seconds'],peak_allocated_gib=progress['peak_allocated_gib'],
        actual_updates=2,gradients=progress['gradients'],changed_tensors=terminal['changed_parameter_tensors'],
        clip_order_sha256=history[0]['clip_order_sha256'],strict_reload=True)
assert orders[0]==orders[1]
pair['geometric_mean_clips_per_second']=(pair['H']['clips_per_second']*pair['HA']['clips_per_second'])**.5
state['measured_batches']['128']=pair
chosen=max((32,64,128),key=lambda batch:state['measured_batches'][str(batch)]['geometric_mean_clips_per_second'])
state.update(status='DENSE_M0_COMPLETE',chosen_common_batch=chosen,
    completed_utc=datetime.now(timezone.utc).isoformat(),formal_training_started=False,
    inherited_parity='Exact24 TRAIN frames from immutable successful base qualification; not re-run for128.',
    real_qualification_updates_total=12,goal_achieved=False)
save()
print(json.dumps(state),flush=True)
