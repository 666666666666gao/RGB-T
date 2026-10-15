from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time

project=Path('/home/gaob/CADTrack-ABC')
output=project/'profile_output/r2_dense_m0_20261015_attempt2'
python='/data/wangwj/envs/CADTrack/bin/python'
config=project/'cadtrack_abc/config.json'
state=dict(status='DENSE_M0_RUNNING',controller_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
    phases={},formal_training_started=False,baseline_lasher_rerun=False)


def save():
    (output/'state.json').write_text(json.dumps(state,indent=2))


def launch(name, gpu, command):
    stream=(output/(name+'.log')).open('w')
    child=subprocess.Popen([python,*command],cwd=project,stdout=stream,stderr=subprocess.STDOUT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    state['phases'][name]=dict(pid=child.pid,physical_gpu=gpu,command=[python,*command],status='RUNNING',
        started_utc=datetime.now(timezone.utc).isoformat())
    save()
    return name,child,stream


def finish(job):
    name,child,stream=job
    code=child.wait()
    stream.close()
    state['phases'][name].update(actual_exit_code=code,status='EXITED',completed_utc=datetime.now(timezone.utc).isoformat())
    save()
    return code


save()
parity=launch('epoch0_parity',2,['-m','cadtrack_abc.qualify_dense','--config',str(config),
    '--output',str(output/'epoch0_parity.json')])
assert finish(parity)==0
parity_receipt=json.loads((output/'epoch0_parity.json').read_text())
assert parity_receipt['status']=='DENSE_EPOCH0_PARITY_PASS' and parity_receipt['actual_causal_query_frames']==24
results={}
for batch in (32,64):
    jobs=[]
    for memory,gpu,label in [(False,2,'H'),(True,3,'HA')]:
        name=f'{label}_batch{batch}'
        command=['-m','cadtrack_abc.train_dense','--config',str(config),
            '--output',str(output/name),'--batch',str(batch),'--probe']
        if memory:
            command.append('--memory')
        jobs.append(launch(name,gpu,command))
    codes=[finish(job) for job in jobs]
    assert codes==[0,0],codes
    pair={}
    orders=[]
    for label in ('H','HA'):
        folder=output/f'{label}_batch{batch}'
        terminal=json.loads((folder/'fit_complete.json').read_text())
        history=json.loads((folder/'history.json').read_text())
        progress=json.loads((folder/'progress.json').read_text())
        assert terminal['status']=='DENSE_QUALIFICATION_COMPLETE' and terminal['updates']==2
        assert terminal['raw_clips']==batch*2 and terminal['strict_dense_reload_passed']
        assert terminal['frozen_backbone_before']==terminal['frozen_backbone_after']
        assert all(terminal['changed_parameter_tensors'].values())
        assert all(value>0 for value in progress['gradients'].values())
        orders.append(history[0]['clip_order_sha256'])
        pair[label]=dict(clips_per_second=batch*2/history[0]['epoch_seconds'],
            peak_allocated_gib=progress['peak_allocated_gib'],actual_updates=2,
            gradients=progress['gradients'],changed_tensors=terminal['changed_parameter_tensors'],
            clip_order_sha256=history[0]['clip_order_sha256'],strict_reload=True)
    assert orders[0]==orders[1]
    pair['geometric_mean_clips_per_second']=(pair['H']['clips_per_second']*pair['HA']['clips_per_second'])**.5
    results[str(batch)]=pair
    state['measured_batches']=results
    save()
chosen=max((32,64),key=lambda batch:results[str(batch)]['geometric_mean_clips_per_second'])
state.update(status='DENSE_M0_COMPLETE',completed_utc=datetime.now(timezone.utc).isoformat(),
    chosen_common_batch=chosen,measured_batches=results,
    source_manifest_sha256=hashlib.sha256((project/'cadtrack_abc/results/three_benchmark_closure_20261015/r2_dense_dependency_fix_1405/source_manifest.json').read_bytes()).hexdigest(),
    parity_scope='24 causal predictions on two TRAIN clips, not a benchmark',
    training_throughput_scope='Two real updates per batch per variant; includes first data loading and optimizer initialization, not an isolated long-run efficiency result.',
    formal_training_started=False,all_three_modules_integrated=False,goal_achieved=False)
save()
print(json.dumps(state),flush=True)
