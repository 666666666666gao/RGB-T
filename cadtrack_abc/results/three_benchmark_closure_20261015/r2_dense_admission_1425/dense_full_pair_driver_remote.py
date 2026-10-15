from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess

project=Path('/home/gaob/CADTrack-ABC')
root=project/'outputs/r2_dense_pair_20261015'
python='/data/wangwj/envs/CADTrack/bin/python'
qualification=project/'profile_output/r2_dense_m0_capacity128_20261015'
qualified=json.loads((qualification/'state.json').read_text())
assert qualified['status']=='DENSE_M0_COMPLETE' and qualified['chosen_common_batch']==128
assert json.loads((qualification/'exit.json').read_text())['actual_controller_exit_code']==0
baseline234=project/'cadtrack_abc/results/pure_cad_rgbt234_20261015/full_report.json'
baseline210=project/'cadtrack_abc/results/three_benchmark_closure_20261015/pure_cad_rgbt210_1225/full_report.json'
for dataset,path in [('rgbt234',baseline234),('rgbt210',baseline210)]:
    reference=json.loads(path.read_text())
    assert reference['completed'] and reference['dataset']==dataset
    assert reference['checkpoint']=='/home/gaob/CADTrack_ep0014.pth.tar'
state=dict(status='DENSE_H_HA_FULL_RUNS_RUNNING',controller_pid=os.getpid(),
    started_utc=datetime.now(timezone.utc).isoformat(),jobs={},common_batch=128,
    planned_epochs=20,planned_clips_per_variant=400000,planned_updates_per_variant=3140,
    planned_query_observations_per_variant=3200000,full_ABC=False,goal_achieved=False)


def save():
    (root/'pair_state.json').write_text(json.dumps(state,indent=2))


save()
jobs=[]
for label,gpu in [('H',2),('HA',3)]:
    command=[python,'-m','cadtrack_abc.dense_pipeline','--config',str(project/'cadtrack_abc/config.json'),
        '--output',str(root/label),'--m0-root',str(qualification),'--variant',label,'--gpu',str(gpu),
        '--baseline-234',str(baseline234),'--baseline-210',str(baseline210)]
    stream=(root/f'{label}_pipeline.log').open('w')
    child=subprocess.Popen(command,cwd=project,stdout=stream,stderr=subprocess.STDOUT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
    jobs.append((label,child,stream))
    state['jobs'][label]=dict(controller_pid=child.pid,physical_gpu=gpu,command=command,status='RUNNING')
    save()
codes=[]
for label,child,stream in jobs:
    code=child.wait()
    stream.close()
    codes.append(code)
    state['jobs'][label].update(actual_exit_code=code,status='EXITED')
    save()
assert codes==[0,0],codes
histories={label:json.loads((root/label/'fit/history.json').read_text()) for label in ('H','HA')}
assert len(histories['H'])==len(histories['HA'])==20
for left,right in zip(histories['H'],histories['HA']):
    assert left['epoch']==right['epoch']
    assert left['clips']==right['clips']==20000
    assert left['clip_order_sha256']==right['clip_order_sha256']
for label in ('H','HA'):
    receipt=json.loads((root/label/'pipeline_complete.json').read_text())
    assert receipt['status']=='DENSE_FULL_TRAINING_AND_THREE_BENCHMARKS_COMPLETE'
    assert receipt['full98_selection_complete'] and set(receipt['reports'])=={'lasher','rgbt234','rgbt210'}
    state.setdefault('formal_results',{})[label]=receipt
state.update(status='DENSE_MATCHED_PAIR_FULL_TRAINING_AND_EVALUATION_COMPLETE',
    completed_utc=datetime.now(timezone.utc).isoformat(),all20_actual_clip_orders_equal=True,
    actual_exit_codes=codes,full_ABC=False,goal_achieved=False)
save()
(root/'pair_complete.json').write_text(json.dumps(state,indent=2))
print(json.dumps(state),flush=True)
