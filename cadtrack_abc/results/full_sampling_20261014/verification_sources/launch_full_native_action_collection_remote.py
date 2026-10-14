from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys

project=Path('/home/gaob/CADTrack-ABC')
sys.path.insert(0,str(project))
from cadtrack_abc.future_pipeline import validate_parent,validate_sampling
parent=project/'outputs/cad_abc_20261013'
sampling=project/'outputs/native_future_sampling_20261014'
teacher,parent_report=validate_parent(parent)
teacher_sha=hashlib.sha256(teacher.read_bytes()).hexdigest()
verified=json.loads((sampling/'sampling_complete_verified.json').read_text())
assert verified['full_sampling_complete'] and verified['all_original_drivers_exited']
assert verified['actual_total_frames']==514081 and verified['teacher_sha256']==teacher_sha
manifest=parent/'manifest.json'
inventory=json.loads(manifest.read_text())
receipts={}
for split in ['train','validation']:
    expected=set(inventory['train_sequences' if split=='train' else 'validation_sequences'])
    receipts[split]=validate_sampling(sampling/('queries_'+split),expected,teacher_sha)
    assert sum(row['queries'] for row in receipts[split])==verified['splits'][split]['queries']
old_launch=json.loads((sampling/'launch.json').read_text())
assert all(not (Path('/proc')/str(row['pid'])).exists() for row in old_launch['jobs'])
memory=subprocess.run(['nvidia-smi','-i','2,3','--query-gpu=index,memory.used',
                       '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
assert all(int(row.split(',')[1])<500 for row in memory.splitlines()),memory
output=project/'outputs/native_action_collection_20261014'
output.mkdir(exist_ok=False)
fixed_teacher=output/'teacher.pth'
shutil.copyfile(teacher,fixed_teacher)
assert fixed_teacher.read_bytes()==teacher.read_bytes()
blank_annotation_files=[]
for name in inventory['train_sequences']+inventory['validation_sequences']:
    lines=(Path(inventory['train_root'])/name/'init.txt').read_text().splitlines()
    blank=sum(not line.strip() for line in lines)
    if blank:
        blank_annotation_files.append(dict(sequence=name,blank_lines=blank))
assert sum(row['blank_lines'] for row in blank_annotation_files)==2
python='/data/wangwj/envs/CADTrack/bin/python'
jobs=[]
for shard,gpu in enumerate([2,3]):
    commands=[]
    for split in ['train','validation']:
        commands.append([python,'-m','cadtrack_abc.collect_actions','--config',str(project/'cadtrack_abc/config.json'),
                         '--teacher',str(fixed_teacher),'--teacher-policy','native','--manifest',str(manifest),
                         '--queries',str(sampling/('queries_'+split)/f'queries_shard{shard}.json'),
                         '--split',split,'--output',str(output/('collection_'+split)/f'shard{shard}'),
                         '--device','cuda','--horizon','3','--action-batch','64'])
    driver=output/f'driver_shard{shard}.py'
    driver.write_text('import subprocess\n'+'\n'.join(f'subprocess.run({command!r},cwd={str(project)!r},check=True)' for command in commands)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    log=output/f'collection_shard{shard}.log'
    with log.open('w') as stream:
        process=subprocess.Popen([python,str(driver)],cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append(dict(pid=process.pid,physical_gpu=gpu,shard=shard,commands=commands,log=str(log)))
receipt=dict(status='FULL_NATIVE_H3_ACTION_COLLECTION_LAUNCHED',started_utc=datetime.now(timezone.utc).isoformat(),
             jobs=jobs,physical_gpus=[2,3],teacher=str(fixed_teacher),teacher_sha256=teacher_sha,
             train_sequences=881,development_sequences=98,train_queries=5092,development_queries=605,
             total_queries=5697,actions_per_query=210,total_action_labels=1196370,horizon=3,action_batch=64,
             optimizer_updates=0,fit_automatically_started=False,baseline_rerun=False,official_test_used=False,
             blank_annotation_files=blank_annotation_files,actual_sampled_frames=514081,
             launch_gpu_memory_only=memory,home_free_bytes=shutil.disk_usage(project).free,
             scope='Same-state native-teacher H3 labels; all four local-or-one-region contexts and six submit modes. No fitting or formal score follows automatically.')
(output/'launch.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
