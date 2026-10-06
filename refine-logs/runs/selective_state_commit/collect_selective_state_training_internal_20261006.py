"""Copy already-closed collection, training and full98 reports; no NN replay."""
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

base = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/selective_state_commit')
target = base / 'training_internal_complete'
assert not target.exists()
payload = '''import json,tarfile
from pathlib import Path
root=Path('/data/gb/outputs/selective_state_training_endpoint_20261006')
collection=Path('/data/gb/outputs/selective_state_collection_20261006')
labels=json.loads((collection/'collection_complete.json').read_text())
assert labels['complete'] and labels['queries']==1789
arms=('H3_lost0','H3_lost01','H32_lost0','H32_lost01')
selected=json.loads((root/'selected_model.json').read_text())
report=json.loads((root/'full98_report/full_recoverability_report.json').read_text())
assert report['completed'] and report['sequences']==98
fits=[]
for arm in arms:
 folder=root/'fit_full'/arm
 done=json.loads((folder/'training_complete.json').read_text())
 config=json.loads((folder/'config.json').read_text())
 history=json.loads((folder/'history.json').read_text())
 assert done['status']=='COMPLETE_FULL_C_EXTENSION_TRAINING'
 assert done['epochs']==60 and done['optimizer_updates']==840==config['required_optimizer_updates']
 assert [r['epoch'] for r in history]==list(range(61))
 assert history[-1]['optimizer_updates']==840
 assert done['strict_reload_pass'] and done['last_strict_reload_pass'] and done['M0_parent_exact']
 assert done['actual_gradient_parameters'] and done['actual_changed_parameters']
 fits.append({'arm':arm,'training':done,'config':config})
 for checkpoint in ('best','last'):
  label=arm+'_'+checkpoint
  receipt=json.loads((root/'full98'/label/'predictions/inference_completion.json').read_text())
  assert receipt['completed'] and (receipt['sequences'],receipt['frames'])==(98,49418)
  assert label in report['variants']
candidates=[arm+'_'+checkpoint for checkpoint in ('best','last') for arm in arms]
assert selected['selected_candidate']==max(candidates,key=lambda k:report['variants'][k]['sequence_mean_iou'])
assert selected['same_checkpoint_both_native_datasets']
files=[(collection/'collection_complete.json','collection/collection_complete.json'),
       (root/'selected_model.json','training/selected_model.json')]
for stage in ('sanity','full'):
 for gpu in range(4):
  folder=collection/stage/f'gpu{gpu}'
  for name in ('config.json','summary.json','COMPLETE'):
   p=folder/name
   if p.is_file(): files.append((p,'collection/'+str(p.relative_to(collection))))
for stage in ('fit_sanity','fit_full','full98','full98_report'):
 for p in sorted((root/stage).rglob('*')):
  if p.is_file() and (p.suffix in ('.json','.csv','.png','.pdf') or p.name=='COMPLETE'):
   files.append((p,'training/'+str(p.relative_to(root))))
archive=Path('/data/gb/setup/selective_state_training_internal_text_20261006.tar')
assert not archive.exists()
with tarfile.open(archive,'w') as stream:
 for path,name in files: stream.add(path,arcname=name,recursive=False)
print(json.dumps({'status':'CLOSED_TRAINING_AND_FULL98_INTAKE','archive':str(archive),
 'archive_bytes':archive.stat().st_size,'files':len(files),'training':fits,'selection':selected,
 'NN_replays':0,'active_progress_process_GPU_queries':0}))
'''
compile(payload, 'closed_training_full98_intake', 'exec')
run = subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'], input=payload,
                     text=True, encoding='utf-8', capture_output=True)
(base/'training_internal_intake_stdout.txt').write_text(run.stdout,encoding='utf-8')
(base/'training_internal_intake_stderr.txt').write_text(run.stderr,encoding='utf-8')
run.check_returncode()
receipt=json.loads(run.stdout)
assert shutil.disk_usage(base).free > receipt['archive_bytes']*2 + 256*1024**2
archive=base/'training_internal_text_archive.tar'
assert not archive.exists()
subprocess.run(['scp','2027:'+receipt['archive'],str(archive)],check=True)
assert archive.stat().st_size==receipt['archive_bytes']
target.mkdir()
with tarfile.open(archive) as stream:
 assert len(stream.getmembers())==receipt['files']
 assert all(n.startswith(('training/','collection/')) and '..' not in Path(n).parts for n in stream.getnames())
 stream.extractall(target)
(base/'actual_training_internal_intake.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'files':receipt['files'],'archive_bytes':receipt['archive_bytes'],
                  'selected_epoch':receipt['selection']['selected_epoch'],'NN_replays':0}),flush=True)
