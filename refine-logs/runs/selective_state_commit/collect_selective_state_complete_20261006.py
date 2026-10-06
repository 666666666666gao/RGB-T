"""One intake of actual completed full-training/native artifacts; never rerun NN."""
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

base = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/selective_state_commit')
target = base / 'complete'
assert not target.exists()
payload = '''import json,tarfile
from pathlib import Path
root=Path('/data/gb/outputs/selective_state_training_endpoint_20261006')
collection=Path('/data/gb/outputs/selective_state_collection_20261006')
launch=json.loads(Path('/data/gb/setup/selective_state_training_endpoint_launch_20261006.json').read_text())
progress=json.loads((root/'progress.json').read_text())
assert progress['stage']=='COMPLETE_FULL_TRAINING_FULL98_BOTH_NATIVE_FIVE_METRICS'
proc=Path('/proc')/str(launch['pid'])/'stat'
assert not proc.exists() or proc.read_text().rsplit(')',1)[1].split()[0]=='Z'
labels=json.loads((collection/'collection_complete.json').read_text())
assert labels['complete'] and labels['queries']==1789
arms=('H3_lost0','H3_lost01','H32_lost0','H32_lost01')
train=[]
for arm in arms:
 folder=root/'fit_full'/arm
 complete=json.loads((folder/'training_complete.json').read_text())
 config=json.loads((folder/'config.json').read_text())
 history=json.loads((folder/'history.json').read_text())
 assert complete['status']=='COMPLETE_FULL_C_EXTENSION_TRAINING' and complete['epochs']==60
 assert complete['optimizer_updates']==config['required_optimizer_updates']==history[-1]['optimizer_updates']
 assert [r['epoch'] for r in history]==list(range(61))
 assert complete['strict_reload_pass'] and complete['M0_parent_exact']
 assert complete['last_epoch']==60 and complete['last_strict_reload_pass']
 assert complete['actual_changed_parameters'] and complete['actual_gradient_parameters']
 train.append({'arm':arm,'training':complete,'config':config})
internal=json.loads((root/'full98_report/full_recoverability_report.json').read_text())
assert internal['completed'] and internal['sequences']==98
selection=json.loads((root/'selected_model.json').read_text())
candidates=[arm+'_'+checkpoint for checkpoint in ('best','last') for arm in arms]
selected=selection['selected_candidate']
assert selected==max(candidates,key=lambda a:internal['variants'][a]['sequence_mean_iou'])
arm,checkpoint=selection['selected_arm'],selection['selected_checkpoint']
assert arm in arms and checkpoint in ('best','last') and selected==arm+'_'+checkpoint
assert selection['state_commit_model']==str(root/'fit_full'/arm/(checkpoint+'.pth'))
training=json.loads((root/'fit_full'/arm/'training_complete.json').read_text())
assert selection['selected_epoch']==(60 if checkpoint=='last' else training['best_epoch'])
assert selection['same_checkpoint_both_native_datasets']
assert Path(selection['state_commit_model']).is_file()
cleanup=json.loads((root/'unused_own_weight_cleanup.json').read_text())
assert cleanup['kept']==selection['state_commit_model'] and cleanup['old4_and_all_dependencies_preserved']
assert len(cleanup['removed'])==11 and all(not Path(row['path']).exists() for row in cleanup['removed'])
native=json.loads((root/'native/complete_metrics.json').read_text())
assert native['completed'] and len(native['five_metrics'])==5
assert native['same_state_commit_model_both_datasets']==selection['state_commit_model']
for dataset,expected in [('lasher',(245,220703)),('rgbt234',(234,116649))]:
 pred=root/'native'/dataset/'predictions'
 config=json.loads((pred/'inference_config.json').read_text())
 done=json.loads((pred/'inference_completion.json').read_text())
 assert done['completed'] and not done['smoke_only'] and (done['sequences'],done['frames'])==expected
 assert config['state_commit_model']==selection['state_commit_model'] and config['search_value']=='gross'
 report=json.loads((root/'native'/dataset/'core_report/full_report.json').read_text())
 assert report['all_actual_ground_truth_verified'] and (report['sequences'],report['frames'])==expected
 assert len(report['variants']['state_commit']['attributes'])==(19 if dataset=='lasher' else 12)
 mechanism=json.loads((root/'native'/dataset/'mechanism_report/full_recoverability_report.json').read_text())
 assert mechanism['completed'] and mechanism['sequences']==expected[0]
 counters=mechanism['variants']['state_commit']['counters']
 assert sum(counters[k] for k in ('state_regular_acceptance_frames','state_paused_acceptance_frames','state_geometry_holds'))==expected[1]-expected[0]
 assert 0<=counters['state_commit_interventions']<=expected[1]-expected[0]
 assert 0<=counters['failed_motion_with_correct_active_template_source_frames_localization_proxy']<=counters['failed_committed_motion_frames_localization_proxy']<=counters['known_geometry_commit_frames']
 for row in [r for r in native['five_metrics'] if r['dataset']==dataset]:
  metric=row['metric']
  actual=report['variants']['state_commit']['overall_metrics_percent'][metric]
  baseline=report['variants']['baseline']['overall_metrics_percent'][metric]
  assert abs(row['percent']-actual)<1e-10 and abs(row['baseline_percent']-baseline)<1e-10
  assert abs(row['delta_vs_baseline_pp']-(actual-baseline))<1e-10
  assert row['meets_plus_two']==(actual>=baseline+2)
 for reference in ('baseline','c1','old4','gross_parent'):
  paired=json.loads((root/'native'/dataset/(reference+'_paired_report')/'paired_bootstrap.json').read_text())
  assert paired['args']['iterations']==5000 and paired['args']['seed']==42
  assert paired['datasets'][dataset]['sequences']==expected[0]
assert native['all_five_plus_two']==all(r['meets_plus_two'] for r in native['five_metrics'])
archive=Path('/data/gb/setup/selective_state_completed_text_20261006.tar')
assert not archive.exists()
files=[]
for label,folder in [('training',root),('collection',collection)]:
 for p in sorted(folder.rglob('*')):
  if p.is_file() and (p.suffix in ('.json','.jsonl','.csv','.log','.txt','.png','.pdf') or p.name=='COMPLETE'):
   files.append((p,label+'/'+str(p.relative_to(folder))))
with tarfile.open(archive,'w') as stream:
 for p,name in files: stream.add(p,arcname=name,recursive=False)
receipt={'status':'COMPLETE_FULL_TRAINING_NATIVE_ARTIFACT_INTAKE','archive':str(archive),
 'archive_bytes':archive.stat().st_size,'files':len(files),'training':train,'selection':selection,
 'native':native,'cleanup':cleanup,'new_NN_forwards':0,'report_program_reruns':0}
print(json.dumps(receipt))
'''
run = subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'],input=payload,
                     text=True,encoding='utf-8',capture_output=True)
(base/'complete_intake_stdout.txt').write_text(run.stdout,encoding='utf-8')
(base/'complete_intake_stderr.txt').write_text(run.stderr,encoding='utf-8')
run.check_returncode()
receipt=json.loads(run.stdout)
assert shutil.disk_usage(base).free > receipt['archive_bytes'] * 2 + 256 * 1024**2
archive=base/'completed_text_archive.tar'
assert not archive.exists()
subprocess.run(['scp','2027:'+receipt['archive'],str(archive)],check=True)
assert archive.stat().st_size==receipt['archive_bytes']
target.mkdir()
with tarfile.open(archive) as stream:
    assert len(stream.getmembers())==receipt['files']
    assert all(n.startswith(('training/','collection/')) and '..' not in Path(n).parts for n in stream.getnames())
    stream.extractall(target)
(base/'actual_completed_artifact_intake.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({'status':receipt['status'],'files':receipt['files'],
                  'five_metrics':receipt['native']['five_metrics'],
                  'all_five_plus_two':receipt['native']['all_five_plus_two']}),flush=True)
