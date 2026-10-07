"""Collect a terminal relation-control run once; no NN or report replay."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
base = parent / 'GOLA-source/refine-logs/runs/post_search_relation_control'
destination = base / 'complete'
assert not destination.exists()
observer = json.loads((base / 'sole_observer_complete.json').read_text())
assert observer['final_observation']['progress']['stage'] == 'COMPLETE_POST_SEARCH_RELATION_CONTROL_AND_NATIVE'
launch = json.loads((base / 'actual_pipeline_launch.json').read_text())
assert observer['final_observation']['owner_pid'] == launch['pid']

payload = '''import hashlib,json,shutil,tarfile
from pathlib import Path
root=Path('/data/gb/outputs/post_search_relation_control_20261007')
assert not Path('/proc/OWNER').exists()
def read(path):return json.loads(path.read_text())
progress=read(root/'progress.json');plan=read(root/'plan.json');selection=read(root/'selected_model.json')
assert progress['stage']=='COMPLETE_POST_SEARCH_RELATION_CONTROL_AND_NATIVE'
assert plan['seed']==42 and plan['batch_size']==336 and plan['full_epochs']==64 and plan['optimizer_updates_per_full_arm']==576
assert plan['train_policy_states']==2832 and plan['train_unique_queries']==2576 and plan['observed_pair_training']
check=read(root/'causal_pair_check.json')
assert check['completed'] and check['optimizer_updates']==0 and check['post_search_keep_reference_recomputed']
fits={};candidates=[]
for name,bidirectional,lr in plan['arms']:
 for stage,epochs in (('fit_sanity',2),('fit_full',64)):
  folder=root/stage/name;config=read(folder/'config.json');done=read(folder/'completion.json');history=read(folder/'metrics.json')
  assert done['completed'] and done['epochs']==epochs and done['optimizer_steps']==epochs*9
  assert config['train_clips']==2832 and config['validation_clips']==196 and config['observed_pair_training']
  assert config['post_search_bidirectional']==bidirectional
  assert config['module']==('ABC_post_search_relations' if bidirectional else 'ABC_candidate_relations')
  assert config['initial_checkpoint_epoch']==5 and config['initial_checkpoint_weights_exact']
  assert len({j[0] for j in config['train_jobs']})==881 and len({j[0] for j in config['validation_jobs']})==98
  assert len({(j[0],j[1]) for j in config['train_jobs']})==2576 and len({j[2] for j in config['train_jobs']})==2
  assert [r['epoch'] for r in history]==list(range(epochs+1))
  assert done['modules_changed']=={'A':True,'B':True,'C':True} and all(v>0 for v in done['max_module_gradient_norms'].values())
  assert done['relation_parameters_changed'] and done['max_relation_gradient_norm']>0
  assert done['strict_reload_metrics_equal'] and done['last_strict_reload_metrics_equal'] and done['frozen_c1_gradients_absent']
  if stage=='fit_full':fits[name]=dict(epochs=64,optimizer_steps=576,bidirectional=bidirectional,best_epoch=done['best_epoch'])
 for checkpoint in ('best','last'):
  label=name+'_'+checkpoint;candidates.append(label)
  done=read(root/'full98'/label/'predictions/inference_completion.json')
  assert done['completed'] and (done['sequences'],done['frames'])==(98,49418)
initial=read(root/'full98/post_search_initial/predictions/inference_completion.json')
assert initial['completed'] and initial['full_shard_union_verified'] and (initial['sequences'],initial['frames'])==(98,49418)
candidates.append('post_search_initial')
for gpu,offset,count in ((0,0,25),(1,25,25),(2,50,24),(3,74,24)):
 folder=root/'initial_shards'/('gpu'+str(gpu));config=read(folder/'inference_config.json');done=read(folder/'inference_completion.json')
 assert done['completed'] and done['sequences']==count and config['sequence_offset']==offset and config['limit_sequences']==count
 assert config['model']==str(root/'fit_full/post_search_lr5/initial.pth') and config['max_frames']==0
report=read(root/'full98_report/full_recoverability_report.json')
assert report['completed'] and report['sequences']==98
assert selection['same_checkpoint_both_native_datasets'] and not selection['native_metrics_completed']
assert selection['selected_candidate'] in candidates and selection['state_commit_model'] is None
assert report['variants'][selection['selected_candidate']]['sequence_mean_iou']==max(report['variants'][name]['sequence_mean_iou'] for name in candidates)
assert selection['sequence_mean_iou']=={name:v['sequence_mean_iou'] for name,v in report['variants'].items()}
model=Path(selection['parent_model']);assert model.is_file() and model.resolve().is_relative_to(root.resolve())
if selection['selected_checkpoint']=='initial':assert selection['selected_epoch']==0
native=read(root/'native/complete_metrics.json')
assert native['completed'] and native['same_ABC_model_both_datasets']==str(model) and native['same_state_commit_model_both_datasets'] is None
assert native['complete_attribute_settings']==31 and len(native['five_metrics'])==5
expected={'lasher':(245,220703,('PR','NPR','SR'),19),'rgbt234':(234,116649,('MPR','MSR'),12)}
assert {(r['dataset'],r['metric']) for r in native['five_metrics']}=={(d,m) for d,spec in expected.items() for m in spec[2]}
for dataset,(sequences,frames,metrics,attributes) in expected.items():
 core=read(root/'native'/dataset/'core_report/full_report.json')
 assert core['all_actual_ground_truth_verified'] and (core['sequences'],core['frames'])==(sequences,frames)
 output=core['variants'][native['model_label']]
 assert len(output['attributes'])==attributes and set(output['mean_curves'])==set(metrics)
 done=read(root/'native'/dataset/'predictions/inference_completion.json');config=read(root/'native'/dataset/'predictions/inference_config.json')
 assert done['completed'] and (done['sequences'],done['frames'])==(sequences,frames)
 assert config['model']==str(model) and config['validation_split'] is None and config['max_frames']==0
 for row in (r for r in native['five_metrics'] if r['dataset']==dataset):
  assert row['percent']==output['overall_metrics_percent'][row['metric']]
  assert row['baseline_percent']==core['variants']['baseline']['overall_metrics_percent'][row['metric']]
  assert row['delta_vs_baseline_pp']==row['percent']-row['baseline_percent']
  assert row['target_percent']==row['baseline_percent']+2 and row['meets_plus_two']==(row['percent']>=row['target_percent'])
assert native['all_five_plus_two']==all(r['meets_plus_two'] for r in native['five_metrics'])
cleanup=read(root/'unused_own_weight_cleanup.json')
assert cleanup['kept']==str(model) and len(cleanup['removed'])==19 and cleanup['protected_current_parent_old4_GOLA_C1_motion_untouched']
assert all(not Path(item['path']).exists() for item in cleanup['removed'])
archive=Path('/data/gb/setup/post_search_relation_control_completed_text_20261007.tar.gz');assert not archive.exists()
allowed={'.json','.jsonl','.csv','.txt','.log','.md','.png','.pdf'}
files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in allowed]
manifest=[dict(path='training/'+p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
size=sum(v['bytes'] for v in manifest);assert shutil.disk_usage(archive.parent).free>size+20*1024**2
with tarfile.open(archive,'w:gz') as stream:
 for p in files:stream.add(p,arcname='training/'+p.relative_to(root).as_posix(),recursive=False)
print(json.dumps(dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest=manifest,
 files=len(files),bytes=size,full_fit_summary=fits,selected_model=selection,native_complete=native,
 selected_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),completed_stage=progress['stage'],
 no_neural_training_inference_or_metric_report_replay=True)))
'''.replace('OWNER', str(launch['pid']))
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'], input=payload, text=True, encoding='utf-8', capture_output=True)
(base / 'complete_intake_stdout.txt').write_text(result.stdout, encoding='utf-8')
(base / 'complete_intake_stderr.txt').write_text(result.stderr, encoding='utf-8')
result.check_returncode()
receipt = json.loads(result.stdout)
assert shutil.disk_usage(base).free > 2 * receipt['bytes'] + 20 * 1024**2
archive = base / Path(receipt['archive']).name
subprocess.run(['scp', '2027:' + receipt['archive'], str(archive)], capture_output=True, check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['archive_sha256']
destination.mkdir()
with tarfile.open(archive) as stream: stream.extractall(destination)
actual = {p.relative_to(destination).as_posix(): p for p in destination.rglob('*') if p.is_file()}
assert set(actual) == {v['path'] for v in receipt['manifest']}
for item in receipt['manifest']:
    raw = actual[item['path']].read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
(base / 'actual_completed_artifact_intake.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: receipt[k] for k in ('files', 'bytes', 'full_fit_summary', 'selected_model', 'native_complete', 'completed_stage')}, ensure_ascii=False), flush=True)
