"""Collect the completed original aggregation run once; no NN or report replay."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
base = parent / 'GOLA-source/refine-logs/runs/current_policy_aggregation'
destination = base / 'complete'
assert not destination.exists()
observer = json.loads((base / 'sole_observer_complete.json').read_text())
stage = 'COMPLETE_CURRENT_POLICY_AGGREGATION_FULL_TRAINING_AND_NATIVE'
assert observer['final_observation']['progress']['stage'] == stage
launch = json.loads((base / 'actual_pipeline_launch.json').read_text())
assert observer['final_observation']['owner_pid'] == launch['pid']

payload = '''import hashlib,json,shutil,tarfile
from pathlib import Path
root=Path('/data/gb/outputs/current_policy_aggregation_20261006')
assert not Path('/proc/OWNER_PID').exists()
def read(path):return json.loads(path.read_text())
progress=read(root/'progress.json')
assert progress['stage']=='COMPLETE_CURRENT_POLICY_AGGREGATION_FULL_TRAINING_AND_NATIVE'
plan=read(root/'plan.json');selection=read(root/'selected_model.json')
assert plan['seed']==42 and plan['batch_size']==336 and plan['optimizer_updates_per_full_arm']==576
assert plan['aggregate_queries']==2832 and plan['aggregate_unique_sequence_query_pairs']==2576
assert selection['same_checkpoint_both_native_datasets'] and not selection['native_metrics_completed']
assert selection['selection_scope'].startswith('Eight completed checkpoints on reused developer98')
assert selection['state_commit_model'] is None
model=Path(selection['parent_model'])
assert model.is_file() and model.resolve().is_relative_to(root.resolve())
fits={}
for name,aggregate,lr,epochs in plan['arms']:
 folder=root/'fit_full'/name
 config=read(folder/'config.json');done=read(folder/'completion.json');history=read(folder/'metrics.json')
 assert done['completed'] and done['epochs']==epochs and done['optimizer_steps']==576
 assert config['module']=='ABC_candidate_relations' and config['train_clips']==(2832 if aggregate else 2576)
 assert config['validation_clips']==196 and config['initial_checkpoint_epoch']==5 and config['initial_checkpoint_weights_exact']
 assert len({j[0] for j in config['train_jobs']})==881 and len({j[0] for j in config['validation_jobs']})==98
 assert len({(j[0],j[1]) for j in config['train_jobs']})==2576
 assert len({j[2] for j in config['train_jobs']})==(2 if aggregate else 1)
 assert [r['epoch'] for r in history]==list(range(epochs+1))
 assert done['modules_changed']=={'A':True,'B':True,'C':True} and not done['frozen_modules']
 assert all(v>0 for v in done['max_module_gradient_norms'].values())
 assert done['relation_parameters_changed'] and done['max_relation_gradient_norm']>0
 assert done['frozen_c1_gradients_absent'] and done['strict_reload_metrics_equal'] and done['last_strict_reload_metrics_equal']
 fits[name]={'epochs':epochs,'optimizer_steps':576,'train_policy_states':config['train_clips'],
             'unique_train_queries':2576,'train_policy_count':2 if aggregate else 1,
             'best_epoch':done['best_epoch'],'ABC_and_relation_gradients_and_updates_verified':True}
 for checkpoint in ('best','last'):
  completion=read(root/'full98'/(name+'_'+checkpoint)/'predictions/inference_completion.json')
  assert completion['completed'] and (completion['sequences'],completion['frames'])==(98,49418)
assert len(fits)==4
full98=read(root/'full98_report/full_recoverability_report.json')
assert full98['completed'] and full98['sequences']==98
assert selection['sequence_mean_iou']=={k:v['sequence_mean_iou'] for k,v in full98['variants'].items()}
candidates=[name+'_'+checkpoint for name in fits for checkpoint in ('best','last')]
assert selection['selected_candidate'] in candidates
assert full98['variants'][selection['selected_candidate']]['sequence_mean_iou']==max(full98['variants'][label]['sequence_mean_iou'] for label in candidates)
native=read(root/'native/complete_metrics.json')
assert native['completed'] and native['same_ABC_model_both_datasets']==str(model)
assert native['same_state_commit_model_both_datasets'] is None
assert native['complete_attribute_settings']==31 and len(native['five_metrics'])==5
expected={'lasher':(245,220703,('PR','NPR','SR'),19),'rgbt234':(234,116649,('MPR','MSR'),12)}
assert {(row['dataset'],row['metric']) for row in native['five_metrics']}=={(d,m) for d,spec in expected.items() for m in spec[2]}
for dataset,(sequences,frames,metrics,attributes) in expected.items():
 report=read(root/'native'/dataset/'core_report/full_report.json')
 assert report['all_actual_ground_truth_verified'] and (report['sequences'],report['frames'])==(sequences,frames)
 output=report['variants'][native['model_label']]
 assert len(output['attributes'])==attributes and set(output['mean_curves'])==set(metrics)
 inference=read(root/'native'/dataset/'predictions/inference_completion.json')
 assert inference['completed'] and (inference['sequences'],inference['frames'])==(sequences,frames)
 config=read(root/'native'/dataset/'predictions/inference_config.json')
 assert config['model']==str(model) and config['validation_split'] is None and config['max_frames']==0
 for row in (r for r in native['five_metrics'] if r['dataset']==dataset):
  assert row['percent']==output['overall_metrics_percent'][row['metric']]
  assert row['baseline_percent']==report['variants']['baseline']['overall_metrics_percent'][row['metric']]
  assert row['delta_vs_baseline_pp']==row['percent']-row['baseline_percent']
  assert row['target_percent']==row['baseline_percent']+2
  assert row['meets_plus_two']==(row['percent']>=row['target_percent'])
assert native['all_five_plus_two']==all(r['meets_plus_two'] for r in native['five_metrics'])
cleanup=read(root/'unused_own_weight_cleanup.json')
assert cleanup['kept']==str(model) and len(cleanup['removed'])==15
assert cleanup['protected_current_parent_old4_GOLA_C1_motion_untouched']
assert all(not Path(row['path']).exists() for row in cleanup['removed'])
archive=Path('/data/gb/setup/current_policy_aggregation_completed_text_20261006.tar.gz')
assert not archive.exists()
allowed={'.json','.jsonl','.csv','.txt','.log','.md','.png','.pdf'}
files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in allowed]
manifest=[{'path':'training/'+p.relative_to(root).as_posix(),'bytes':p.stat().st_size,
           'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
size=sum(item['bytes'] for item in manifest)
assert shutil.disk_usage(archive.parent).free>size+20*1024**2
with tarfile.open(archive,'w:gz') as stream:
 for path in files:stream.add(path,arcname='training/'+path.relative_to(root).as_posix(),recursive=False)
print(json.dumps({'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'files':len(files),'bytes':size,'manifest':manifest,'full_fit_summary':fits,
 'selected_model':selection,'native_complete':native,'completed_stage':progress['stage'],
 'selected_model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
 'no_neural_training_inference_or_metric_report_replay':True}))
'''.replace('OWNER_PID', str(launch['pid']))
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                        input=payload, text=True, encoding='utf-8', capture_output=True)
(base / 'complete_intake_stdout.txt').write_text(result.stdout, encoding='utf-8')
(base / 'complete_intake_stderr.txt').write_text(result.stderr, encoding='utf-8')
result.check_returncode()
receipt = json.loads(result.stdout)
assert shutil.disk_usage(base).free > 2 * receipt['bytes'] + 20 * 1024**2
archive = base / Path(receipt['archive']).name
subprocess.run(['scp', '2027:' + receipt['archive'], str(archive)], capture_output=True, check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['archive_sha256']
destination.mkdir()
with tarfile.open(archive) as stream:
    stream.extractall(destination)
actual_files = {p.relative_to(destination).as_posix(): p for p in destination.rglob('*') if p.is_file()}
assert set(actual_files) == {item['path'] for item in receipt['manifest']}
for item in receipt['manifest']:
    raw = actual_files[item['path']].read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
(base / 'actual_completed_artifact_intake.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: receipt[k] for k in ('files', 'bytes', 'full_fit_summary', 'selected_model',
                                        'native_complete', 'completed_stage')}, ensure_ascii=False), flush=True)
