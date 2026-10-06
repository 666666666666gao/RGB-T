"""Collect the four completed original full-fit records once, without NN replay."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
base = parent / 'GOLA-source/refine-logs/runs/current_policy_aggregation'
destination = base / 'completed_full_fit_metadata'
assert not destination.exists()
payload = '''import hashlib,json,tarfile
from pathlib import Path
root=Path('/data/gb/outputs/current_policy_aggregation_20261006')
arms=[('old_lr5',72,2576,1),('aggregate_lr5',64,2832,2),('old_lr3',72,2576,1),('aggregate_lr3',64,2832,2)]
files=[];summary={}
for name,epochs,clips,policies in arms:
 folder=root/'fit_full'/name
 config=json.loads((folder/'config.json').read_text())
 done=json.loads((folder/'completion.json').read_text())
 history=json.loads((folder/'metrics.json').read_text())
 assert done['completed'] and done['epochs']==epochs and done['optimizer_steps']==576
 assert config['module']=='ABC_candidate_relations' and config['train_clips']==clips and config['validation_clips']==196
 assert config['initial_checkpoint_epoch']==5 and config['initial_checkpoint_weights_exact']
 assert len({j[0] for j in config['train_jobs']})==881 and len({j[0] for j in config['validation_jobs']})==98
 assert len({(j[0],j[1]) for j in config['train_jobs']})==2576 and len({j[2] for j in config['train_jobs']})==policies
 assert [r['epoch'] for r in history]==list(range(epochs+1))
 assert done['modules_changed']=={'A':True,'B':True,'C':True} and not done['frozen_modules']
 assert all(value>0 for value in done['max_module_gradient_norms'].values())
 assert done['relation_parameters_changed'] and done['max_relation_gradient_norm']>0
 assert done['frozen_c1_gradients_absent'] and done['strict_reload_metrics_equal'] and done['last_strict_reload_metrics_equal']
 summary[name]={'epochs':epochs,'actual_optimizer_updates':576,'train_policy_states':clips,
  'unique_train_queries':2576,'train_policy_count':policies,'best_epoch':done['best_epoch'],
  'elapsed_seconds':done['elapsed_seconds'],'max_module_gradient_norms':done['max_module_gradient_norms'],
  'relation_gradient':done['max_relation_gradient_norm'],'peak_cuda_mib':done['peak_cuda_mib'],
  'actual_ABC_relation_updates_and_best_last_reload_verified':True}
 files += [folder/name for name in ('config.json','completion.json','metrics.json','train.jsonl')]
archive=Path('/data/gb/setup/current_policy_completed_full_fit_metadata_20261007.tar.gz')
assert not archive.exists()
manifest=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,
           'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
with tarfile.open(archive,'w:gz') as stream:
 for path in files:stream.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
print(json.dumps({'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'manifest':manifest,'four_completed_full_fits':True,'total_actual_optimizer_updates':2304,
 'arms':summary,'current_actual_pipeline_stage':json.loads((root/'progress.json').read_text())['stage'],
 'new_formal_accuracy_acceptance':False,'no_NN_or_metrics_report_replay':True}))
'''
compile(payload, '<completed-full-fit-metadata>', 'exec')
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                        input=payload, text=True, encoding='utf-8', capture_output=True)
(base / 'full_fit_intake_stdout.txt').write_text(result.stdout, encoding='utf-8')
(base / 'full_fit_intake_stderr.txt').write_text(result.stderr, encoding='utf-8')
result.check_returncode()
receipt = json.loads(result.stdout)
archive = base / Path(receipt['archive']).name
subprocess.run(['scp', '2027:' + receipt['archive'], str(archive)], capture_output=True, check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['archive_sha256']
destination.mkdir()
with tarfile.open(archive) as stream: stream.extractall(destination)
actual_files = {p.relative_to(destination).as_posix(): p for p in destination.rglob('*') if p.is_file()}
assert set(actual_files) == {item['path'] for item in receipt['manifest']}
for item in receipt['manifest']:
    raw = actual_files[item['path']].read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
(base / 'actual_completed_full_fit_summary_20261007.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key:receipt[key] for key in ('four_completed_full_fits','total_actual_optimizer_updates',
                                            'arms','current_actual_pipeline_stage','new_formal_accuracy_acceptance')},ensure_ascii=False),flush=True)
