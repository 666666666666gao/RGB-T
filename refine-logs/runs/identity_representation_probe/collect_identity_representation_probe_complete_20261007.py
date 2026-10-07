"""Copy the closed original probe and derive bounded CPU query diagnostics."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
base = parent / 'GOLA-source/refine-logs/runs/identity_representation_probe'
assert not (base / 'complete').exists()
terminal = json.loads((base / 'sole_observer_complete.json').read_text())['final_observation']
assert not terminal['owner_alive'] and terminal['progress']['stage'] == 'COMPLETE_IDENTITY_REPRESENTATION_PROBE'
payload = '''import csv,json,tarfile,hashlib
from pathlib import Path
import numpy as np
root=Path('/data/gb/outputs/identity_representation_probe_20261007')
report=json.loads((root/'complete_report.json').read_text())
assert report['completed'] and report['identity_only_pretraining'] and not report['native_accuracy']
assert len(report['results'])==4 and len(report['removed'])==7
assert Path(report['kept']).is_file() and all(not Path(v['path']).exists() for v in report['removed'])
for stage,epochs in [('fit_sanity',2),('fit_full',32)]:
 for name in report['results']:
  folder=root/stage/name;done=json.loads((folder/'completion.json').read_text());history=json.loads((folder/'metrics.json').read_text())
  assert done['completed'] and done['epochs']==epochs and done['parameters_changed'] and done['strict_reload_metrics_equal']
  assert [row['epoch'] for row in history]==list(range(epochs+1))
  if stage=='fit_full':assert done['optimizer_steps']==896
coverage={};validation=[];names={}
for partition,sequences,queries in [('train',881,14096),('validation',98,1568)]:
 identifiers=[];samples=[]
 for gpu in range(4):
  folder=root/'full_features'/('gpu'+str(gpu))/partition
  done=json.loads((folder/'completion.json').read_text());cfg=json.loads((folder/'config.json').read_text())
  assert done['completed'] and done['optimizer_updates']==0 and done['frozen_base_gradients_absent']
  names.update({int(k):v for k,v in cfg['sequence_names'].items()})
  with np.load(folder/'samples.npz') as archive:
   identifiers.extend(archive['sequence_index'].tolist());samples.extend(archive['sample_index'].tolist())
   if partition=='validation':validation.append({k:archive[k].copy() for k in ['sequence_index','sample_index','quality','valid','c1_score']})
 assert len(identifiers)==queries and len(set(identifiers))==sequences
 assert np.array_equal(np.sort(samples),np.arange(queries))
 coverage[partition]=dict(sequences=sequences,queries=queries)
data={k:np.concatenate([row[k] for row in validation]) for k in validation[0]}
choice=np.where(data['valid'],data['c1_score'],-np.inf).argmax(-1)
c1=data['quality'][np.arange(len(choice)),choice]
oracle=np.where(data['valid'],data['quality'],-1).max(-1)
assert abs(float(c1.mean())-report['c1_query_mean_iou'])<1e-6
selected={name:np.load(root/'fit_full'/name/'best_selected_iou.npy') for name in report['results']}
diagnosis=dict(validation_queries=1568,candidate_recall_at5=float((oracle>=.5).mean()),
 c1_failed_queries=int((c1<.2).sum()),c1_failed_candidate_present_queries=int(((c1<.2)&(oracle>=.5)).sum()),
 baseline_query_mean_iou=float(c1.mean()),scope='Supervised sampled localization queries, not native SR or own-policy rollouts',
 epoch_metrics={})
for name,row in report['results'].items():
 history=json.loads((root/'fit_full'/name/'metrics.json').read_text())
 diagnosis['epoch_metrics'][name]=dict(initial=history[0],best=history[row['best_epoch']],last=history[-1])
 for key in ['rescued','harmed']:
  value=((c1<.2)&(selected[name]>=.5)).sum() if key=='rescued' else ((c1>=.5)&(selected[name]<.2)).sum()
  assert int(value)==row['validation_metrics'][key]
with (root/'validation_per_query.csv').open('w',newline='') as stream:
 writer=csv.writer(stream);writer.writerow(['sequence','sample_index','C1_iou','oracle_best_iou',*report['results']])
 for i in range(len(c1)):writer.writerow([names[int(data['sequence_index'][i])],int(data['sample_index'][i]),float(c1[i]),float(oracle[i]),*[float(selected[n][i]) for n in report['results']]])
(root/'derived_query_diagnostics.json').write_text(json.dumps(diagnosis,indent=2))
archive=Path('/data/gb/setup/identity_representation_probe_completed_text_20261007.tar.gz');assert not archive.exists()
allowed={'.json','.log','.txt','.csv','.npy'}
files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in allowed]
with tarfile.open(archive,'w:gz') as stream:
 for p in files:stream.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
print(json.dumps(dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),
 complete_report=report,coverage=coverage,derived_query_diagnostics=diagnosis,no_neural_or_GT_metric_replay=True)))
'''
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'], input=payload,
    text=True, encoding='utf-8', capture_output=True)
(base / 'complete_intake_stdout.txt').write_text(result.stdout, encoding='utf-8')
(base / 'complete_intake_stderr.txt').write_text(result.stderr, encoding='utf-8')
result.check_returncode()
receipt = json.loads(result.stdout)
archive = parent / Path(receipt['archive']).name
assert not archive.exists()
subprocess.run(['scp', '2027:' + receipt['archive'], str(archive)], capture_output=True, check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['archive_sha256']
with tarfile.open(archive) as stream:
    stream.extractall(base / 'complete')
assert len([p for p in (base / 'complete').rglob('*') if p.is_file()]) == receipt['files']
(base / 'actual_complete_intake.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key:receipt[key] for key in ['files','coverage','complete_report','derived_query_diagnostics']}), flush=True)
