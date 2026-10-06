"""Plot actual closed training histories and full98 outcomes; no model execution."""
import json
import subprocess
from pathlib import Path

base=Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/selective_state_commit')
out=base/'closed_training_plots'
assert not out.exists()
payload='''import csv,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/data/gb/outputs/selective_state_training_endpoint_20261006')
out=root/'closed_training_plots'
out.mkdir(exist_ok=False)
arms=('H3_lost0','H3_lost01','H32_lost0','H32_lost01')
fig,axes=plt.subplots(2,4,figsize=(16,7))
rows=[]
for col,arm in enumerate(arms):
 history=json.loads((root/'fit_full'/arm/'history.json').read_text())
 assert [r['epoch'] for r in history]==list(range(61))
 axes[0,col].plot([r['epoch'] for r in history[1:]], [r['loss'] for r in history[1:]])
 axes[0,col].set(title=arm,ylabel='Training objective',xlabel='Epoch')
 axes[1,col].plot([r['epoch'] for r in history],
                  [100*r['validation']['event_weighted_utility_advantage'] for r in history])
 axes[1,col].axhline(0,color='black',lw=.8)
 axes[1,col].set(xlabel='Epoch',ylabel='Cached validation advantage (pp)')
 for axis in axes[:,col]: axis.grid(alpha=.2)
 for r in history:
  rows.append({'arm':arm,'epoch':r['epoch'],'optimizer_updates':r['optimizer_updates'],
   'loss':r['loss'] if r['epoch'] else '',
   'validation_advantage':r['validation']['event_weighted_utility_advantage'],
   'validation_changed_queries':r['validation']['changed_queries']})
fig.suptitle('C extension only; frozen pretrained GOLA/C1/old4 A/B. Reused developer validation.')
fig.tight_layout()
for ext in ('png','pdf'): fig.savefig(out/('training_histories.'+ext),dpi=150)
plt.close(fig)
with (out/'training_history.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report=json.loads((root/'full98_report/full_recoverability_report.json').read_text())
assert report['completed'] and report['sequences']==98
names=['baseline','c1','old4','gross_parent']+[a+'_'+c for c in ('best','last') for a in arms]
scores=[report['variants'][name]['sequence_mean_iou']*100 for name in names]
fig,axis=plt.subplots(figsize=(10,6))
axis.barh(names[::-1],scores[::-1],color=['#b34b4b' if score<scores[3] else '#28786b' for score in scores[::-1]])
axis.axvline(scores[3],color='black',ls='--',label='Frozen old4/gross parent')
axis.set(xlabel='Full98 sequence mean IoU (%) — not official SR',xlim=(45,78))
axis.grid(axis='x',alpha=.2);axis.legend(loc='lower right')
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(out/('full98_results.'+ext),dpi=150)
plt.close(fig)
print(json.dumps({'status':'CLOSED_TRAINING_PLOTS_COMPLETE','history_rows':len(rows),
 'full98_variants':len(names),'new_NN_forwards':0,'new_optimizer_updates':0,
 'files':[str(p) for p in sorted(out.iterdir())]}))
'''
compile(payload,'plot_closed_training','exec')
run=subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'],input=payload,
                   text=True,encoding='utf-8',capture_output=True)
(base/'closed_training_plots_stdout.txt').write_text(run.stdout,encoding='utf-8')
(base/'closed_training_plots_stderr.txt').write_text(run.stderr,encoding='utf-8')
run.check_returncode()
receipt=json.loads(run.stdout)
assert receipt['history_rows']==244 and receipt['full98_variants']==12
out.mkdir()
for path in receipt['files']:
 subprocess.run(['scp','2027:'+path,str(out/Path(path).name)],check=True)
(base/'closed_training_plots_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)
