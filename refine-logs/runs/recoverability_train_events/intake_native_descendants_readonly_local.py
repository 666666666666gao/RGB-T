"""Read actual native descendants and original already completed cleanup; delete nothing."""
from pathlib import Path
import json
import subprocess
r=Path('.')
code=r"""import datetime,json,pathlib,subprocess,hashlib
root=pathlib.Path('/data/gb/GOLA');setup=pathlib.Path('/data/gb/setup')
state=json.loads((setup/'train_events_complete_evaluation_progress_20261005.json').read_text())
selection=json.loads((setup/'train_events_native_selection_20261005.json').read_text())
cleanup=json.loads((setup/'train_events_unselected_full_weight_cleanup_20261005.json').read_text())
text=subprocess.check_output(['ps','-eo','pid,ppid,stat,etimes,comm,args','--no-headers'],text=True)
rows=[]
for line in text.splitlines():
 fields=line.split(None,5)
 if len(fields)==6:rows.append({'pid':int(fields[0]),'ppid':int(fields[1]),'state':fields[2],'elapsed':int(fields[3]),'command':fields[4],'args':fields[5]})
ids={1423909};changed=True
while changed:
 added={x['pid'] for x in rows if x['ppid'] in ids};changed=not added<=ids;ids|=added
actual=[x for x in rows if x['pid'] in ids]
logs=[]
for child in state['children']:
 records=[]
 for line in pathlib.Path(child['log']).read_text(errors='replace').splitlines():
  if line.startswith('SEQUENCE '):records.append(json.loads(line.split(' ',1)[1]))
 logs.append({'gpu':child['gpu'],'log':child['log'],'completed_sequence_records':len(records),'completed_frames':sum(x['frames'] for x in records),'last_sequence':records[-1]['sequence'] if records else None,'completed_sequence_elapsed_sum':sum(x['elapsed_seconds'] for x in records)})
selected=pathlib.Path(selection['checkpoint'])
old4=pathlib.Path('/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth')
print(json.dumps({'observed_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'evaluation':state,'selection':selection,'actual_descendant_processes':actual,'actual_completed_native_log_records':logs,'unselected_cleanup':cleanup,'cleanup_removed_paths_actually_missing':all(not pathlib.Path(x['path']).exists() for x in cleanup['deleted']),'selected_checkpoint_actually_present':selected.exists(),'selected_checkpoint_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),'retained_old4_sha256':hashlib.sha256(old4.read_bytes()).hexdigest(),'GPU_queries':0,'new_NN_calls':0,'new_deletions':0}))
"""
p=subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'],input=code,capture_output=True,text=True)
assert p.returncode==0,p.stderr
d=json.loads(p.stdout)
assert d['evaluation']['pid']==1423909 and d['selection']['status']=='PASS'
assert len(d['selection']['candidates'])==16 and d['selection']['selected_best_epoch']==3
assert d['cleanup_removed_paths_actually_missing'] and d['selected_checkpoint_actually_present']
assert d['retained_old4_sha256']=='a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
assert len(d['unselected_cleanup']['deleted'])==15
(r/'refine-logs/runs/recoverability_train_events/actual_full98_selection_native_descendants_intake.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'observed':d['observed_at_cst'],'phase':d['evaluation']['stage'],
 'actual_neural_descendant_processes':[{'pid':x['pid'],'ppid':x['ppid'],'command':x['command'],'state':x['state']} for x in d['actual_descendant_processes']],
 'logs':d['actual_completed_native_log_records'],'deleted_count':len(d['unselected_cleanup']['deleted']),
 'deleted_bytes':sum(x['bytes'] for x in d['unselected_cleanup']['deleted']),
 'selected_checkpoint_sha256':d['selected_checkpoint_sha256'],'old4_unchanged':True}))
