"""One near-finish observer of the47 TRAIN queries; no model or benchmark work."""
import datetime
import json
from pathlib import Path
import subprocess
import time

first = datetime.datetime.fromisoformat('2026-10-05T15:24:00+08:00')
time.sleep(max(0, first.timestamp() - time.time()))
code = """import datetime,json,pathlib,subprocess
b=pathlib.Path('/data/gb/outputs/recoverability_geometry_commit_train_20261005');rows=[]
for name in ['gpu0','gpu3']:
 r=json.loads((b/(name+'_launch.json')).read_text());p=b/name
 log=(b/(name+'.log')).read_text();ps=subprocess.run(['ps','-p',str(r['pid']),'-o','pid,stat,etimes,args','--no-headers'],capture_output=True,text=True)
 rows.append({'name':name,'complete':(p/'COMPLETE').is_file(),'owner_process':ps.stdout.strip(),'owner_present':ps.returncode==0,'log_tail':log[-1600:]})
print(json.dumps({'observed_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'rows':rows}))
"""
while True:
    p = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'], input=code,
                       capture_output=True, text=True, check=True)
    d = json.loads(p.stdout)
    Path('refine-logs/runs/recoverability_state_commit/actual_probe_completion_observation.json').write_text(json.dumps(d, indent=2) + '\n')
    print(json.dumps(d), flush=True)
    if all(row['complete'] for row in d['rows']):
        break
    assert all(row['complete'] or row['owner_present'] for row in d['rows']), d
    time.sleep(300)
