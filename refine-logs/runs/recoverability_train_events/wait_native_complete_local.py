"""One local observer of the original1423909 native/report owner, no neural work."""
import datetime
import json
from pathlib import Path
import subprocess
import time

first = datetime.datetime.fromisoformat('2026-10-05T15:14:00+08:00')
time.sleep(max(0, first.timestamp() - time.time()))
code = """import datetime,json,pathlib,subprocess
p=pathlib.Path('/data/gb/setup/train_events_complete_evaluation_progress_20261005.json')
s=json.loads(p.read_text())
ps=subprocess.run(['ps','-p','1423909','-o','pid,stat,etimes,args','--no-headers'],capture_output=True,text=True)
print(json.dumps({'observed_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'state':s,'owner_process':ps.stdout.strip(),'owner_present':ps.returncode==0}))
"""
while True:
    result = subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'],
                            input=code,capture_output=True,text=True,check=True)
    observed = json.loads(result.stdout)
    Path('refine-logs/runs/recoverability_train_events/actual_native_completion_observation.json').write_text(json.dumps(observed,indent=2)+'\n')
    state = observed['state']
    print(json.dumps({'observed':observed['observed_at_cst'],'stage':state['stage'],'owner_present':observed['owner_present']}),flush=True)
    if state['stage'] == 'COMPLETE420_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY':
        assert Path('refine-logs/runs/recoverability_train_events/actual_native_completion_observation.json').is_file()
        break
    assert observed['owner_present'], observed
    time.sleep(300)
