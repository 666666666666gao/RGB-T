"""One read-only observer: first inspect near the estimated collection finish, then every300s."""
import datetime
import json
import pathlib
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
zone = datetime.timezone(datetime.timedelta(hours=8))
deadline = datetime.datetime(2026, 10, 5, 9, 46, tzinfo=zone)
delay = max(0., (deadline - datetime.datetime.now(zone)).total_seconds())
time.sleep(delay)
target = pathlib.Path(r'C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source\refine-logs\runs\recoverability_train_events\actual_collection_capacity_and_full_start_observation.json')
code = """
import json,pathlib,subprocess,datetime
setup=pathlib.Path('/data/gb/setup')
states={name:json.loads((setup/(name+'_progress_20261005.json')).read_text()) for name in ('train_events_collection','train_events_fit')}
pids={1297701,1320786}
for value in states.values():
 pids.update(row['pid'] for row in value.get('children',[]))
actual=subprocess.run(['ps','-o','pid,ppid,stat,etimes,comm','-p',','.join(map(str,sorted(pids)))],capture_output=True,text=True)
shards=[]
for gpu in range(4):
 root=pathlib.Path('/data/gb/outputs/recoverability_train_events_collect_full_gpu'+str(gpu)+'_20261005')
 shards.append({'gpu':gpu,'progress':json.loads((root/'progress.json').read_text()),'completion':json.loads((root/'completion.json').read_text()) if (root/'completion.json').exists() else None})
receipts={}
for tag in ('full_cpu_acceptance','m0_fit_cpu_acceptance'):
 path=setup/('train_events_'+tag+'_20261005.json')
 if path.exists(): receipts[tag]=json.loads(path.read_text())
print(json.dumps({'observed_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'states':states,'collection_shards':shards,'actual_process_status':actual.stdout,'live_pids':[int(line.split()[0]) for line in actual.stdout.splitlines()[1:]],'receipts':receipts}))
"""
while True:
    output = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                            input=code, capture_output=True, text=True, encoding='utf-8', check=True)
    value = json.loads(output.stdout)
    target.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    stage = value['states']['train_events_fit']['stage']
    print(json.dumps({'at_cst': value['observed_at_cst'], 'fit_stage': stage,
                      'collection_stage': value['states']['train_events_collection']['stage'],
                      'clips': [row['progress']['completed_clips'] for row in value['collection_shards']],
                      'live_pids': value['live_pids']}), flush=True)
    if stage in ('FULL_FOUR_GPU_ABC_TRAINING_RUNNING', 'FULL_TRAINING_COMPLETE_CPU_RELOAD_AUDIT_RUNNING',
                 'FULL_ALL_FOUR_ACTUAL_TRAINING_AND_CPU_RELOAD_AUDIT_PASS',
                 'FOUR_COMPLETE_MATCHED420UPDATE_ABC_FITS_READY_FOR_FULL_VIDEO_SELECTION'):
        assert 'm0_fit_cpu_acceptance' in value['receipts']
        assert value['receipts']['m0_fit_cpu_acceptance']['status'] == 'PASS'
        break
    assert 1320786 in value['live_pids'], value
    time.sleep(300)
