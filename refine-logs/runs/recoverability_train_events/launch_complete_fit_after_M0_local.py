"""Continue once at the actual failed log-creation boundary; no collection or M0 replay."""
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')
root = pathlib.Path(r'C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source')
folder = 'refine-logs/runs/recoverability_train_events/'
review = json.loads((root / folder / 'complete_fit_recovery_source_review.json').read_text())
assert review['status'] == 'PASS' and review['runtime_attested'] is False
names = [folder + name for name in ('complete_fit_after_M0.py', 'evaluation_after_complete_fit.py',
         'complete_fit_recovery_source_review.json', 'actual_after_M0_controller_failure_intake.json')]
expected = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
fit_receipt = root / folder / 'actual_complete_fit_after_M0_launch.json'
eval_receipt = root / folder / 'actual_complete_evaluation_after_recovery_launch.json'
assert not fit_receipt.exists() and not eval_receipt.exists()
for name in names:
    subprocess.run(['scp', str(root / name), '2027:/data/gb/GOLA/' + name], check=True, capture_output=True, text=True)
code = "import json,pathlib,hashlib; names=" + repr(names) + "; print(json.dumps({n:hashlib.sha256((pathlib.Path('/data/gb/GOLA')/n).read_bytes()).hexdigest() for n in names}))"
reply = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -c ' + shlex.quote(code)],
                       check=True, capture_output=True, text=True)
assert json.loads(reply.stdout) == expected
fit_code = """
import datetime,json,os,pathlib,subprocess
root=pathlib.Path('/data/gb/GOLA')
state=pathlib.Path('/data/gb/setup/train_events_complete_fit_progress_20261005.json')
assert not state.exists()
env=os.environ | {'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(root),'LD_LIBRARY_PATH':'/data/gb/envs/gola/lib','OMP_NUM_THREADS':'4'}
log=pathlib.Path('/data/gb/setup/train_events_complete_fit_controller_20261005.log')
with log.open('x') as stream:
 process=subprocess.Popen(['/data/gb/envs/gola/bin/python','-u',str(root/'refine-logs/runs/recoverability_train_events/complete_fit_after_M0.py')],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'coordinator_pid':process.pid,'started_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'log':str(log),'progress':str(state),'purpose':'Four previously unstarted full420update ABC fits after actual716GT and fourM0 CPU PASS; collection and M0 not repeated; only full-fit log namespace corrected'}))
"""
reply = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -c ' + shlex.quote(fit_code)],
                       check=True, capture_output=True, text=True)
fit = json.loads(reply.stdout)
fit['deployed_files_sha256'] = expected
fit_receipt.write_text(json.dumps(fit, indent=2) + '\n', encoding='utf-8')
subprocess.run(['scp', str(fit_receipt), '2027:/data/gb/GOLA/' + folder + fit_receipt.name], check=True, capture_output=True, text=True)
eval_code = """
import datetime,json,os,pathlib,subprocess
root=pathlib.Path('/data/gb/GOLA')
fit=json.loads((root/'refine-logs/runs/recoverability_train_events/actual_complete_fit_after_M0_launch.json').read_text())
current=json.loads(pathlib.Path(fit['progress']).read_text())
assert current['pid']==fit['coordinator_pid'] and current['stage']=='FULL_FOUR_GPU_ABC_TRAINING_RUNNING'
actual=subprocess.run(['ps','-o','stat=','-p',str(fit['coordinator_pid'])],check=True,capture_output=True,text=True).stdout.strip()
assert actual and not actual.startswith('Z')
state=pathlib.Path('/data/gb/setup/train_events_complete_evaluation_progress_20261005.json')
assert not state.exists()
env=os.environ | {'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(root),'LD_LIBRARY_PATH':'/data/gb/envs/gola/lib','OMP_NUM_THREADS':'4'}
log=pathlib.Path('/data/gb/setup/train_events_complete_evaluation_controller_20261005.log')
with log.open('x') as stream:
 process=subprocess.Popen(['/data/gb/envs/gola/bin/python','-u',str(root/'refine-logs/runs/recoverability_train_events/evaluation_after_complete_fit.py')],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'coordinator_pid':process.pid,'started_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'log':str(log),'progress':str(state),'upstream_complete_fit_pid':fit['coordinator_pid'],'purpose':'Wait actual four full420update fits and CPU acceptance, then unchanged full98 selection/full479 native scoring/31attributes/curves/paired bootstrap/mechanisms/final report; no original waiter restart'}))
"""
reply = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -c ' + shlex.quote(eval_code)],
                       check=True, capture_output=True, text=True)
evaluation = json.loads(reply.stdout)
eval_receipt.write_text(json.dumps(evaluation, indent=2) + '\n', encoding='utf-8')
subprocess.run(['scp', str(eval_receipt), '2027:/data/gb/GOLA/' + folder + eval_receipt.name], check=True, capture_output=True, text=True)
print(json.dumps({'fit': fit, 'evaluation': evaluation}))
