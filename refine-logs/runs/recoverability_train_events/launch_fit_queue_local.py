"""Deploy the reviewed full-fit successor and start one queue behind active collection."""
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')
root = pathlib.Path(r'C:\Users\gb\.codex_tmp\gola_setup_20261002\GOLA-source')
folder = 'refine-logs/runs/recoverability_train_events/'
review = json.loads((root / folder / 'fit_source_review.json').read_text())
assert review['status'] == 'PASS' and review['runtime_attested'] is False
names = [folder + name for name in ['fit_plan.json', 'fit_pipeline.py', 'audit_fit_cpu.py',
         'train_fixed_epochs_private.py', 'fit_source_review.json', 'cpu_staging_trainer_source_review_corrected.json']]
expected = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
receipt_path = root / folder / 'actual_fit_queue_launch.json'
assert not receipt_path.exists()
for name in names:
    subprocess.run(['scp', str(root / name), '2027:/data/gb/GOLA/' + name],
                   capture_output=True, text=True, check=True)
verify = "import json,pathlib,hashlib; names=" + repr(names) + "; print(json.dumps({n:hashlib.sha256((pathlib.Path('/data/gb/GOLA')/n).read_bytes()).hexdigest() for n in names}))"
reply = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -c ' + shlex.quote(verify)],
                       capture_output=True, text=True, check=True)
assert json.loads(reply.stdout) == expected
code = """
import os,pathlib,subprocess,json,datetime
root=pathlib.Path('/data/gb/GOLA')
state=pathlib.Path('/data/gb/setup/train_events_fit_progress_20261005.json')
assert not state.exists()
env=os.environ | {'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(root),'LD_LIBRARY_PATH':'/data/gb/envs/gola/lib','OMP_NUM_THREADS':'4'}
log=pathlib.Path('/data/gb/setup/train_events_fit_controller_20261005.log')
with log.open('x') as stream:
    process=subprocess.Popen(['/data/gb/envs/gola/bin/python','-u',str(root/'refine-logs/runs/recoverability_train_events/fit_pipeline.py')],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'coordinator_pid':process.pid,'started_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'log':str(log),'progress':str(state),'purpose':'Wait actual716 TRAIN GT acceptance, then four capacity fits and four matched420update complete ABC fits; not yet optimization at queue startup'}))
"""
reply = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -c ' + shlex.quote(code)],
                       capture_output=True, text=True, check=True)
receipt = json.loads(reply.stdout)
receipt['deployed_files_sha256'] = expected
receipt['source_review'] = 'same-family provisional source PASS; runtime not attested by reviewer'
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
subprocess.run(['scp', str(receipt_path), '2027:/data/gb/GOLA/' + folder + receipt_path.name],
               capture_output=True, text=True, check=True)
print(json.dumps(receipt, ensure_ascii=False))
