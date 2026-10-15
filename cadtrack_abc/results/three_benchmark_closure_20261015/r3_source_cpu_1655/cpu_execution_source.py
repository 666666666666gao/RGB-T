from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path('/home/gaob/CADTrack-ABC')
expected={'cadtrack_abc/dense_temporal.py': '5b992ff1b6a05e6207eaab86904f419ffc2468992c366e5648db5800d300c6cc', 'cadtrack_abc/dense_submission.py': '869afb855de1fe7d1a3975afc40e5353006e981e7008c0ea5b393709e65b4fba', 'cadtrack_abc/dense_main_model.py': '467c87d70d1dd2b5bd9808c724c5341e30a77dcfd15da912ca9fe267c91b923c', 'cadtrack_abc/dense_main_tracker.py': '183c96a1a0fecd6eec3992208ec80ece80b0e59c750443a284a400e5f98b03d2', 'cadtrack_abc/train_dense_motion.py': '9c025cbf7f83d166163460d9472c21dd5b4f5a4dbc41f65c436489526ede875d', 'cadtrack_abc/check_dense_main_contracts.py': '92268f2ea6e8f11e221e7c0519b0ad91ee148bdc05ecaaba8b5e53ff15df989b'}
assert {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in expected}==expected
output=root/'profile_output/dense_r3_cpu_contracts_20261015'
output.mkdir(exist_ok=False)
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
command=['/data/wangwj/envs/CADTrack/bin/python','-m','cadtrack_abc.check_dense_main_contracts','--output',str(output/'contracts.json')]
started=datetime.now(timezone.utc).isoformat()
begin=time.monotonic()
with (output/'stdout.txt').open('w') as stdout,(output/'stderr.txt').open('w') as stderr:
    result=subprocess.run(command,cwd=root,env=environment,stdout=stdout,stderr=stderr)
receipt=dict(started_utc=started,ended_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-begin,
    command=command,actual_exit_code=result.returncode,CUDA_VISIBLE_DEVICES='',sources=expected,
    scope='CPU synthetic component contracts only, no real images/pretrained forward/GPU execution.',
    new_GPU_jobs=0,training_source_or_configuration_changed=False,full_ABC_training_complete=False,formal_metrics=None)
(output/'execution.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(receipt,stdout=(output/'stdout.txt').read_text(),stderr=(output/'stderr.txt').read_text()),indent=2),flush=True)
raise SystemExit(result.returncode)
