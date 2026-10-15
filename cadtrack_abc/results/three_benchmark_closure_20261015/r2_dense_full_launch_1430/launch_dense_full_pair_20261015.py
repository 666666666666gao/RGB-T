from pathlib import Path
import hashlib
import json
import subprocess

p=Path(__file__).parent
repo=Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source')
publication=json.loads((p/'r2_dense_admission_1425_publication.json').read_text())
assert publication['desktop_repo_git_server_same_bytes'] and publication['R2_gpu_qualified']
assert publication['R1_control_formal_complete'] and publication['R1_recovery_actual_exit_code']==0
proof=json.loads((repo/'cadtrack_abc/results/three_benchmark_closure_20261015/r2_dense_dependency_fix_1405/source_manifest.json').read_text())
expected=proof['sources']
dependencies=['ported_modules.py','pure_baseline_tracker.py','baseline.py','model.py','data.py',
    'vendor/CADTrack/lib/models/layers/head.py','vendor/CADTrack/lib/models/cadtrack/vit_care.py',
    'vendor/CADTrack/lib/utils/box_ops.py','vendor/CADTrack/experiments/cadtrack/cadtrack.yaml','dense_pipeline.py']
expected.update({'cadtrack_abc/'+name:hashlib.sha256((repo/'cadtrack_abc'/name).read_bytes()).hexdigest() for name in dependencies})
payload={'driver.py':(p/'dense_full_pair_driver_remote.py').read_text(encoding='utf-8'),
    'wrapper.py':(p/'dense_full_pair_wrapper_remote.py').read_text(encoding='utf-8')}
code='''from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,shutil,subprocess,time
project=Path('/home/gaob/CADTrack-ABC')
output=project/'outputs/r2_dense_pair_20261015'
qualification=project/'profile_output/r2_dense_m0_capacity128_20261015'
qualified=json.loads((qualification/'state.json').read_text())
assert qualified['status']=='DENSE_M0_COMPLETE' and qualified['chosen_common_batch']==128
assert json.loads((qualification/'exit.json').read_text())['actual_controller_exit_code']==0
expected=EXPECTED
assert {name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in expected}==expected
pretrained=Path('/home/gaob/CADTrack_ep0014.pth.tar')
assert hashlib.sha256(pretrained.read_bytes()).hexdigest()=='cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731'
resources=subprocess.run(['nvidia-smi','-i','2,3','--query-gpu=index,memory.used,memory.total,utilization.gpu',
    '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
assert {int(line.split(',')[0]) for line in resources.splitlines()}=={2,3}
assert all(int(line.split(',')[1])<500 for line in resources.splitlines()),resources
free=shutil.disk_usage(project).free
budget=pretrained.stat().st_size*20
assert free>budget
environment=json.loads((project/'cadtrack_abc/environment.json').read_text())
assert environment['reuse_mode']=='warm_existing_conda_no_package_changes'
output.mkdir(parents=True,exist_ok=False)
payload=PAYLOAD
for name,source in payload.items():
    (output/name).write_text(source)
preflight=dict(observed_utc=datetime.now(timezone.utc).isoformat(),gpu_memory_and_utilization_csv=resources,
    home_free_bytes=free,output_budget_bytes=budget,data_free_bytes=shutil.disk_usage('/data').free,
    expected_sources=expected,environment=environment,environment_rebuilt=False,
    pretrained_sha256=hashlib.sha256(pretrained.read_bytes()).hexdigest(),
    qualification_state_sha256=hashlib.sha256((qualification/'state.json').read_bytes()).hexdigest(),
    source_publication_commit=COMMIT,dataset_moved=False,temperature_power_queried=False)
(output/'preflight.json').write_text(json.dumps(preflight,indent=2))
wrapper=subprocess.Popen(['/data/wangwj/envs/CADTrack/bin/python',str(output/'wrapper.py')],
    cwd=project,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
receipt=dict(wrapper_pid=wrapper.pid,started_utc=datetime.now(timezone.utc).isoformat(),output=str(output),
    full_pipeline_launched=True,formal_training_started=False,preflight=preflight,
    phase='H GPU2 / HA GPU3; each20x20k raw8queryclips, sixfull98, same positive checkpoint all3',
    common_batch=128,pretrained_not_probe_initialization=True,
    new_finite_qualification_launched=False,new_full_training_launched=True,expected_physical_gpus=[2,3])
(output/'launch.json').write_text(json.dumps(receipt,indent=2))
time.sleep(5)
receipt['wrapper_actual_command']=subprocess.run(['ps','-p',str(wrapper.pid),'-o','args='],capture_output=True,text=True,check=True).stdout.strip()
assert str(output/'wrapper.py') in receipt['wrapper_actual_command']
pair=json.loads((output/'pair_state.json').read_text())
actual=[]
for label,gpu in [('H',2),('HA',3)]:
    queue=json.loads((output/label/'queue_state.json').read_text())
    phase=queue['phases']['full_training']
    assert len(phase['pids'])==1 and phase['physical_gpu']==gpu
    pid=phase['pids'][0]
    command=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True,check=True).stdout.strip()
    assert 'cadtrack_abc.train_dense' in command and '--batch 128' in command and '--epochs 20' in command
    actual.append(dict(variant=label,physical_gpu=gpu,neural_pid=pid,actual_command=command))
receipt.update(formal_training_started=True,actual_neural_jobs=actual,verified_utc=datetime.now(timezone.utc).isoformat())
(output/'launch_verified.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
'''.replace('EXPECTED',repr(expected)).replace('PAYLOAD',repr(payload)).replace('COMMIT',repr(publication['commit']))
(p/'launch_dense_full_pair_20261015.executed_source.py').write_text(code,encoding='utf-8')
command=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ServerAliveInterval=60','-o','ConnectTimeout=10','2028','python3 -']
completed=subprocess.run(command,input=code.encode(),capture_output=True)
(p/'launch_dense_full_pair_20261015.stdout.txt').write_bytes(completed.stdout)
(p/'launch_dense_full_pair_20261015.stderr.txt').write_bytes(completed.stderr)
completed.check_returncode()
receipt=json.loads(completed.stdout)
(p/'dense_full_pair_launch_verified.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='preflight'},indent=2),flush=True)
