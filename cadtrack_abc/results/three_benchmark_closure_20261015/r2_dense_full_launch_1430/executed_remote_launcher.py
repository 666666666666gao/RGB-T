from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,shutil,subprocess,time
project=Path('/home/gaob/CADTrack-ABC')
output=project/'outputs/r2_dense_pair_20261015'
qualification=project/'profile_output/r2_dense_m0_capacity128_20261015'
qualified=json.loads((qualification/'state.json').read_text())
assert qualified['status']=='DENSE_M0_COMPLETE' and qualified['chosen_common_batch']==128
assert json.loads((qualification/'exit.json').read_text())['actual_controller_exit_code']==0
expected={'cadtrack_abc/dense_model.py': '079afa56818a997001b3766b3c0ee5826a8d4c644c8dc09c4a6840de02649364', 'cadtrack_abc/dense_tracker.py': '03f5f5b76dea623e4cfdff7b7eb6b6e01b2afb4ce35ca693724a7907e9e1d318', 'cadtrack_abc/train_dense.py': '4e3064dd8efa8d5752242c755c03be09d2faa7fe05358406eb1c6d27065decd6', 'cadtrack_abc/dense_evaluation.py': '9b264f38664675d401760abc6341fbd8bd32112ef5088b49a571ec0cdc9859f3', 'cadtrack_abc/infer_dense.py': '52fa8325efccea691470da2358a0bf2c9afa2d4cf85bea7fd8a4d7a73f2673cd', 'cadtrack_abc/validate_dense.py': '4efb7644ae6be3b02d1d539589a387d53d79e783fbb6075ce6855e36450f9ccc', 'cadtrack_abc/select_dense.py': '1938a7fff1c22aa54990cbda5ef5619c8347cb3f290670e1fa88a23e1c2b44b0', 'cadtrack_abc/score_dense.py': '0d88208f2e2708e9fdd8bb4f99785108067521ffa6ef42ec93f008560fcb9e21', 'cadtrack_abc/qualify_dense.py': '89001a6c785e479759a87bda12b5c95d9f2f0d4da265acbca6b23435f722c13c', 'cadtrack_abc/vendor/CADTrack/lib/utils/focal_loss.py': '11ccd7a16af34a3f54bd36191f086f7b0d85e95669e07bc072deb6226b40a965', 'cadtrack_abc/vendor/CADTrack/lib/utils/heapmap_utils.py': 'cc5ddbc0d65cad55f1a727f43e1d9966caa17a10c043a10586d0a13c1d4f3d4d', 'cadtrack_abc/vendor/CADTrack/lib/utils/tensor.py': '526729efede32e08be41db1827b0c15f32d5edaa590dcdf26eb4159bffbad135', 'cadtrack_abc/ported_modules.py': '5c038fcd2b346541a109fb79597f42122082a7d4def49c745957cb8515ce87bb', 'cadtrack_abc/pure_baseline_tracker.py': '26d4a94fafad25356017da2ebcfd0a1a99bfaa288df32a5c80f671a7d2553415', 'cadtrack_abc/baseline.py': 'c638fbef2ddf89ce1b2956023ae0e8c09a59bea81c057a40f062b2aff050555f', 'cadtrack_abc/model.py': '4dab198957f758e963e275f72c1a5df67c508b80bdf384d31cc211cfbf5114b2', 'cadtrack_abc/data.py': '7a608de3e43827e3ac6cfe5322bcfe3ce5d737deafda9b0d8645598b6e436c6d', 'cadtrack_abc/vendor/CADTrack/lib/models/layers/head.py': '958ee314ae200dd52355050dc1da8b46d5268da032e5fc1d0c6af99bab4c7431', 'cadtrack_abc/vendor/CADTrack/lib/models/cadtrack/vit_care.py': 'b9bd0a9b46c1a0694b26391df70c55fdc44ca5502ecd92427615b1ced7e2db5b', 'cadtrack_abc/vendor/CADTrack/lib/utils/box_ops.py': 'cee9d309a9c4a6c5f00fab575719bc7bf8d865127c91b10d4fb233996a111fbc', 'cadtrack_abc/vendor/CADTrack/experiments/cadtrack/cadtrack.yaml': 'eb85514008d97dd7266249dfd5c30c30c173ac939ef7d155a8576a9de3e6d188', 'cadtrack_abc/dense_pipeline.py': 'a849900cbe3c5ebdcdb59567a934cbd04b065b4d0b5f83dee74f8ebbe2ebebfb'}
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
payload={'driver.py': "from pathlib import Path\nfrom datetime import datetime, timezone\nimport json\nimport os\nimport subprocess\n\nproject=Path('/home/gaob/CADTrack-ABC')\nroot=project/'outputs/r2_dense_pair_20261015'\npython='/data/wangwj/envs/CADTrack/bin/python'\nqualification=project/'profile_output/r2_dense_m0_capacity128_20261015'\nqualified=json.loads((qualification/'state.json').read_text())\nassert qualified['status']=='DENSE_M0_COMPLETE' and qualified['chosen_common_batch']==128\nassert json.loads((qualification/'exit.json').read_text())['actual_controller_exit_code']==0\nbaseline234=project/'cadtrack_abc/results/pure_cad_rgbt234_20261015/full_report.json'\nbaseline210=project/'cadtrack_abc/results/three_benchmark_closure_20261015/pure_cad_rgbt210_1225/full_report.json'\nfor dataset,path in [('rgbt234',baseline234),('rgbt210',baseline210)]:\n    reference=json.loads(path.read_text())\n    assert reference['completed'] and reference['dataset']==dataset\n    assert reference['checkpoint']=='/home/gaob/CADTrack_ep0014.pth.tar'\nstate=dict(status='DENSE_H_HA_FULL_RUNS_RUNNING',controller_pid=os.getpid(),\n    started_utc=datetime.now(timezone.utc).isoformat(),jobs={},common_batch=128,\n    planned_epochs=20,planned_clips_per_variant=400000,planned_updates_per_variant=3140,\n    planned_query_observations_per_variant=3200000,full_ABC=False,goal_achieved=False)\n\n\ndef save():\n    (root/'pair_state.json').write_text(json.dumps(state,indent=2))\n\n\nsave()\njobs=[]\nfor label,gpu in [('H',2),('HA',3)]:\n    command=[python,'-m','cadtrack_abc.dense_pipeline','--config',str(project/'cadtrack_abc/config.json'),\n        '--output',str(root/label),'--m0-root',str(qualification),'--variant',label,'--gpu',str(gpu),\n        '--baseline-234',str(baseline234),'--baseline-210',str(baseline210)]\n    stream=(root/f'{label}_pipeline.log').open('w')\n    child=subprocess.Popen(command,cwd=project,stdout=stream,stderr=subprocess.STDOUT,\n        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))\n    jobs.append((label,child,stream))\n    state['jobs'][label]=dict(controller_pid=child.pid,physical_gpu=gpu,command=command,status='RUNNING')\n    save()\ncodes=[]\nfor label,child,stream in jobs:\n    code=child.wait()\n    stream.close()\n    codes.append(code)\n    state['jobs'][label].update(actual_exit_code=code,status='EXITED')\n    save()\nassert codes==[0,0],codes\nhistories={label:json.loads((root/label/'fit/history.json').read_text()) for label in ('H','HA')}\nassert len(histories['H'])==len(histories['HA'])==20\nfor left,right in zip(histories['H'],histories['HA']):\n    assert left['epoch']==right['epoch']\n    assert left['clips']==right['clips']==20000\n    assert left['clip_order_sha256']==right['clip_order_sha256']\nfor label in ('H','HA'):\n    receipt=json.loads((root/label/'pipeline_complete.json').read_text())\n    assert receipt['status']=='DENSE_FULL_TRAINING_AND_THREE_BENCHMARKS_COMPLETE'\n    assert receipt['full98_selection_complete'] and set(receipt['reports'])=={'lasher','rgbt234','rgbt210'}\n    state.setdefault('formal_results',{})[label]=receipt\nstate.update(status='DENSE_MATCHED_PAIR_FULL_TRAINING_AND_EVALUATION_COMPLETE',\n    completed_utc=datetime.now(timezone.utc).isoformat(),all20_actual_clip_orders_equal=True,\n    actual_exit_codes=codes,full_ABC=False,goal_achieved=False)\nsave()\n(root/'pair_complete.json').write_text(json.dumps(state,indent=2))\nprint(json.dumps(state),flush=True)\n", 'wrapper.py': "from pathlib import Path\nfrom datetime import datetime, timezone\nimport json\nimport subprocess\n\nroot=Path('/home/gaob/CADTrack-ABC/outputs/r2_dense_pair_20261015')\nwith (root/'pair_controller.log').open('w') as stream:\n    child=subprocess.run(['/data/wangwj/envs/CADTrack/bin/python',str(root/'driver.py')],\n        stdout=stream,stderr=subprocess.STDOUT)\n(root/'pair_exit.json').write_text(json.dumps(dict(actual_controller_exit_code=child.returncode,\n    completed_utc=datetime.now(timezone.utc).isoformat()),indent=2))\n"}
for name,source in payload.items():
    (output/name).write_text(source)
preflight=dict(observed_utc=datetime.now(timezone.utc).isoformat(),gpu_memory_and_utilization_csv=resources,
    home_free_bytes=free,output_budget_bytes=budget,data_free_bytes=shutil.disk_usage('/data').free,
    expected_sources=expected,environment=environment,environment_rebuilt=False,
    pretrained_sha256=hashlib.sha256(pretrained.read_bytes()).hexdigest(),
    qualification_state_sha256=hashlib.sha256((qualification/'state.json').read_bytes()).hexdigest(),
    source_publication_commit='e3a5f08214aa1480487e9620f4d1948e845366be',dataset_moved=False,temperature_power_queried=False)
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
