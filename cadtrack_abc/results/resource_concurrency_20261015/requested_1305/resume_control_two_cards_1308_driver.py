"""Resume only the two unstarted formal datasets after a verified SSH interruption."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess

project = Path('/home/gaob/CADTrack-ABC')
python = '/data/wangwj/envs/CADTrack/bin/python'
root = project / 'outputs/r1_control_reports_20261015'
checkpoint = project / 'outputs/r1_matched_fits_20261015/fit_future_actions/best.pth'
config = project / 'cadtrack_abc/config.json'
digest = 'cffca0302460b60f505c2a09a5b651b10f1d1e2211178f68307a4d708080bf62'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
original = json.loads((root / 'benchmark_state.json').read_text())
assert subprocess.run(['ps','-p',str(original['controller_pid']),'-o','args='], capture_output=True).returncode != 0
assert original['phases']['infer_selected_lasher']['return_codes'] == [0,0]
assert original['phases']['infer_selected_lasher']['status'] == 'VERIFIED_COMPLETE'
selection_path = project / 'outputs/r1_matched_fits_20261015/fit_future_actions/selection.json'
selection = json.loads(selection_path.read_text())
assert selection['best_checkpoint'] == selection['trained_best_checkpoint'] == str(checkpoint)
assert selection['selected']['epoch'] == 20
lasher = json.loads((root/'selected/lasher_report/full_report.json').read_text())
assert lasher['completed'] and lasher['checkpoint'] == str(checkpoint) and lasher['epoch'] == 20
assert lasher['sequences'] == 245 and lasher['frames'] == 220703
for shard in (0,1):
    receipt = json.loads((root/f'selected/lasher_predictions/complete_shard{shard}.json').read_text())
    assert receipt['completed'] and receipt['policy_family'] == 'future_actions'
    assert receipt['checkpoint'] == str(checkpoint)
memory = subprocess.run(['nvidia-smi','-i','2,3','--query-gpu=index,memory.used',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
assert all(int(line.split(',')[1]) < 500 for line in memory.splitlines())
state_path = root/'recovery_1308_state.json'
assert not state_path.exists()
state = dict(status='RUNNING', controller_pid=os.getpid(),
    started_utc=datetime.now(timezone.utc).isoformat(),
    original_controller_pid=original['controller_pid'], original_controller_stopped=True,
    interruption='Original SSH transport returned255/Connection reset; historical benchmark_state remains unchanged.',
    policy='future_actions', checkpoint=str(checkpoint), checkpoint_sha256=digest, epoch=20,
    physical_gpus=[2,3], max_workers_per_card=2, assignment={'rgbt234':2,'rgbt210':3},
    new_training_started=False, completed_lasher_inference_reused=True,
    completed_lasher_score_reused=True, original_lasher_score_exit_code='Unavailable after transport interruption; full report and score stdout present.',
    reports={'selected':{'lasher':dict(report=str(root/'selected/lasher_report/full_report.json'),
        checkpoint=str(checkpoint),epoch=20,metrics=lasher['overall_metrics_percent'])}}, phases={})

def save():
    state_path.write_text(json.dumps(state, indent=2))

save()
datasets = [('rgbt234',2,project/'dataset/rgbt234',234,116649),
    ('rgbt210',3,project/'dataset/rgbt210/RGBT210',210,104706)]
jobs, streams = [], []
for dataset, gpu, data_root, count, frames in datasets:
    predictions = root/f'selected/{dataset}_predictions'
    assert not predictions.exists()
    phase = dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),jobs=[])
    state['phases']['infer_'+dataset] = phase
    for shard in (0,1):
        command = [python,'-m','cadtrack_abc.infer','--config',str(config),'--checkpoint',str(checkpoint),
            '--policy','future_actions','--dataset',dataset,'--root',str(data_root),
            '--output',str(predictions),'--shard',str(shard),'--shards','2']
        stream = (root/f'recovery_1308_infer_{dataset}_shard{shard}.log').open('w')
        streams.append(stream)
        child = subprocess.Popen(command,cwd=project,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),
            stdout=stream,stderr=subprocess.STDOUT)
        phase['jobs'].append(dict(pid=child.pid,physical_gpu=gpu,shard=shard,command=command))
        jobs.append((dataset,shard,child))
    save()
codes = {}
for dataset, shard, child in jobs:
    code = child.wait()
    codes.setdefault(dataset,[]).append(code)
    state['phases']['infer_'+dataset].setdefault('actual_exit_codes',{})[str(shard)] = code
    save()
for stream in streams:
    stream.close()
assert all(code == 0 for group in codes.values() for code in group), codes
for dataset, gpu, data_root, count, frames in datasets:
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    predictions = root/f'selected/{dataset}_predictions'
    receipts = [json.loads((predictions/f'complete_shard{shard}.json').read_text()) for shard in (0,1)]
    assert all(r['completed'] and r['dataset'] == dataset and r['checkpoint'] == str(checkpoint)
        and r['policy_family'] == 'future_actions' and r['epoch'] == 20 for r in receipts)
    assert sum(r['sequences'] for r in receipts) == count and sum(r['frames'] for r in receipts) == frames
    state['phases']['infer_'+dataset].update(status='VERIFIED_COMPLETE',completed_utc=datetime.now(timezone.utc).isoformat(),sequences=count,frames=frames)
    report_root = root/f'selected/{dataset}_report'
    command = [python,'-m','cadtrack_abc.score','--dataset',dataset,'--root',str(data_root),
        '--predictions',str(predictions),'--output',str(report_root),'--lasher-target-gain','6']
    state['phases']['score_'+dataset] = dict(status='RUNNING',command=command)
    save()
    with (root/f'recovery_1308_score_{dataset}.log').open('w') as stream:
        completed = subprocess.run(command,cwd=project,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'),
            stdout=stream,stderr=subprocess.STDOUT)
    state['phases']['score_'+dataset].update(actual_exit_code=completed.returncode)
    save()
    assert completed.returncode == 0
    report = json.loads((report_root/'full_report.json').read_text())
    assert report['completed'] and report['dataset'] == dataset and report['checkpoint'] == str(checkpoint)
    assert report['epoch'] == 20 and report['sequences'] == count and report['frames'] == frames
    assert set(report['overall_metrics_percent']) == ({'MPR','MSR'} if dataset=='rgbt234' else {'PR','SR'})
    state['reports']['selected'][dataset] = dict(report=str(report_root/'full_report.json'),
        checkpoint=str(checkpoint),epoch=20,metrics=report['overall_metrics_percent'])
    state['phases']['score_'+dataset]['status'] = 'VERIFIED_COMPLETE'
    save()
selection.update(formal_metrics_complete=True,formal_reports=state['reports'])
selection_path.write_text(json.dumps(selection,indent=2))
state.update(status='CONTROL_R1_THREE_BENCHMARKS_COMPLETE',
    completed_utc=datetime.now(timezone.utc).isoformat(),trained_checkpoint_formal_metrics_complete=True)
save()
print(json.dumps(dict(status=state['status'],reports=state['reports'])),flush=True)
