"""Run the already qualified pure-CAD protocol after main R1 frees GPU2."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess

project = Path('/home/gaob/CADTrack-ABC')
python = '/data/wangwj/envs/CADTrack/bin/python'
main = json.loads((project / 'outputs/r1_main_reports_20261015/benchmark_state.json').read_text())
assert main['status'] == 'MAIN_R1_THREE_BENCHMARKS_COMPLETE'
assert set(main['reports']) == {'lasher', 'rgbt234', 'rgbt210'}
controller = subprocess.run(['ps', '-p', str(main['controller_pid']), '-o', 'args='],
    capture_output=True, text=True)
assert controller.returncode != 0
memory = subprocess.run(['nvidia-smi', '-i', '2', '--query-gpu=memory.used',
    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
assert int(memory.strip()) < 500
checkpoint = Path('/home/gaob/CADTrack_ep0014.pth.tar')
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == 'cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731'
inventory = json.loads((project / 'dataset/rgbt210_complete.json').read_text())
assert inventory['completed'] and inventory['sequences'] == 210 and inventory['frames'] == 104706
root = Path(inventory['root'])
output = project / 'outputs/pure_cad_rgbt210_20261015'
output.mkdir(exist_ok=False)
state_path = output / 'pipeline_state.json'
state = dict(status='PURE_RGBT210_INFERENCE_RUNNING', controller_pid=os.getpid(),
    started_utc=datetime.now(timezone.utc).isoformat(), dataset='rgbt210',
    checkpoint=str(checkpoint), physical_gpu=2, max_workers_per_gpu=2,
    jobs=[], baseline_lasher_rerun=False, model_or_environment_modified=False)
jobs, streams = [], []
for shard in (0, 1):
    command = [python, '-m', 'cadtrack_abc.infer_baseline', '--checkpoint', str(checkpoint),
        '--dataset', 'rgbt210', '--root', str(root), '--output', str(output / 'predictions'),
        '--shard', str(shard)]
    stream = (output / f'infer_shard{shard}.log').open('w')
    streams.append(stream)
    child = subprocess.Popen(command, cwd=project,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'),
        stdout=stream, stderr=subprocess.STDOUT)
    jobs.append(child)
    state['jobs'].append(dict(pid=child.pid, command=command, physical_gpu=2))
state_path.write_text(json.dumps(state, indent=2))
print(json.dumps(dict(event='PURE_RGBT210_STARTED', jobs=state['jobs'])), flush=True)
codes = [child.wait() for child in jobs]
for stream in streams:
    stream.close()
state.update(inference_exit_codes=codes, status='INFERENCE_EXITED_ZERO' if codes == [0, 0] else 'FAILED')
state_path.write_text(json.dumps(state, indent=2))
assert codes == [0, 0]
receipts = [json.loads((output / f'predictions/complete_shard{shard}.json').read_text()) for shard in (0, 1)]
assert all(row['completed'] and row['policy_family'] == 'pure_cad' and row['epoch'] == 14
    and row['checkpoint'] == str(checkpoint) for row in receipts)
assert sum(row['sequences'] for row in receipts) == 210
assert sum(row['frames'] for row in receipts) == 104706
with (output / 'score_cpu.log').open('w') as stream:
    score = subprocess.run([python, '-m', 'cadtrack_abc.score', '--dataset', 'rgbt210',
        '--root', str(root), '--predictions', str(output / 'predictions'),
        '--output', str(output / 'reports')], cwd=project,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'),
        stdout=stream, stderr=subprocess.STDOUT, check=True)
report = json.loads((output / 'reports/full_report.json').read_text())
assert report['completed'] and report['checkpoint'] == str(checkpoint)
assert report['dataset'] == 'rgbt210' and report['sequences'] == 210 and report['frames'] == 104706
assert set(report['overall_metrics_percent']) == {'PR', 'SR'}
state.update(status='PURE_RGBT210_COMPLETE', completed=True, score_exit_code=score.returncode,
    completed_utc=datetime.now(timezone.utc).isoformat(), model_role='pure_cad_baseline',
    full_report=str(output / 'reports/full_report.json'), metrics=report['overall_metrics_percent'])
state_path.write_text(json.dumps(state, indent=2))
print(json.dumps(state), flush=True)
