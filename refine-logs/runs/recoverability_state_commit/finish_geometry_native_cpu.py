"""Wait near the four workers' estimated end, then run the reviewed CPU stages."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_state_commit'
BASE = Path('/data/gb/outputs/recoverability_geometry_commit_native_20261005')
PYTHON = '/data/gb/envs/gola/bin/python'


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    review = json.loads((FOLDER / 'geometry_native_finish_source_review.json').read_text())
    assert review['status'] == 'PASS'
    launches = [json.loads((BASE / f'gpu{i}_launch.json').read_text()) for i in range(4)]
    deadline = max(datetime.datetime.fromisoformat(r['launched_at_cst']).timestamp() for r in launches) + 45 * 60
    time.sleep(max(0, deadline - time.time()))
    while True:
        live = []
        for r in launches:
            path = Path('/proc') / str(r['pid'])
            if path.exists():
                state = (path / 'stat').read_text().split(') ', 1)[1].split()[0]
                if state != 'Z':
                    live.append(r['pid'])
        markers = [BASE / dataset / 'shards' / f'gpu{i}' / 'inference_completed.txt'
                   for dataset in ('lasher', 'rgbt234') for i in range(4)]
        print(json.dumps({'observed_at_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                          'live_original_workers': live, 'completed_shards': sum(p.is_file() for p in markers)}), flush=True)
        if not live:
            assert all(p.is_file() for p in markers), 'A native worker ended without its original completion marker; inspect gpu*.log. No rerun.'
            break
        time.sleep(240)
    for dataset in ('lasher', 'rgbt234'):
        subprocess.run([PYTHON, '-u', str(FOLDER / 'native_pipeline.py'), 'merge', '--dataset', dataset], cwd=ROOT, check=True)
        subprocess.run([PYTHON, '-u', str(FOLDER / 'native_pipeline.py'), 'report', '--dataset', dataset], cwd=ROOT, check=True)
        subprocess.run([PYTHON, '-u', str(FOLDER / 'audit_geometry_native_cpu.py'), '--dataset', dataset], cwd=ROOT, check=True)
    subprocess.run([PYTHON, '-u', str(FOLDER / 'native_pipeline.py'), 'complete'], cwd=ROOT, check=True)
    (BASE / 'cpu_finish_completed.txt').write_text('Reviewed native stages, actual-GT arithmetic and complete report finished\n')


if __name__ == '__main__':
    main()
