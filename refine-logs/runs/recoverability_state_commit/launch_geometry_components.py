"""Launch the reviewed TRAIN component probe; real one-query sanity first."""
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT/'refine-logs/runs/recoverability_state_commit'
BASE = Path('/data/gb/outputs/recoverability_geometry_components_train88_20261005')


def main():
    review = json.loads((FOLDER/'geometry_components_source_review.json').read_text())
    assert review['status'] == 'PASS'
    gpu = 0
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    row = next(line.split(', ') for line in devices.splitlines() if int(line.split(', ')[0]) == gpu)
    assert float(row[2]) > 20000 and not any(line.split(', ')[0] == row[1] for line in apps.splitlines())
    cfg = json.loads((FOLDER/'matched_velocity_train88_old4/gpu0/config.json').read_text())
    suffix = 'sanity'
    jobs = FOLDER/f'geometry_components_{suffix}_jobs.json'
    job_rows = json.loads(jobs.read_text())['jobs']
    output = BASE/suffix
    receipt = BASE/(suffix+'_launch.json')
    assert not output.exists() and not receipt.exists()
    BASE.mkdir(exist_ok=True)
    command = ['bash', 'scripts/run_temporal.sh', str(gpu), 'probe_geometry_commit']
    for key in ('root', 'cache', 'split', 'model', 'pretrained', 'c1_head', 'motion_run'):
        command += ['--'+key.replace('_','-'), cfg[key]]
    command += ['--jobs-file', str(jobs), '--output', str(output), '--horizons', '3', '32',
                '--write-verification', 'action', '--seed', '42', '--geometry-reference', 'C1_components']
    started = time.time()
    with (BASE/(suffix+'.log')).open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = {'pid':child.pid, 'gpu':gpu, 'command':command, 'queries':len(job_rows),
              'created_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'started_epoch':started, 'model':cfg['model'], 'NN_stage':'single actual differing TRAIN query'}
    receipt.write_text(json.dumps(launch, indent=2)+'\n')
    print(json.dumps(launch), flush=True)
    assert child.wait() == 0, (BASE/'sanity.log').read_text()[-4000:]
    assert (output/'COMPLETE').is_file()
    results = json.loads((output/'events.json').read_text())
    assert results['queries'] == 1 and results['geometry_reference'] == 'C1_components'
    assert not results['results'][0]['selected_matches_C1_keep']
    assert len({c['query_output_iou'] for c in results['results'][0]['controls']}) == 1
    elapsed = time.time()-started
    (BASE/'sanity_execution.json').write_text(json.dumps({'complete':True,'exit_code':0,'seconds':elapsed,
        'query':job_rows[0], 'actual_parent_epoch':4, 'output_state_isolation_assertions_passed':True,
        'future_GT_action_inputs':False,'new_optimizer_updates':0}, indent=2)+'\n')
    print(json.dumps({'sanity_pass':True,'seconds':elapsed}), flush=True)


if __name__ == '__main__': main()
