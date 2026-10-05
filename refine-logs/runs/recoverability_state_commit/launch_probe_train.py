"""Launch one reviewed TRAIN probe only on a completed native shard's GPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_state_commit'
BASE = Path('/data/gb/outputs/recoverability_geometry_commit_train_20261005')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['sanity', 'full'], required=True)
    p.add_argument('--gpu', type=int, choices=[0, 3], required=True)
    args = p.parse_args()
    gpu = args.gpu
    review = json.loads((FOLDER / 'probe_launch_source_review.json').read_text())
    assert review['status'] == 'PASS' and review['runtime_attested'] is False
    assert json.loads((FOLDER / 'probe_geometry_commit_source_review.json').read_text())['status'] == 'PASS_SOURCE_RECHECK'
    selection = json.loads(Path('/data/gb/setup/train_events_native_selection_20261005.json').read_text())
    assert selection['checkpoint'] == '/data/gb/outputs/recoverability_train_events_aug_budgeted_full_20261005/best.pth'
    assert selection['selected_best_epoch'] == 3 and selection['selected_at_cst'] < '2026-10-05T13:34'
    for dataset in ['lasher', 'rgbt234']:
        assert (Path('/data/gb/outputs/recoverability_train_events_native_full_20261005') /
                dataset / 'shards' / ('gpu' + str(gpu)) / 'inference_completed.txt').is_file()
    # Current main NN children are on GPUs1/2. Check actual GPU ownership, never temperature/power.
    ownership = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    device = next(line.split(', ') for line in devices.splitlines() if int(line.split(', ')[0]) == gpu)
    assert float(device[2]) > 20000, devices
    current = [line for line in ownership.splitlines() if line.split(', ')[0] == device[1]]
    assert not any('python' in line.lower() for line in current), current
    BASE.mkdir(parents=True, exist_ok=True)
    if args.stage == 'sanity':
        assert gpu == 0
        name = 'sanity'
        jobs = FOLDER / 'probe_train_sanity_jobs.json'
    else:
        sanity = json.loads((BASE / 'sanity/events.json').read_text())
        assert (BASE / 'sanity/COMPLETE').is_file() and sanity['queries'] == 1
        assert sanity['status'] == 'COMPLETE_MATCHED_QUERY_COMMIT_ROLLOUTS'
        assert {c['name'] for c in sanity['results'][0]['controls']} == {'raw', 'pause', 'geometry_C1', 'pause_geometry_C1'}
        assert len({c['query_output_iou'] for c in sanity['results'][0]['controls']}) == 1
        name = 'gpu' + str(gpu)
        jobs = FOLDER / ('probe_train_' + name + '_jobs.json')
    output = BASE / name
    receipt = BASE / (name + '_launch.json')
    assert not output.exists() and not receipt.exists()
    env = os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu)}
    command = ['bash', str(FOLDER / 'run_probe_train.sh'), str(gpu), str(jobs), str(output)]
    with (BASE / (name + '.log')).open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'stage': args.stage, 'gpu': gpu, 'pid': child.pid, 'command': command,
              'launched_at_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'model': selection['checkpoint'], 'native_test_selection': False, 'jobs': str(jobs),
              'output': str(output), 'log': str(BASE / (name + '.log')),
              'completed_native_shard_gpu': gpu, 'prior_gpu_compute_apps': current}
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
