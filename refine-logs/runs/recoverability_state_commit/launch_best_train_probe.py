"""Fixed old4 TRAIN recheck: same48 predeclared states, four free GPUs."""
import argparse
import datetime
import json
from pathlib import Path
import os
import subprocess

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_state_commit'
BASE = Path('/data/gb/outputs/recoverability_geometry_commit_best_train_20261005')
ACTOR = '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['sanity', 'full'], required=True)
    p.add_argument('--gpu', type=int, choices=[0, 1, 2, 3], required=True)
    a = p.parse_args()
    review = json.loads((FOLDER / 'best_probe_launch_source_review.json').read_text())
    assert review['status'] == 'PASS' and review['runtime_attested'] is False
    original = json.loads(Path('/data/gb/setup/train_events_complete_evaluation_progress_20261005.json').read_text())
    assert original['stage'] == 'COMPLETE420_TRAINING_FULL98_AND_BOTH_FULL_NATIVE_REPORTS_READY'
    for part in ['sanity', 'gpu0', 'gpu3']:
        assert (Path('/data/gb/outputs/recoverability_geometry_commit_train_20261005') / part / 'COMPLETE').is_file()
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    device = next(line.split(', ') for line in devices.splitlines() if int(line.split(', ')[0]) == a.gpu)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
    owned = [line for line in apps.splitlines() if line.split(', ')[0] == device[1]]
    assert float(device[2]) > 20000 and not any('python' in line.lower() for line in owned)
    BASE.mkdir(parents=True, exist_ok=True)
    if a.stage == 'sanity':
        assert a.gpu == 0
        name, jobs = 'sanity', FOLDER / 'best_probe_sanity_jobs.json'
    else:
        assert (BASE / 'sanity/COMPLETE').is_file()
        d = json.loads((BASE / 'sanity/events.json').read_text())
        assert d['queries'] == 1 and d['status'] == 'COMPLETE_MATCHED_QUERY_COMMIT_ROLLOUTS'
        assert len({c['query_output_iou'] for c in d['results'][0]['controls']}) == 1
        name, jobs = 'gpu' + str(a.gpu), FOLDER / ('best_probe_gpu' + str(a.gpu) + '_jobs.json')
    output, receipt = BASE / name, BASE / (name + '_launch.json')
    assert not output.exists() and not receipt.exists()
    command = ['bash', str(ROOT / 'scripts/run_temporal.sh'), str(a.gpu), 'probe_geometry_commit',
               '--root', '/data/wangwj/dataset/LasHeR',
               '--cache', 'trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np',
               '--split', '/data/gb/outputs/c1_initial_seed42/split.json', '--jobs-file', str(jobs),
               '--model', ACTOR, '--pretrained', '/data/gb/GOLA/pretrained_models/gola_b224.bin',
               '--c1-head', '/data/gb/outputs/c1_initial_seed42/best.pth', '--motion-run', '/data/gb/outputs/abc_joint_v1_seed42',
               '--write-verification', 'action', '--horizons', '3', '32', '--seed', '42', '--output', str(output)]
    with (BASE / (name + '.log')).open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=os.environ | {'CUDA_VISIBLE_DEVICES': str(a.gpu)},
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': child.pid, 'gpu': a.gpu, 'stage': a.stage, 'actor': ACTOR, 'jobs': str(jobs),
              'output': str(output), 'command': command, 'formal_scores_used_to_choose_states': False,
              'launched_at_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
