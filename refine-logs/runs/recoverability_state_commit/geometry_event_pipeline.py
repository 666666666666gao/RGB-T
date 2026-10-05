"""One fixed e41 continuation control; use all four GPUs on disjoint full videos."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT/'refine-logs/runs/recoverability_state_commit'
NATIVE = Path('/data/gb/outputs/recoverability_geometry_commit_native_20261005')
FIT = Path('/data/gb/outputs/recoverability_geometry_commit_fit_20261005/gpu1')
BASE = Path('/data/gb/outputs/recoverability_geometry_event_control_20261005')
TRAIN = '/data/wangwj/dataset/LasHeR/traingset'
SPLIT = '/data/gb/outputs/c1_initial_seed42/split.json'
PYTHON = '/data/gb/envs/gola/bin/python'
read = lambda path: json.loads(Path(path).read_text())


def gate():
    assert read(FOLDER/'geometry_event_continuation_source_review.json')['status'] == 'PASS'
    selection = read(FOLDER/'selected_geometry_native_checkpoint.json')
    assert selection['commit_model'] == str(FIT/'best.pth') and selection['commit_epoch'] == 41
    for i in range(4):
        launch = read(NATIVE/f'gpu{i}_launch.json')
        proc = Path('/proc')/str(launch['pid'])
        assert not proc.exists() or (proc/'stat').read_text().split(') ', 1)[1].split()[0] == 'Z'
        for dataset in ('lasher', 'rgbt234'):
            assert (NATIVE/dataset/'shards'/f'gpu{i}'/'inference_completed.txt').is_file()
    return selection


def sanity():
    selection = gate()
    assert not (BASE/'sanity').exists()
    subprocess.run(['bash', 'scripts/run_temporal.sh', '0', 'check_geometry_event_sanity',
                    '--parent', selection['parent'], '--commit-model', selection['commit_model'],
                    '--sequence-root', TRAIN+'/manatwhiteright', '--output', str(BASE/'sanity')], cwd=ROOT, check=True)


def prepare():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    gate()
    accepted = read(BASE/'sanity/completion.json')
    assert accepted['complete'] and accepted['complete_event_windows'] > 0 and accepted['matched_old4_frames'] > 0
    done = read(FIT/'continuous/predictions/inference_completion.json')
    assert done['completed'] and done['sequences'] == 98 and done['frames'] == 49418
    names = sorted(read(SPLIT)['validation'])
    assert len(names) == 98 and {r['sequence'] for r in done['records']} == set(names)
    lengths = {r['sequence']: r['frames'] for r in done['records']}
    cumulative = np.concatenate(([0], np.cumsum([lengths[n] for n in names])))
    cuts = [0]
    for k in range(1, 4):
        cuts.append(min(range(cuts[-1]+1, 98-(4-k)+1), key=lambda j: abs(cumulative[j]-49418*k/4)))
    cuts.append(98)
    shards = [{'gpu': i, 'offset': cuts[i], 'count': cuts[i+1]-cuts[i],
               'frames': int(cumulative[cuts[i+1]]-cumulative[cuts[i]]), 'sequences': names[cuts[i]:cuts[i+1]]} for i in range(4)]
    path = BASE/'prepared_shards.json'
    assert not path.exists()
    path.write_text(json.dumps({'sequences': 98, 'frames': 49418, 'shards': shards, 'GT_used_for_partition': False}, indent=2)+'\n')


def launch(gpu):
    selection = gate()
    assert read(BASE/'sanity/completion.json')['complete']
    shard = read(BASE/'prepared_shards.json')['shards'][gpu]
    output = BASE/'shards'/f'gpu{gpu}'/'predictions'
    receipt = BASE/f'gpu{gpu}_launch.json'
    assert not output.exists() and not receipt.exists()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    row = next(line.split(', ') for line in devices.splitlines() if int(line.split(', ')[0]) == gpu)
    assert float(row[2]) > 20000 and not any(line.split(', ')[0] == row[1] and 'python' in line.lower() for line in apps.splitlines())
    command = ['bash', 'scripts/run_temporal.sh', str(gpu), 'evaluate_recoverability', '--dataset', 'lasher', '--root', TRAIN,
               '--model', selection['parent'], '--commit-model', selection['commit_model'], '--commit-continuation', 'teacher_horizon',
               '--pretrained', '/data/gb/GOLA/pretrained_models/gola_b224.bin', '--write-verification', 'action', '--seed', '42',
               '--validation-split', SPLIT, '--sequence-offset', str(shard['offset']), '--limit-sequences', str(shard['count']),
               '--max-frames', '0', '--output', str(output)]
    with (BASE/f'gpu{gpu}.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt.write_text(json.dumps({'pid': child.pid, 'gpu': gpu, 'command': command, 'parent': selection['parent'],
                                  'commit_model': selection['commit_model'], 'commit_epoch': 41, 'continuation_horizon': 3}, indent=2)+'\n')
    print(json.dumps({'gpu': gpu, 'pid': child.pid, 'frames': shard['frames']}), flush=True)


def merge_score():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    selection = gate()
    prepared = read(BASE/'prepared_shards.json')
    rows, configs, completions, files, times = [], [], [], [], []
    for shard in prepared['shards']:
        receipt = read(BASE/f"gpu{shard['gpu']}_launch.json")
        proc = Path('/proc')/str(receipt['pid'])
        assert not proc.exists() or (proc/'stat').read_text().split(') ', 1)[1].split()[0] == 'Z'
        source = BASE/'shards'/f"gpu{shard['gpu']}"/'predictions'
        cfg, done = read(source/'inference_config.json'), read(source/'inference_completion.json')
        assert done['completed'] and done['sequences'] == cfg['limit_sequences'] == shard['count'] and done['frames'] == shard['frames']
        assert cfg['sequence_offset'] == shard['offset'] and cfg['max_frames'] == 0 and cfg['validation_split'] == SPLIT and cfg['root'] == TRAIN
        assert cfg['model'] == selection['parent'] and cfg['commit_model'] == selection['commit_model'] and cfg['commit_head_epoch'] == 41
        assert cfg['commit_continuation'] == 'teacher_horizon' and cfg['commit_continuation_horizon'] == 3 and cfg['write_verification'] == 'action'
        assert [r['sequence'] for r in done['records']] == shard['sequences']
        assert {p.stem for p in source.glob('*.txt')} == set(shard['sequences'])
        for r in done['records']:
            name = r['sequence']
            prediction = np.loadtxt(source/(name+'.txt'), ndmin=2)
            latency = np.load(source/(name+'_latency.npy'), allow_pickle=False)
            assert prediction.shape == (r['frames'], 4) and latency.shape == (r['frames']-1,)
            assert np.isfinite(prediction).all() and np.isfinite(latency).all() and (latency > 0).all()
            with np.load(source/(name+'_recoverability_decisions.npz'), allow_pickle=False) as timeline:
                assert len(timeline['old4_continuation']) == r['frames']-1
                assert int(timeline['old4_continuation'].sum()) == r['stats']['old4_continuation_frames']
            files += [source/(name+s) for s in ('.txt', '_latency.npy', '_recoverability_decisions.npz')]
            times.append(latency)
        rows += done['records']; configs.append(cfg); completions.append(done)
    assert len(rows) == 98 and len({r['sequence'] for r in rows}) == 98 and sum(r['frames'] for r in rows) == 49418
    out = BASE/'predictions'; assert not out.exists(); out.mkdir()
    for p in files: os.link(p, out/p.name)
    cfg = dict(configs[0], output=str(out), sequence_offset=0, limit_sequences=0, full_shard_union_verified=True)
    (out/'inference_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    latency = np.concatenate(times)
    for i, r in enumerate(rows, 1): r['sequence_index'], r['sequences'] = i, 98
    done = {'completed': True, 'smoke_only': True, 'sequences': 98, 'frames': 49418, 'records': rows,
            'fps_including_decode_crop_update': len(latency)/latency.sum(), 'official_accuracy': None,
            'peak_cuda_allocated_mib': max(c['peak_cuda_allocated_mib'] for c in completions)}
    (out/'inference_completion.json').write_text(json.dumps(done, indent=2)+'\n')
    subprocess.run([PYTHON, '-u', str(ROOT/'refine-logs/runs/recoverability_train_events/score_full98_cpu.py'),
                    '--predictions', str(out), '--model', selection['parent']], cwd=ROOT,
                   env=os.environ | {'PYTHONPATH': str(ROOT)}, check=True)
    event, frame = read(BASE/'full98_selection_score.json'), read(FIT/'continuous/full98_selection_score.json')
    assert event['actual_TRAIN_GT_scored'] and frame['actual_TRAIN_GT_scored']
    before = {r['sequence']: r['mean_iou'] for r in frame['per_sequence']}
    delta = np.array([r['mean_iou']-before[r['sequence']] for r in event['per_sequence']]) * 100
    draws = np.random.default_rng(42).integers(98, size=(5000, 98))
    result = {'complete': True, 'continuation_horizon': 3, 'commit_epoch': 41, 'event_mean_iou': event['sequence_mean_iou'],
              'frame_policy_mean_iou': frame['sequence_mean_iou'], 'paired_mean_delta_pp': float(delta.mean()),
              'paired_95_interval_pp': np.percentile(delta[draws].mean(1), [2.5, 97.5]).tolist(),
              'improved_sequences': int((delta > 0).sum()), 'worsened_sequences': int((delta < 0).sum()), 'tied_sequences': int((delta == 0).sum()),
              'actual_TRAIN_GT_scored': True, 'official_native_metrics': None, 'scope': 'Fixed weights and different causal continuation only;98 reused developer sequences.'}
    (BASE/'event_vs_frame_control.json').write_text(json.dumps(result, indent=2)+'\n')
    (BASE/'COMPLETE').write_text('Actual98/49418 causal continuation control scored\n')
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['sanity', 'prepare', 'launch', 'merge_score']); p.add_argument('--gpu', type=int, choices=range(4))
    a = p.parse_args()
    if a.stage == 'sanity': sanity()
    elif a.stage == 'prepare': prepare()
    elif a.stage == 'launch': assert a.gpu is not None; launch(a.gpu)
    else: merge_score()


if __name__ == '__main__': main()
