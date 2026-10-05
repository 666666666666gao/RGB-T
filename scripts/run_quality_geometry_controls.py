"""Sanity then four-GPU matched TRAIN-root developer query consequences."""
import argparse
import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path('/data/gb/GOLA')
PYTHON = '/data/gb/envs/gola/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    assert root == Path('/data/gb/outputs/quality_geometry_query_controls_20261006') and not root.exists()
    native = Path('/data/gb/outputs/recoverability_search_gain_native_full_20261006')
    assert json.loads((native / 'progress.json').read_text())['stage'] == 'COMPLETE_BOTH_NATIVE_FULL_FIVE_METRICS_ACTUAL_GT'
    launch = json.loads(Path('/data/gb/setup/search_gain_native_launch_20261006.json').read_text())
    proc = Path('/proc') / str(launch['pid']) / 'stat'
    assert not proc.exists() or proc.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    review = json.loads(Path(args.review).read_text())
    assert review['status'] == 'PASS' and review['scope'] == 'DRIVER_SAMPLER_CONTROLLER_SOURCE'
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True)
    values = [tuple(int(x.strip()) for x in line.split(',')) for line in cards.splitlines()]
    assert [r[0] for r in values] == [0, 1, 2, 3] and all(r[1] < 1024 and r[2] == 0 for r in values)
    assert shutil.disk_usage('/data/gb').free > 1024**3
    jobs = json.loads(Path(args.jobs).read_text())['jobs']
    assert len(jobs) == 12 and len({(j['sequence'], j['event_id']) for j in jobs}) == 12
    root.mkdir()

    def record(stage, **extra):
        event = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(), **extra}
        (root / 'progress.json').write_text(json.dumps(event, indent=2))
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)

    common = ['--root', '/data/wangwj/dataset/LasHeR', '--cache',
              'trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np',
              '--split', '/data/gb/outputs/c1_initial_seed42/split.json', '--model',
              '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth',
              '--pretrained', '/data/gb/GOLA/pretrained_models/gola_b224.bin',
              '--c1-head', '/data/gb/outputs/c1_initial_seed42/best.pth',
              '--motion-run', '/data/gb/outputs/abc_joint_v1_seed42']
    groups = [jobs[gpu::4] for gpu in range(4)]
    for stage in ('sanity', 'full'):
        children = []
        for gpu, group in enumerate(groups):
            selected = group[:1] if stage == 'sanity' else group
            jobfile = root / (stage + '_gpu' + str(gpu) + '_jobs.json')
            jobfile.write_text(json.dumps({'jobs': selected}, indent=2))
            out = root / stage / ('gpu' + str(gpu))
            log = (root / (stage + '_gpu' + str(gpu) + '.log')).open('w')
            command = ['bash', 'scripts/run_temporal.sh', str(gpu), 'probe_quality_geometry_rollouts',
                       *common, '--jobs-file', str(jobfile), '--output', str(out),
                       '--horizons', *(['3'] if stage == 'sanity' else ['3', '32'])]
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log, gpu, out, len(selected)))
        record(stage.upper(), children=[{'gpu': g, 'pid': p.pid} for p, _, g, _, _ in children])
        exits = []
        for child, log, gpu, out, count in children:
            code = child.wait()
            log.close()
            record(stage.upper() + '_WORKER_ENDED', gpu=gpu, exit_code=code)
            exits.append((gpu, code))
        assert all(code == 0 for _, code in exits), f'{stage} worker exits {exits}; original logs retained, no retry'
        for _, _, gpu, out, count in children:
            done = json.loads((out / 'events.json').read_text())
            assert (out / 'COMPLETE').is_file() and done['status'] == 'COMPLETE_MATCHED_QUERY_CONTROLS'
            assert done['event_count'] == count and done['raw_base_exact_all_frames']
        record(stage.upper() + '_PASS')
    results = []
    for gpu in range(4):
        results += json.loads((root / 'full' / ('gpu' + str(gpu)) / 'events.json').read_text())['results']
    assert len(results) == 12 and len({(r['sequence'], r['event_id']) for r in results}) == 12
    record('COMPLETE_ALL12_MATCHED_QUERY_EVENTS_RAW_PARITY_PASS', results=results,
           new_training_steps=0, new_weights=0, native_metric_claim=False)


if __name__ == '__main__':
    main()
