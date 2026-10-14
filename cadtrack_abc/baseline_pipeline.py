"""Two supervised pure-CAD inference shards followed by full official scoring."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=['rgbt210', 'rgbt234'], required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    jobs = []
    for shard, gpu in enumerate([2, 3]):
        command = [sys.executable, '-m', 'cadtrack_abc.infer_baseline', '--checkpoint', args.checkpoint,
            '--dataset', args.dataset, '--root', args.root,
            '--output', str(output / 'predictions'), '--shard', str(shard)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with (output / f'infer_shard{shard}.log').open('w') as stream:
            child = subprocess.Popen(command, cwd=project, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT)
        jobs.append(dict(child=child, pid=child.pid, physical_gpu=gpu, command=command))
    (output / 'launch.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
        dataset=args.dataset, checkpoint=args.checkpoint,
        jobs=[{key: value for key, value in job.items() if key != 'child'} for job in jobs]), indent=2))
    exits = [job['child'].wait() for job in jobs]
    (output / 'child_exits.json').write_text(json.dumps(dict(exit_codes=exits,
        observed_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    assert exits == [0, 0], exits
    command = [sys.executable, '-m', 'cadtrack_abc.score', '--dataset', args.dataset,
        '--root', args.root, '--predictions', str(output / 'predictions'),
        '--output', str(output / 'reports')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (output / 'score_cpu.log').open('w') as stream:
        score = subprocess.run(command, cwd=project, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
    report_path = output / 'reports/full_report.json'
    report = json.loads(report_path.read_text())
    assert report['completed'] and report['checkpoint'] == args.checkpoint
    report['model_role'] = 'pure_cad_baseline'
    report['baseline_source'] = 'Supplied CAD epoch14 module-free checkpoint with the original CAD test policy; actual dataset GT.'
    report_path.write_text(json.dumps(report, indent=2))
    complete = dict(completed=True, dataset=args.dataset, checkpoint=args.checkpoint,
        epoch=report['epoch'], model_role='pure_cad_baseline', inference_exit_codes=exits,
        score_exit_code=score.returncode, full_report=str(report_path),
        sequences=report['sequences'], frames=report['frames'],
        overall_metrics_percent=report['overall_metrics_percent'],
        observed_utc=datetime.now(timezone.utc).isoformat())
    (output / 'pipeline_complete.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
