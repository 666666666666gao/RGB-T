"""One authorized training round, then the best model's complete evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2', '--master_port=29528',
               '-m', 'cadtrack_abc.train', '--config', args.config]
    with (output / 'train.log').open('w') as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    checkpoint = str(output / 'best.pth')
    reports, pending = {}, []
    for dataset, key in [('lasher', 'test_root'), ('rgbt234', 'rgbt234_root')]:
        # Data download may finish during this long, fixed training round.
        config = json.loads(Path(args.config).read_text())
        root = config[key]
        if root is None:
            pending.append(dataset + ': data unavailable')
            continue
        prediction_root = output / (dataset + '_predictions')
        jobs, streams = [], []
        for shard, physical_gpu in enumerate([2, 3]):
            stream = (output / (dataset + '_shard' + str(shard) + '.log')).open('w')
            streams.append(stream)
            command = [sys.executable, '-m', 'cadtrack_abc.infer', '--config', args.config,
                       '--checkpoint', checkpoint, '--dataset', dataset, '--root', root,
                       '--output', str(prediction_root), '--shard', str(shard)]
            jobs.append(subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(physical_gpu)),
                                         stdout=stream, stderr=subprocess.STDOUT))
        codes = [job.wait() for job in jobs]
        for stream in streams: stream.close()
        assert codes == [0, 0], (dataset, codes)
        subprocess.run([sys.executable, '-m', 'cadtrack_abc.score', '--dataset', dataset, '--root', root,
                        '--predictions', str(prediction_root), '--output', str(output / (dataset + '_report'))],
                       env=env, check=True)
        reports[dataset] = json.loads((output / (dataset + '_report/full_report.json')).read_text())
    terminal = {'training_complete': True, 'best_checkpoint': checkpoint, 'reports': reports,
                'pending': pending, 'all_five_metrics_complete': len(reports) == 2,
                'no_new_training_rounds': True}
    (output / 'pipeline_complete.json').write_text(json.dumps(terminal, indent=2))
    # last is necessary for resume during training; after strict best reload and
    # completed evaluations only the selected best modules checkpoint is retained.
    (output / 'last.pth').unlink()


if __name__ == '__main__':
    main()
