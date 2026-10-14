"""Matched R1 fits on separate cards, followed by serial full benchmark rounds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .future_pipeline import validate_benchmark_inventory, validate_collection, validate_parent, validate_sampling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sampling-root', required=True)
    parser.add_argument('--collection-root', required=True)
    parser.add_argument('--rgbt210-root', required=True)
    parser.add_argument('--rgbt234-root', required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    assert config['physical_gpus'] == [2, 3]
    parent = Path(config['output']).resolve()
    teacher, _ = validate_parent(parent)
    teacher_sha = hashlib.sha256(teacher.read_bytes()).hexdigest()
    manifest_path = parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    roots = {'lasher': Path(config['test_root']), 'rgbt210': Path(args.rgbt210_root),
             'rgbt234': Path(args.rgbt234_root)}
    inventories = {name: validate_benchmark_inventory(name, root) for name, root in roots.items()}
    sampling, collection = Path(args.sampling_root).resolve(), Path(args.collection_root).resolve()
    for split in ['train', 'validation']:
        allowed = set(manifest['train_sequences' if split == 'train' else 'validation_sequences'])
        validate_sampling(sampling / f'queries_{split}', allowed, teacher_sha)
        folders = [collection / f'collection_{split}' / f'shard{shard}' for shard in range(2)]
        validate_collection(folders, sampling / f'queries_{split}', teacher_sha)
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=False)
    project = Path(__file__).resolve().parent.parent
    state = dict(status='RUNNING', started_unix=time.time(), teacher=str(teacher),
        teacher_sha256=teacher_sha, formal_inventory=inventories, phases={},
        train_assignment={'future_actions': 2, 'search_actions': 3},
        evaluation_assignment='One family at a time, one worker per physical GPU2/3.',
        new_raw_sampling=False, existing_labels_reused=True, overall_goal_completed=False)

    def store():
        (output / 'queue_state.json').write_text(json.dumps(state, indent=2))

    def run(name, commands, devices):
        phase = dict(status='RUNNING', commands=commands, devices=devices, started_unix=time.time())
        state['phases'][name] = phase
        jobs = []
        for index, (command, device) in enumerate(zip(commands, devices)):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            with (output / f'{name}_{index}.log').open('w') as stream:
                jobs.append(subprocess.Popen([sys.executable, '-m', *command], cwd=project,
                    env=env, stdout=stream, stderr=subprocess.STDOUT))
        phase['pids'] = [job.pid for job in jobs]
        store()
        codes = [job.wait() for job in jobs]
        phase.update(status='COMMANDS_EXITED_SUCCESSFULLY' if codes == [0] * len(jobs) else 'FAILED',
                     return_codes=codes, exited_unix=time.time())
        if any(code != 0 for code in codes):
            state['status'] = 'FAILED'
        store()
        assert codes == [0] * len(jobs), (name, codes)

    indices = {}
    for split in ['train', 'validation']:
        indices[split] = output / f'{split}_index.json'
        folders = [collection / f'collection_{split}' / f'shard{shard}' for shard in range(2)]
        run(f'merge_{split}', [['cadtrack_abc.merge_action_packets', '--collections', *map(str, folders),
            '--manifest', str(manifest_path), '--split', split, '--output', str(indices[split])]], [''])
    policies = ['future_actions', 'search_actions']
    fits = {policy: output / f'fit_{policy}' for policy in policies}
    commands = [['cadtrack_abc.train_future_actions', '--config', str(config_path), '--teacher', str(teacher),
        '--manifest', str(manifest_path), '--train-index', str(indices['train']),
        '--validation-index', str(indices['validation']), '--output', str(fits[policy]),
        '--epochs', '20', '--batch', '64', '--lr', '0.0001', '--workers', '4',
        '--device', 'cuda', '--policy', policy] for policy in policies]
    run('matched_fits', commands, [2, 3])
    for policy in policies:
        receipt = json.loads((fits[policy] / 'fit_complete.json').read_text())
        queries = json.loads(indices['train'].read_text())['queries']
        assert receipt['epochs'] == 20 and receipt['updates'] == ((queries + 63) // 64) * 20
        assert receipt['policy_family'] == policy and receipt['teacher_sha256'] == teacher_sha
        assert receipt['train_queries'] == queries and receipt['train_sequences'] == 881
        assert receipt['strict_head_reload'] and receipt['changed_head_tensors']
        run(f'full_round_{policy}', [['cadtrack_abc.future_pipeline', '--config', str(config_path),
            '--output', str(output / f'round_{policy}'), '--sampling-root', str(sampling),
            '--collection-root', str(collection), '--fit-root', str(fits[policy]), '--policy', policy,
            '--rgbt210-root', str(roots['rgbt210']), '--rgbt234-root', str(roots['rgbt234']),
            '--lasher-target-gain', '6']], [''])
        complete = json.loads((output / f'round_{policy}' / 'pipeline_complete.json').read_text())
        assert complete['full_three_benchmarks_complete'] and complete['trained_checkpoint_formal_metrics_complete']
        state.setdefault('formal_rounds', {})[policy] = complete['formal_reports']
        store()
    state.update(status='R1_MATCHED_FITS_AND_FULL_ROUNDS_COMPLETE', completed_unix=time.time())
    store()
    (output / 'pipeline_complete.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
