"""Collect the prepared 716 TRAIN transition states on four GPUs, sanity first."""
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path('/data/gb/GOLA')
FOLDER = ROOT / 'refine-logs/runs/recoverability_train_events'
SETUP = Path('/data/gb/setup')
PYTHON = '/data/gb/envs/gola/bin/python'
STATE = SETUP / 'train_events_collection_progress_20261005.json'


def read(path):
    return json.loads(Path(path).read_text())


def record(stage, **details):
    value = {'stage': stage, 'at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'pid': os.getpid(), 'optimization_started': False, 'official_metrics_completed': False} | details
    STATE.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def main():
    assert Path.cwd() == ROOT and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['PYTHONPATH'] == str(ROOT) and not STATE.exists()
    review = read(FOLDER / 'collection_source_review.json')
    assert review['status'] == 'PASS' and review['review_independence'] == 'same-family'
    assert review['acceptance_status'] == 'provisional' and review['runtime_attested'] is False
    plan = read(FOLDER / 'plan.json')
    base = read(Path(plan['base_train']) / 'config.json')
    source = read(ROOT / plan['source_eligibility'])['new_event_jobs']
    all_jobs = [job for entry in plan['full'].values() for job in read(entry['jobs_file'])['jobs']]
    pairs = lambda jobs: {(job['sequence'], job['query_frame']) for job in jobs}
    assert len(all_jobs) == len(pairs(all_jobs)) == 716 and pairs(all_jobs) == pairs(source)
    assert not pairs(all_jobs) & pairs(base['jobs'])
    assert Path(plan['teacher']).is_file() and base['prefix_model'] == plan['teacher']
    split = read(base['split'])
    assert {job['sequence'] for job in all_jobs} <= set(split['train'])
    assert not set(split['train']) & set(split['validation'])
    for phase in ('sanity', 'full'):
        for gpu in range(4):
            entry = plan[phase][str(gpu)]
            assert not Path(entry['output']).exists(), entry['output']
            assert len(read(entry['jobs_file'])['jobs']) == entry['clips']
    gpu_output = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
    gpu_rows = [[int(item.strip()) for item in line.split(',')] for line in gpu_output.splitlines()]
    assert len(gpu_rows) == 4 and all(row[1] < 500 for row in gpu_rows)
    free_bytes = shutil.disk_usage('/data/gb').free
    assert free_bytes > 10 * 2**30
    record('PREFLIGHT_COMPLETE', disk_free_bytes=free_bytes, gpu_memory_and_utilization=gpu_rows,
           new_TRAIN_queries=716, combined_TRAIN_queries=2478, unchanged_validation_queries=196)
    for phase in ('sanity', 'full'):
        children = []
        for gpu in range(4):
            entry = plan[phase][str(gpu)]
            command = ['bash', 'scripts/run_temporal.sh', str(gpu), 'collect_recoverability', '--partition', 'train',
                       '--prefix-model', plan['teacher'], '--future-policy', 'own', '--prefix-write-verification', 'action',
                       '--jobs-file', entry['jobs_file'], '--output', entry['output'], '--clips', str(entry['clips'])]
            for key in ('root', 'cache', 'split', 'pretrained', 'head', 'motion_run', 'batch_clips',
                        'forward_batch', 'max_prefix', 'seed'):
                command.extend(['--' + key.replace('_', '-'), str(base[key])])
            log_path = SETUP / ('train_events_' + phase + '_gpu' + str(gpu) + '_20261005.log')
            with log_path.open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            children.append((process, entry, log_path))
        record(phase.upper() + '_FOUR_GPU_COLLECTION_RUNNING',
               children=[{'pid': process.pid, 'output': entry['output'], 'log': str(log)}
                         for process, entry, log in children])
        for process, entry, log in children:
            returncode = process.wait()
            assert returncode == 0, (process.pid, returncode, str(log))
            completed = read(Path(entry['output']) / 'completion.json')
            assert completed['completed'] and completed['clips'] == entry['clips']
            (Path(entry['output']) / 'job_completed.txt').write_text('Actual collector exit0; TRAIN supervision cache only.\n')
        record(phase.upper() + '_COLLECTION_COMPLETE_CPU_GT_AUDIT_RUNNING')
        audit_log = SETUP / ('train_events_' + phase + '_cpu_audit_20261005.log')
        with audit_log.open('x') as log:
            subprocess.run([PYTHON, '-u', str(FOLDER / 'audit_collection_cpu.py'), '--phase', phase],
                           cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        acceptance = read(SETUP / ('train_events_' + phase + '_cpu_acceptance_20261005.json'))
        assert acceptance['status'] == 'PASS'
        record(phase.upper() + '_COLLECTION_AND_ACTUAL_GT_CPU_AUDIT_PASS',
               clips=acceptance['new_clips'], acceptance=acceptance)
    record('NEW716_TRAIN_STATES_READY_FOR_CAPACITY_AND_COMPLETE_FIT', full_train_queries=2478,
           validation_queries=196, next_stage='Actual four-GPU capacity M0, then matched420update full ABC fits; no training launched by this collection controller.')


if __name__ == '__main__':
    main()
