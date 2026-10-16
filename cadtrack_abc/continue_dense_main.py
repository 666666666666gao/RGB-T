"""One dependent R3 launch after the original R2 pair actually succeeds.

The existing local observer still owns training-progress inspection. This
stdlib-only queue reads no parent files before the prescribed first check,
    then checks terminal receipts every300s. The main starts on freeGPU3 and
    its matched control independently waits for GPU2. It never preempts a GPU,
starts a second fit on a card or treats failed parents as successful inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', required=True)
    parser.add_argument('--r2-pair-root', required=True)
    parser.add_argument('--not-before', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    first_check = datetime.fromisoformat(args.not_before)
    assert first_check.tzinfo is not None
    project = Path(__file__).resolve().parent.parent
    scope_path = Path(args.scope).resolve()
    scope = json.loads(scope_path.read_text())
    command = scope['prepared_command_not_executed']
    assert Path(command[0]).resolve() == Path(sys.executable).resolve()
    assert command[1:3] == ['-m', 'cadtrack_abc.dense_main_pipeline']
    parent_root = Path(args.r2_pair_root).resolve()
    assert Path(command[command.index('--r2-pair-root') + 1]).resolve() == parent_root
    destination = Path(command[command.index('--output') + 1]).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='WAITING_UNTIL_NOT_BEFORE', queue_pid=os.getpid(),
        created_utc=datetime.now(timezone.utc).isoformat(), parent_root=str(parent_root),
        first_check_utc=first_check.isoformat(), poll_seconds=300,
        scope=str(scope_path), command=command, launch_count=0, launch_attempts=0,
        last_dependency_check_utc=None, main_controller_started=False,
        GPU_jobs_started_directly=0, goal_achieved=False)

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'queue_state.json').write_text(json.dumps(state, indent=2))

    save()
    time.sleep(max(0, first_check.timestamp() - time.time()))
    while True:
        state['last_dependency_check_utc'] = datetime.now(timezone.utc).isoformat()
        parent_exit = parent_root / 'pair_exit.json'
        if not parent_exit.exists():
            state['status'] = 'WAITING_FOR_ACTUAL_R2_EXIT'
            save()
            time.sleep(300)
            continue
        exited = json.loads(parent_exit.read_text())
        state['parent_exit'] = exited
        if exited['actual_controller_exit_code'] != 0:
            state['status'] = 'FAILED_R2_DEPENDENCY'
            save()
            return 1
        parent = json.loads((parent_root / 'pair_complete.json').read_text())
        assert parent['status'] == 'DENSE_MATCHED_PAIR_FULL_TRAINING_AND_EVALUATION_COMPLETE'
        assert parent['actual_exit_codes'] == [0, 0] and parent['all20_actual_clip_orders_equal']
        capacity = subprocess.run(['nvidia-smi', '-i', '2,3',
            '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout
        cards = [[int(value.strip()) for value in line.split(',')] for line in capacity.splitlines()]
        assert {row[0] for row in cards} == {2, 3}
        state['capacity_csv'] = capacity
        if next(row[1] for row in cards if row[0] == 3) >= 500:
            state['status'] = 'WAITING_FOR_AUTHORIZED_GPU_RELEASE'
            save()
            time.sleep(300)
            continue
        break

    # Reuse the already published review scope, rather than choosing another
    # weight, seed, source version or command at the wave transition.
    assert all(hashlib.sha256((project / name).read_bytes()).hexdigest() == digest
        for name, digest in scope['sources'].items())
    assert not destination.exists()
    config = json.loads(Path(command[command.index('--config') + 1]).read_text())
    free = shutil.disk_usage(project).free
    budget = Path(config['pretrained']).stat().st_size * 20
    assert free > budget
    state.update(home_free_bytes=free, required_output_budget_bytes=budget,
        status='R3_MAIN_CONTROLLER_STARTING', launch_attempts=1)
    save()
    with (output / 'main_controller.log').open('w') as stream:
        child = subprocess.Popen(command, cwd=project, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
        state.update(status='R3_MAIN_CONTROLLER_RUNNING', main_controller_pid=child.pid,
            main_controller_started_utc=datetime.now(timezone.utc).isoformat(),
            main_controller_started=True, launch_count=1,
            NN_training_and_evaluation_evidence=str(destination / 'queue_state.json'))
        save()
        code = child.wait()
    state.update(actual_main_controller_exit_code=code,
        main_controller_ended_utc=datetime.now(timezone.utc).isoformat())
    if code != 0:
        state['status'] = 'FAILED_R3_MAIN_CONTROLLER'
        save()
        return code
    completed = json.loads((destination / 'pipeline_complete.json').read_text())
    assert completed['status'] == 'DENSE_R3_COMPLETE_TRAINING_AND_EVALUATION'
    assert completed['full_ABC_training_complete']
    state.update(status='R3_COMPLETE_TRAINING_AND_EVALUATION_FINISHED',
        result=str(destination / 'pipeline_complete.json'),
        seven_metric_plus_two=completed['seven_metric_plus_two'], goal_achieved=False)
    save()
    print(json.dumps(state), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
