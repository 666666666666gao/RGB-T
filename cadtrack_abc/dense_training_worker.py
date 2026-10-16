"""One fixed R3 branch: wait for its card, qualify2 updates, then full20.

The main branch uses physicalGPU3; the matched control waits for physicalGPU2.
The parent supplies its existing training command lists without changing losses,
initialization, sampling, optimizer, batch or training budget.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    variant = spec['variant']
    device = {'raw': 3, 'control': 2}[variant]
    module = {'raw': 'cadtrack_abc.train_dense_motion', 'control': 'cadtrack_abc.train_dense_continuation'}[variant]
    assert spec['physical_gpu'] == device
    assert spec['probe_command'][0] == spec['full_command'][0] == module
    assert '--probe' in spec['probe_command'] and '--probe' not in spec['full_command']
    output = Path(spec['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parent.parent
    state = dict(status='WAITING_FOR_CARD', worker_pid=os.getpid(), variant=variant,
                 physical_gpu=device, created_utc=datetime.now(timezone.utc).isoformat(),
                 phases={}, goal_achieved=False)

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'worker_state.json').write_text(json.dumps(state, indent=2))

    save()
    while True:
        capacity = subprocess.run(['nvidia-smi', '-i', str(device),
            '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout
        card = [int(value.strip()) for value in capacity.strip().split(',')]
        assert card[0] == device
        state['capacity_csv'] = capacity
        if card[1] < 500:
            break
        save()
        time.sleep(300)

    def run(name, command):
        phase = dict(status='RUNNING', command=command,
                     started_utc=datetime.now(timezone.utc).isoformat())
        state['phases'][name] = phase
        state['status'] = name.upper() + '_RUNNING'
        with (output / (name + '.log')).open('w') as stream:
            child = subprocess.Popen([sys.executable, '-m', *command], cwd=project,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='4',
                         OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'),
                stdout=stream, stderr=subprocess.STDOUT)
            phase['pid'] = child.pid
            save()
            code = child.wait()
        phase.update(status='COMPLETE' if code == 0 else 'FAILED', actual_exit_code=code,
                     completed_utc=datetime.now(timezone.utc).isoformat())
        if code != 0:
            state['status'] = 'FAILED'
        save()
        return code

    code = run('probe', spec['probe_command'])
    if code != 0:
        return code
    probe = json.loads((Path(spec['probe_output']) / 'fit_complete.json').read_text())
    expected = {'raw': 'DENSE_MOTION_QUALIFICATION_COMPLETE', 'control': 'DENSE_QUALIFICATION_COMPLETE'}[variant]
    qualified = (probe['status'] == expected and probe['probe'] and probe['updates'] == 2
        and probe['raw_clips'] == spec['batch'] * 2 and all(probe['changed_parameter_tensors'].values()))
    if not qualified:
        state['status'] = 'FAILED_PROBE_QUALIFICATION'
        save()
        return 1
    state['own_probe_qualified_before_full_fit'] = True
    save()
    code = run('full20', spec['full_command'])
    if code != 0:
        return code
    state['status'] = 'BRANCH_COMMANDS_COMPLETE'
    save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
