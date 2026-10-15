"""Launch one CPU-only dependency queue, not the new neural pipeline yet."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


os.environ['CUDA_VISIBLE_DEVICES'] = ''
project = Path('/home/gaob/CADTrack-ABC')
python = Path('/data/wangwj/envs/CADTrack/bin/python')
assert Path(sys.executable).resolve() == python.resolve()
environment = json.loads((project / 'cadtrack_abc/environment.json').read_text())
assert environment['provider'] == 'ssh2028'
assert environment['reuse_mode'] == 'warm_existing_conda_no_package_changes'
scope = project / 'cadtrack_abc/results/three_benchmark_closure_20261015/r3_complete_chain_source_1805/qualified_scope.json'
reviewed = json.loads(scope.read_text())
assert all(hashlib.sha256((project / name).read_bytes()).hexdigest() == digest
    for name, digest in reviewed['sources'].items())
queue_source = project / 'cadtrack_abc/continue_dense_main.py'
assert hashlib.sha256(queue_source.read_bytes()).hexdigest() == '919e59cdf8e0a1b1f0ea2ed941f733f5da1dcbd3c79c738d81946321b6525d1e'
root = project / 'outputs/r3_dense_dependency_20261015'
root.mkdir(parents=True, exist_ok=False)
not_before = '2026-10-15T23:39:58.342327+00:00'
command = [str(python), '-m', 'cadtrack_abc.continue_dense_main', '--scope', str(scope),
    '--r2-pair-root', str(project / 'outputs/r2_dense_pair_20261015'),
    '--not-before', not_before, '--output', str(root / 'state')]
wrapper = root / 'wrapper.py'
wrapper.write_text('from pathlib import Path\nfrom datetime import datetime,timezone\n'
    'import json,subprocess\nroot=Path(' + repr(str(root)) + ')\n'
    "with (root/'dependency_queue.log').open('w') as stream:\n"
    '    result=subprocess.run(' + repr(command) + ',cwd=' + repr(str(project)) +
    ',stdout=stream,stderr=subprocess.STDOUT)\n'
    "(root/'queue_exit.json').write_text(json.dumps(dict(actual_queue_exit_code=result.returncode,"
    'ended_utc=datetime.now(timezone.utc).isoformat()),indent=2))\n')
child = subprocess.Popen([str(python), str(wrapper)], cwd=project, stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
launch = dict(status='CPU_DEPENDENCY_QUEUE_LAUNCHED', wrapper_pid=child.pid, command=command,
    created_utc=datetime.now(timezone.utc).isoformat(), first_allowed_dependency_check_utc=not_before,
    neural_jobs_started=0, existing_training_modified=False, existing_observer_modified=False,
    parent_progress_or_logs_read=False, temperature_or_power_queried=False,
    home_free_bytes=shutil.disk_usage(project).free,
    environment_reused=True, new_main_pipeline_started=False)
(root / 'launch.json').write_text(json.dumps(launch, indent=2))
time.sleep(5)
state = json.loads((root / 'state/queue_state.json').read_text())
assert state['status'] == 'WAITING_UNTIL_NOT_BEFORE'
assert state['last_dependency_check_utc'] is None and state['launch_count'] == 0
assert not state['main_controller_started']
actual = Path(f"/proc/{state['queue_pid']}/cmdline").read_bytes().replace(b'\x00', b' ').decode()
assert 'cadtrack_abc.continue_dense_main' in actual
wrapper_actual = Path(f'/proc/{child.pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode()
assert str(wrapper) in wrapper_actual
launch.update(queue_pid=state['queue_pid'], queue_actual_command=actual,
    wrapper_actual_command=wrapper_actual, initial_state=state,
    verified_utc=datetime.now(timezone.utc).isoformat(), verified_CPU_queue_live=True)
(root / 'launch_verified.json').write_text(json.dumps(launch, indent=2))
print(json.dumps(launch, indent=2), flush=True)
