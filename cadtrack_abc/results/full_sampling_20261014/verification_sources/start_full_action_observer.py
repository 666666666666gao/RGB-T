from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys

private = Path(__file__).parent
out = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/cadtrack_abc/results/full_sampling_20261014')
receipt_path = private / 'full_action_sole_observer_launch.json'
assert not receipt_path.exists()
eta = json.loads((out / 'action_eta.json').read_bytes())
command = [sys.executable, '-X', 'utf8', '-B', str(private / 'wait_full_actions_near_eta.py')]
with (private / 'full_action_sole_observer.log').open('wb') as stream:
    process = subprocess.Popen(command, cwd=private, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
receipt = dict(pid=process.pid, command=command, started_utc=datetime.now(timezone.utc).isoformat(),
               first_remote_observation_not_before_utc=eta['first_observation_not_before_utc'],
               interval_after_deadline_seconds=240, no_interim_remote_queries=True,
               no_automatic_training=True)
raw = (json.dumps(receipt, indent=2) + '\n').encode()
receipt_path.write_bytes(raw)
(out / 'action_observer_launch.json').write_bytes(raw)
print(json.dumps(receipt))
