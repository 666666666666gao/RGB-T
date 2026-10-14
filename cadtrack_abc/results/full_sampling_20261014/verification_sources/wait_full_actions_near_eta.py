from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

private = Path(__file__).parent
out = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/cadtrack_abc/results/full_sampling_20261014')
eta = json.loads((out / 'action_eta.json').read_bytes())
deadline = datetime.fromisoformat(eta['first_observation_not_before_utc'])
time.sleep(max(0., (deadline - datetime.now(timezone.utc)).total_seconds()))
ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ServerAliveInterval=60', '-o', 'ConnectTimeout=10', '2028']
while True:
    raw = subprocess.run(ssh + ['python3 -'], input=(private / 'observe_full_actions_remote.py').read_bytes(),
                         capture_output=True, check=True).stdout
    result = json.loads(raw)
    with (private / 'full_action_near_eta_observations.jsonl').open('ab') as stream:
        stream.write(raw.rstrip() + b'\n')
    if result['full_action_collection_complete'] or result['incomplete_driver_exited']:
        (private / 'full_action_observer_terminal.json').write_bytes(raw)
        break
    time.sleep(240)
