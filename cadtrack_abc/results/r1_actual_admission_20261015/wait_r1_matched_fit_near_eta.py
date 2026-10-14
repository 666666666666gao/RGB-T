from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

private = Path(__file__).parent
deadline = datetime.fromisoformat('2026-10-14T20:27:30+00:00').timestamp()
time.sleep(max(0, deadline - time.time()))
ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
       '-o', 'ServerAliveInterval=60', '-o', 'ConnectTimeout=10', '2028', 'python3 -']
source = (private / 'observe_r1_matched_fit_terminal_remote.py').read_bytes()
while True:
    response = subprocess.run(ssh, input=source, capture_output=True, check=True)
    result = json.loads(response.stdout)
    with (private / 'r1_matched_fit_near_eta_observations.jsonl').open('ab') as stream:
        stream.write(response.stdout.rstrip() + b'\n')
    if result['full_matched_fits_complete'] or result['incomplete_driver_exited']:
        (private / 'r1_matched_fit_observer_terminal.json').write_bytes(response.stdout)
        print(response.stdout.decode(), flush=True)
        break
    time.sleep(240)
