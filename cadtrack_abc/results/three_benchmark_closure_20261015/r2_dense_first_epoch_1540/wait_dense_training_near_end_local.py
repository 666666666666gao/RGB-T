from datetime import datetime
from pathlib import Path
import json
import subprocess
import sys
import time

p = Path(__file__).parent
schedule = json.loads((p/'dense_first_epoch_1540_ETA.json').read_text())
deadline = datetime.fromisoformat(schedule['next_only_observation_utc']).timestamp()
time.sleep(max(0,deadline-time.time()))
completed = subprocess.run([sys.executable,'-X','utf8','-B',str(p/'remote28.py'),
    str(p/'observe_dense_training_near_end_remote.py')])
sys.exit(completed.returncode)
