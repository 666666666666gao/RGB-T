from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

time.sleep(max(0,datetime(2026,10,15,7,40,0,tzinfo=timezone.utc).timestamp()-time.time()))
p=Path(__file__).parent
completed=subprocess.run([sys.executable,'-X','utf8','-B',str(p/'remote28.py'),str(p/'observe_dense_first_epoch_1540_remote.py')])
sys.exit(completed.returncode)
