from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import os

output=Path('/home/gaob/CADTrack-ABC/profile_output/r2_dense_m0_20261015_attempt2')
with (output/'controller.log').open('w') as stream:
    completed=subprocess.run(['/data/wangwj/envs/CADTrack/bin/python',str(output/'driver.py')],
        stdout=stream,stderr=subprocess.STDOUT)
(output/'exit.json').write_text(json.dumps(dict(wrapper_pid=os.getpid(),actual_controller_exit_code=completed.returncode,
    completed_utc=datetime.now(timezone.utc).isoformat()),indent=2))
