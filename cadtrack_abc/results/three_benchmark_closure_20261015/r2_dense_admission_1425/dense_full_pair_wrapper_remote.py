from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

root=Path('/home/gaob/CADTrack-ABC/outputs/r2_dense_pair_20261015')
with (root/'pair_controller.log').open('w') as stream:
    child=subprocess.run(['/data/wangwj/envs/CADTrack/bin/python',str(root/'driver.py')],
        stdout=stream,stderr=subprocess.STDOUT)
(root/'pair_exit.json').write_text(json.dumps(dict(actual_controller_exit_code=child.returncode,
    completed_utc=datetime.now(timezone.utc).isoformat()),indent=2))
