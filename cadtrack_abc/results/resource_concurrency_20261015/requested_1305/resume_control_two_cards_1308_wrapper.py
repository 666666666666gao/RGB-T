from datetime import datetime, timezone
from pathlib import Path
import json, subprocess, sys
root=Path('/home/gaob/CADTrack-ABC/outputs/r1_control_reports_20261015')
child=subprocess.run(['/data/wangwj/envs/CADTrack/bin/python',str(root/'resume_control_two_cards_1308_driver.py')])
(root/'recovery_1308_exit.json').write_text(json.dumps(dict(controller_exit_code=child.returncode,exited_utc=datetime.now(timezone.utc).isoformat()),indent=2))
sys.exit(child.returncode)