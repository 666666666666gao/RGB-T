"""Run one local audit script and preserve its actual stdout, stderr and exit code."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
name, script, *args = sys.argv[1:]
trace = ROOT / 'execution'
trace.mkdir(exist_ok=True)
prefix = trace / name
assert not prefix.with_suffix('.json').exists(), name
command = [sys.executable, '-B', str(ROOT / script), *args]
record = {'command': command, 'cwd': str(ROOT), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'script_sha256': hashlib.sha256((ROOT / script).read_bytes()).hexdigest()}
completed = subprocess.run(command, cwd=ROOT, capture_output=True)
prefix.with_suffix('.stdout.txt').write_bytes(completed.stdout)
prefix.with_suffix('.stderr.txt').write_bytes(completed.stderr)
record.update(returncode=completed.returncode, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              stdout_file=str(prefix.with_suffix('.stdout.txt')), stderr_file=str(prefix.with_suffix('.stderr.txt')))
prefix.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
sys.stdout.buffer.write(completed.stdout)
sys.stderr.buffer.write(completed.stderr)
print(json.dumps({'execution_record': str(prefix.with_suffix('.json')), 'returncode': completed.returncode}), flush=True)
raise SystemExit(completed.returncode)
