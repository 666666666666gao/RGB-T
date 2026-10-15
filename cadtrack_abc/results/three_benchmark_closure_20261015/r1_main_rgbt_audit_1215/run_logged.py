"""CPU audit command runner; only writes audit-directory execution evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
label = sys.argv[1]
command = sys.argv[2:]
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
t0 = time.perf_counter()
(OUT / (label + '.command.json')).write_text(json.dumps({'command': command, 'started_utc': start}, indent=2), encoding='utf-8')
result = subprocess.run(command, cwd=OUT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(OUT / (label + '.stdout.txt')).write_bytes(result.stdout)
(OUT / (label + '.stderr.txt')).write_bytes(result.stderr)
receipt = {'command': command, 'started_utc': start, 'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - t0, 'exit_code': result.returncode, 'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(), 'stderr_sha256': hashlib.sha256(result.stderr).hexdigest(), 'stdout_file': label + '.stdout.txt', 'stderr_file': label + '.stderr.txt'}
(OUT / (label + '.execution.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
sys.stdout.buffer.write(result.stdout)
sys.stderr.buffer.write(result.stderr)
print(json.dumps(receipt))
sys.exit(result.returncode)
