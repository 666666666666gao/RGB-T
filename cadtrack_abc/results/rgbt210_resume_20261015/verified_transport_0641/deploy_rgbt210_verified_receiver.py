from pathlib import Path
import hashlib
import json
import subprocess

private = Path(__file__).parent
review = json.loads((private / 'rgbt210_partial_transport_source_review.json').read_text())
assert review['verdict'] == 'PASS' and review['blocking_count'] == 0
receiver = (private / 'receive_rgbt210_verified_remaining.py').read_text()
digest = hashlib.sha256(receiver.encode()).hexdigest()
source = ('from pathlib import Path\nimport hashlib,json\n'
    'directory=Path("/home/gaob/CADTrack-ABC/outputs/rgbt210_transport_resume_20261015")\n'
    'directory.mkdir(exist_ok=False)\n'
    'path=directory/"receive.py"\n'
    'path.write_text(' + repr(receiver) + ')\n'
    'assert hashlib.sha256(path.read_bytes()).hexdigest()==' + repr(digest) + '\n'
    'print(json.dumps(dict(deployed=True, receiver_sha256=' + repr(digest) + ')))\n')
result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
    '-o', 'ServerAliveInterval=60', '-o', 'ConnectTimeout=10', '2028', 'python3 -'],
    input=source.encode(), capture_output=True)
(private / 'rgbt210_receiver_deployment.stdout.txt').write_bytes(result.stdout)
(private / 'rgbt210_receiver_deployment.stderr.txt').write_bytes(result.stderr)
result.check_returncode()
print(result.stdout.decode(), flush=True)
