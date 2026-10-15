"""Preserve true final-validation exit and seal every completed audit output."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
prefix = OUT / 'execution/003_final_artifact_validation'
assert not prefix.with_suffix('.json').exists()
command = [sys.executable, '-B', str(OUT / 'validate_artifacts.py')]
record = dict(command=command, cwd=str(OUT), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              script_sha256=hashlib.sha256((OUT / 'validate_artifacts.py').read_bytes()).hexdigest())
completed = subprocess.run(command, cwd=OUT, capture_output=True)
prefix.with_suffix('.stdout.txt').write_bytes(completed.stdout)
prefix.with_suffix('.stderr.txt').write_bytes(completed.stderr)
record.update(returncode=completed.returncode, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              stdout_file=str(prefix.with_suffix('.stdout.txt')), stderr_file=str(prefix.with_suffix('.stderr.txt')))
prefix.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
sys.stdout.buffer.write(completed.stdout)
sys.stderr.buffer.write(completed.stderr)
if completed.returncode != 0:
    raise SystemExit(completed.returncode)
manifest_path = OUT / 'output_hash_manifest.json'
assert not manifest_path.exists()
files = {p.relative_to(OUT).as_posix(): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
         for p in sorted(OUT.rglob('*')) if p.is_file() and p != manifest_path}
manifest = dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), algorithm='SHA256',
                scope='All completed audit outputs, including complete numerical execution logs and validation; self excluded.',
                file_count=len(files), files=files)
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
actual_paths = {p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file() and p != manifest_path}
assert actual_paths == set(files)
assert all((OUT / p).stat().st_size == row['bytes'] and hashlib.sha256((OUT / p).read_bytes()).hexdigest() == row['sha256'] for p, row in files.items())
print(json.dumps(dict(output_manifest=str(manifest_path), file_count=len(files), all_output_hashes_verified=True,
                     manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), final_validation_child_returncode=completed.returncode)), flush=True)
