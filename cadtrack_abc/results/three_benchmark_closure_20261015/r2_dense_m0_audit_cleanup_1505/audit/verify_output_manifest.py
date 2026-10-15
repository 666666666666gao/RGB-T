"""Read-only byte-exact final output verification. Does not alter the seal."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
mp=OUT/'output_manifest.json'
m=json.loads(mp.read_text(encoding='utf-8'))
actual={p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}
assert actual=={f['path'] for f in m['files']}|set(m['exclusions'])
for f in m['files']:
    raw=(OUT/f['path']).read_bytes()
    assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'],f['path']
digest=hashlib.sha256(mp.read_bytes()).hexdigest()
assert (OUT/'output_manifest.sha256').read_text().split()[0]==digest
assert m['count']==len(m['files'])
print(json.dumps(dict(status='PASS',hashed_files=len(m['files']),total_files=len(actual),manifest_sha256=digest),indent=2))
