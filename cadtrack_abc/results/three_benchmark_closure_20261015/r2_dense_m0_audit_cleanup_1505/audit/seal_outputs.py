"""Hash all existing audit deliverables; manifest and detached digest exclude self."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
excluded={'output_manifest.json','output_manifest.sha256'}
files=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.relative_to(OUT).as_posix() not in excluded:
        raw=p.read_bytes()
        files.append(dict(path=p.relative_to(OUT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
manifest=dict(schema='sha256-byte-exact-v1',count=len(files),files=files,
    exclusions=sorted(excluded),exclusion_reason='Manifest cannot contain its own digest; detached digest authenticates manifest bytes.',
    privacy='Private audit output. In particular .aris and private_json_leaf_inventory.json are not public deliverables by default.')
p=OUT/'output_manifest.json'
p.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
digest=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'output_manifest.sha256').write_text(digest+'  output_manifest.json\n',encoding='ascii')
print(json.dumps(dict(count=len(files),manifest_sha256=digest),indent=2))
