"""Read-only, stdlib input seal verifier and evidence reader; writes only here."""
import argparse, ast, hashlib, json
from pathlib import Path

OUT = Path(__file__).resolve().parent
INPUT = OUT.parent / 'dense_m0_fresh_audit_inputs_1435'
MANIFEST = INPUT / 'input_manifest.json'
INDEX = OUT.parent / 'dense_m0_fresh_audit_1435_paths.json'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['seal', 'read', 'scan'])
    ap.add_argument('items', nargs='*')
    ap.add_argument('--phase', default='before')
    args = ap.parse_args()
    m = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
    if args.mode == 'seal':
        rows = []
        for f in m['files']:
            p = Path(f['path']); data = p.read_bytes(); text = data.decode('utf-8-sig')
            parsed = 'text'
            if p.suffix == '.json':
                json.loads(text); parsed = 'json'
            if p.suffix == '.py':
                ast.parse(text, filename=f['relative']); parsed = 'ast'
            row = dict(relative=f['relative'], bytes=len(data), lines=len(text.splitlines()),
                       sha256=digest(data), expected_sha256=f['sha256'],
                       match=digest(data)==f['sha256'] and len(data)==f['bytes'], parsed=parsed)
            rows.append(row)
        idx = json.loads(INDEX.read_text(encoding='utf-8-sig'))
        r = dict(phase=args.phase, manifest_sha256=digest(MANIFEST.read_bytes()),
                 index_sha256=digest(INDEX.read_bytes()), declared_count=m['count'],
                 count=len(rows), bytes=sum(x['bytes'] for x in rows),
                 index_matches_manifest=sorted(idx['paths'])==sorted(f['path'] for f in m['files']),
                 matched=sum(x['match'] for x in rows), files=rows)
        (OUT / ('input_seal_'+args.phase+'.json')).write_text(json.dumps(r, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({k:v for k,v in r.items() if k!='files'}, indent=2))
        assert r['matched']==r['count']==r['declared_count']==85
        assert r['index_matches_manifest']
    elif args.mode == 'read':
        for item in args.items:
            parts = item.rsplit(':',2)
            rel = parts[0]
            lines = (INPUT/'project'/rel).read_text(encoding='utf-8-sig').splitlines()
            start,end = (int(parts[1]),int(parts[2])) if len(parts)==3 else (1,len(lines))
            print('\nFILE '+rel)
            for n in range(start,min(end,len(lines))+1):
                print(f'{n}: {lines[n-1]}')
    else:
        for f in m['files']:
            t=Path(f['path']).read_text(encoding='utf-8-sig')
            print(f['relative'], len(t.splitlines()), digest(t.encode()))

if __name__ == '__main__':
    main()
