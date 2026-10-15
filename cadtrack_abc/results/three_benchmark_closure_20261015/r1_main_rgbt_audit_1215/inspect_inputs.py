import collections
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / 'main_rgbt_audit_inputs_1150'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
def j(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
mode = sys.argv[1]
if mode in ('before', 'after'):
    mapping = {p.relative_to(ROOT).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(ROOT.rglob('*')) if p.is_file()}
    (OUT / ('input_hashes_' + mode + '.json')).write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    initial = j(OUT / 'initial_input_hashes.json')
    print(json.dumps({'files': len(mapping), 'bytes': sum(v['bytes'] for v in mapping.values()), 'initial_type': type(initial).__name__, 'initial_keys_sample': list(initial)[:3], 'initial_value_sample': list(initial.values())[:1]}, indent=2))
elif mode == 'inventory':
    for rel in ['raw_input_manifest.json', 'toolkit_protocols.json', 'main_observation_1150.json', 'documentation/plan_snapshot.json']:
        data = j(ROOT / rel)
        print(rel, json.dumps({'type': type(data).__name__, 'keys': list(data) if isinstance(data, dict) else len(data)}, indent=2))
        if rel == 'raw_input_manifest.json':
            for k,v in data.items():
                print(k, str(v)[:1500])
        elif rel == 'toolkit_protocols.json':
            for k,v in data.items():
                print(k, json.dumps({k2:(len(v2) if isinstance(v2,(list,dict)) else v2) for k2,v2 in v.items()}))
        else:
            print(json.dumps(data, indent=2))
    for p in sorted((ROOT / 'results').rglob('*')):
        if p.is_file():
            print('RESULT', p.relative_to(ROOT).as_posix(), p.stat().st_size)
            if p.suffix == '.json':
                d=j(p)
                print(json.dumps({k:v for k,v in d.items() if k not in ['per_sequence','mean_curves','attributes']}, indent=2))
    for p in sorted((ROOT / 'predictions').rglob('complete*.json')):
        d=j(p)
        print('RECEIPT', p.relative_to(ROOT).as_posix(), json.dumps({k:(len(v) if k=='records' else v) for k,v in d.items()}, indent=2))
        print('record_keys', list(d['records'][0]))
    print('attributes', [str(p.relative_to(ROOT)) for p in sorted((ROOT/'toolkit_attributes').rglob('*')) if p.is_file()])
    print('gt_sample', [str(p.relative_to(ROOT)) for p in sorted((ROOT/'ground_truth').rglob('*')) if p.is_file()][:6])
    print('extension_counts', dict(collections.Counter(p.suffix for p in ROOT.rglob('*') if p.is_file())))
