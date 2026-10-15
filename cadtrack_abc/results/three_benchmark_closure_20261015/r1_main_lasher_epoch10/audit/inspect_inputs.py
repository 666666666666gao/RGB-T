import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_lasher_audit_inputs_1040')
OUT = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_main_lasher_formal_audit_20261015')
PATHS = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_lasher_audit_paths.json')

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

manifest = load(PATHS)
initial = load(OUT / 'initial_input_hashes.json')
before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in manifest['files']}
dump('input_hashes_before.json', before)
assert initial == before
assert len(set(manifest['files'])) == len(manifest['files'])
raw_manifest = load(ROOT / 'raw_input_manifest.json')
local_source_checks = []
for record in raw_manifest['files']:
    path = ROOT / record['file']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    local_source_checks.append({'file': record['file'], 'source': record['source'],
                               'sha256_match': digest == record['sha256'],
                               'size_match': path.stat().st_size == record['bytes']})
dump('source_manifest_verification.json', local_source_checks)
summary = {'python': sys.version, 'python_executable': sys.executable,
           'platform': platform.platform(), 'numpy': np.__version__,
           'files': len(before), 'initial_hash_mismatches': 0,
           'input_groups': dict(Counter(Path(p).relative_to(ROOT).parts[0] for p in before)),
           'raw_manifest_keys': list(raw_manifest),
           'raw_manifest_files': len(raw_manifest['files']),
           'source_hash_mismatches': sum(not r['sha256_match'] for r in local_source_checks),
           'source_size_mismatches': sum(not r['size_match'] for r in local_source_checks),
           'manifest_other_fields': {k: v for k, v in raw_manifest.items() if k != 'files'}}
report = load(ROOT / 'report/full_report.json')
summary['report'] = {k: v for k, v in report.items() if k not in ('per_sequence', 'mean_curves', 'attributes')}
summary['report_attribute_names'] = list(report['attributes'])
summary['report_per_sequence_count'] = len(report['per_sequence'])
protocol = load(ROOT / 'toolkit_protocol.json')
summary['protocol_keys'] = list(protocol)
summary['protocol_sequence_count'] = len(protocol['sequences'])
summary['protocol_attribute_counts'] = {k: len(v) for k, v in protocol['attributes'].items()}
summary['receipts'] = []
for shard in (0, 1):
    receipt = load(ROOT / f'predictions/complete_shard{shard}.json')
    summary['receipts'].append({k: v for k, v in receipt.items() if k != 'records'})
mechanism = load(ROOT / 'report/mechanism_report.json')
summary['mechanism_keys'] = list(mechanism)
summary['mechanism_nonsequence_fields'] = {k: v for k, v in mechanism.items() if not isinstance(v, list)}
dump('input_inventory_summary.json', summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
