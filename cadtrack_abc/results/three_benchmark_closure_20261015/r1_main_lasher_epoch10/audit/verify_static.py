"""Small static checks for protocol bindings and bounded provenance coverage."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/main_lasher_audit_inputs_1040')
OUT = Path('C:/Users/gb/.codex_tmp/cadtrack_abc_20261013/r1_main_lasher_formal_audit_20261015')

def parse(rel):
    return ast.parse((ROOT / rel).read_text(encoding='utf-8'))

def load(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))

def assigned(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise AssertionError(name)

checks = []
def check(label, value):
    checks.append({'label': label, 'pass': bool(value)})

lasher = parse('toolkit/rgbt/dataset/lasher_dataset.py')
attributes = None
bindings = {}
for node in ast.walk(lasher):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                if target.attr == '_attr_list':
                    attributes = ast.literal_eval(node.value)
                if target.attr in ('PR_fun', 'NPR_fun', 'SR_fun'):
                    bindings[target.attr] = node.value.func.id
protocol = load('toolkit_protocol.json')
check('LasHeR attribute order matches exported protocol', list(attributes) == list(protocol['attributes']))
check('LasHeR exact metric class bindings', bindings == {'PR_fun': 'PR_LasHeR', 'NPR_fun': 'NPR', 'SR_fun': 'SR_LasHeR'})
score_tree = parse('project/cadtrack_abc/score.py')
score_metrics = assigned(score_tree, 'metrics')
check('LasHeR scorer invokes NPR then PR then SR', ast.literal_eval(score_metrics.value)['lasher'] == ['NPR', 'PR', 'SR'])
relay = parse('main_formal_relay.py')
source_names = ast.literal_eval(assigned(relay, 'source_names'))
run_gpu = [ast.literal_eval(k.value) for n in ast.walk(relay) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run' for k in n.keywords if k.arg == 'gpu']
check('Formal relay has a single inference GPU routing value 2', run_gpu == [2])
observation = load('main_observation_1040.json')
plan = load('documentation/plan_snapshot.json')
recorded_hashes = []
def collect_hashes(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'selected_checkpoint_sha256':
                recorded_hashes.append(child)
            collect_hashes(child)
    elif isinstance(value, list):
        for child in value:
            collect_hashes(child)
collect_hashes(plan)
check('Observed checkpoint digest appears in prior plan selection record', observation['checkpoint_sha256'] in recorded_hashes)
check('Formal observation has only completed LasHeR report', list(observation['reports']) == ['lasher'])
check('Formal observation records continuing RGBT234 rather than all-dataset completion', observation['status'] == 'RUNNING' and list(observation['current_jobs']) == ['infer_rgbt234'])
check('Report stage target agrees with current plan', load('report/full_report.json')['lasher_stage_goal']['targets_percent'] == plan['current_acceptance_20261015']['lasher_targets_percent'])
check('Plan explicitly denies overall completion', plan['current_acceptance_20261015']['goal_completion_proven'] is False)

relative_imports = []
for path in sorted((ROOT / 'project/cadtrack_abc').glob('*.py')):
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            target = path.parent / (node.module + '.py')
            relative_imports.append({'importer': str(path.relative_to(ROOT)).replace('\\', '/'), 'line': node.lineno,
                                     'module': node.module, 'supplied': target.is_file()})
result = {'checks': checks, 'comparisons': len(checks), 'failed': sum(not c['pass'] for c in checks),
          'sealed_source_names_in_relay': source_names,
          'present_runtime_sources_outside_relay_seal': [name for name in ['search_actions.py', 'ported_modules.py', 'mechanism_summary.py'] if name not in source_names],
          'missing_relative_dependencies': [r for r in relative_imports if not r['supplied']],
          'upstream_vendor_sources_present': (ROOT / 'project/cadtrack_abc/vendor').is_dir(),
          'checkpoint_bytes_present': any(ROOT.rglob('*.pth')),
          'raw_mechanism_trace_count': len(list((ROOT / 'predictions').glob('*_mechanisms.json'))),
          'raw_attribute_text_directory_present': any(p.name == 'AttriSeqsTxt' for p in ROOT.rglob('*') if p.is_dir()),
          'source_file_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT / 'static_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['failed'] == 0 else 1)
