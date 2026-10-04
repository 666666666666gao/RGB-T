import ast
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

setup = Path('/data/gb/setup')
main = Path('/data/gb/GOLA')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
review = json.loads((setup / 'best_policy_collect_source_review_20261004.json').read_text())
assert review['status'] == 'PASS' and not review['blocking_issues']
assert review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
mapping = {
    'scripts/run_recoverability_best_policy_collect.sh': setup / 'run_recoverability_best_policy_collect_20261004.sh',
    'refine-logs/runs/recoverability_best_policy/actual_probe_cpu_audit.py': setup / 'best_policy_probe_actual_cpu_audit_20261004.py',
}
for part in ('train', 'validation'):
    for stage in ('probe', 'rest'):
        mapping['refine-logs/runs/recoverability_best_policy/jobs_' + part + '_' + stage + '.json'] = setup / ('best_policy_jobs_' + part + '_' + stage + '_20261004.json')
for path, destination in mapping.items():
    assert sha(destination) == review['source_sha256'][path], str(destination)
for relative in ('research/collect_recoverability.py', 'research/collect_rollouts.py', 'research/candidate_learning.py',
                 'research/recoverability_tracker.py', 'research/bounded_recovery.py'):
    assert sha(main / relative) == review['source_sha256'][relative], relative
assert sha(main / 'config/_dataset/train-lasher.yaml') == review['source_sha256']['refine-logs/runs/recoverability_best_policy/remote_train_lasher.yaml']
assert sha(main / 'trackit/datasets/MMOT/specialization/memory_mapped/dataset.py') == review['source_sha256']['refine-logs/runs/recoverability_best_policy/remote_MMOT_dataset.py']
assert sha(main / 'research/recoverability_modules.py') == '371269db053cdbd38b510557e213f1acfb9c6427e6eabd2df5685c59bdfbb158'
main_ast = ast.parse((main / 'research/recoverability_modules.py').read_text())
accepted_ast = ast.parse(Path('/data/gb/experiments/recoverability_joint_b416_20261004/research/recoverability_modules.py').read_text())
symbols = ('InstanceMemory', 'RecoverabilityModules', 'select_actions', 'DECISION_FIELDS')
def node(tree, name):
    return next(n for n in tree.body if (isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
                or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))
for name in symbols:
    assert ast.dump(node(main_ast, name), include_attributes=False) == ast.dump(node(accepted_ast, name), include_attributes=False), name
parent = Path('/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/best.pth')
assert sha(parent) == 'a165e06ffd8d0288d7db358359611d3ab1ed919941ff5fc2ccaa0cb14ddded72'
memory = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in memory.splitlines()}
assert used[1] < 500 and used[2] < 500, used
rows = []
for part, gpu in (('train', 1), ('validation', 2)):
    root = Path('/data/gb/outputs') / ('recoverability_best_policy_collect_' + part + '_probe_20261004')
    assert not root.exists()
    log = setup / (root.name + '.log')
    with log.open('wb') as stream:
        proc = subprocess.Popen(['bash', str(mapping['scripts/run_recoverability_best_policy_collect.sh']), str(gpu), part, 'probe'],
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, cwd=main)
    rows.append({'partition': part, 'gpu': gpu, 'runner_pid': proc.pid, 'root': str(root), 'log': str(log)})
record = {'status': 'LAUNCHED_NOT_ACCEPTED', 'launched_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'stage': 'probe', 'runs': rows, 'source_review_sha256': sha(setup / 'best_policy_collect_source_review_20261004.json'),
          'same_reviewed_collector_sources': True, 'frozen_main_model_used': True,
          'same_inference_model_AST_symbols': symbols, 'parent_sha256': sha(parent),
          'stage_source_sha256': {str(path): sha(path) for path in mapping.values()},
          'memory_before_mib': used, 'queries_per_partition': 16, 'source_scope': 'No architecture/tracker/collector modification; actual probes before rest.',
          'temperature_or_power_queries': False}
(setup / 'best_policy_probe_launch_20261004.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
