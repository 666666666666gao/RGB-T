"""Copy the accepted private training sources without changing live tracking code."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

setup = Path('/data/gb/setup')
old = Path('/data/gb/experiments/recoverability_joint_b416_20261004')
private = Path('/data/gb/experiments/recoverability_best_policy_fit_20261004')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
review = json.loads((setup / 'best_policy_fit_source_review_20261004.json').read_text())
assert review['status'] == 'PASS'
prior = json.loads((setup / 'joint_b416_source_staging_20261004.json').read_text())
assert prior['status'] == 'PASS'
assert all(sha(path) == value for path, value in prior['audited_source_sha256'].items())
assert not private.exists()
sources = sorted((old / 'research').rglob('*.py'))
assert len(sources) == 39
source_hashes = {}
for source in sources:
    target = private / source.relative_to(old)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(source) == sha(target)
    source_hashes[str(target)] = sha(target)
for name, target_name in (
    ('run_recoverability_best_policy_fit_20261004.sh', 'scripts/run_recoverability_best_policy_fit.sh'),
    ('best_policy_fit_actual_cpu_audit_20261004.py', 'audit/actual_best_policy_fit_cpu_audit.py'),
):
    source = setup / name
    relative = 'scripts/run_recoverability_best_policy_fit.sh' if name.endswith('.sh') else 'refine-logs/runs/recoverability_best_policy_fit/actual_best_policy_fit_cpu_audit.py'
    assert sha(source) == review['source_sha256'][relative]
    target = private / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    source_hashes[str(target)] = sha(target)
main_sources = {path: value for path, value in prior['audited_source_sha256'].items() if path.startswith('/data/gb/GOLA/')}
source_hashes.update(main_sources)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(private) + ':/data/gb/GOLA', LD_LIBRARY_PATH='/data/gb/envs/gola/lib')
code = 'from pathlib import Path; import research.train_recoverability as t, research.recoverability_modules as m; root=Path("' + str(private / 'research') + '"); assert Path(t.__file__).parent == Path(m.__file__).parent == root; print(t.__file__,m.__file__)'
imports = subprocess.run([sys.executable, '-c', code], cwd=private, env=env, text=True, capture_output=True, check=True)
assert all(sha(path) == value for path, value in source_hashes.items())
record = {'status': 'PASS', 'staged_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'private_source': str(private), 'source_files': len(sources), 'all39_research_sources_identical_to_accepted_joint': True,
          'audited_source_sha256': source_hashes, 'main_models_unchanged': True,
          'private_cpu_imports_verified': True, 'actual_cpu_import_stdout': imports.stdout,
          'review_independence': 'same-family', 'acceptance_status': 'provisional', 'jobs_started': 0}
(setup / 'best_policy_fit_source_staging_20261004.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'source_files': len(sources), 'private': str(private), 'jobs_started': 0}))
