from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

project = Path('/home/gaob/CADTrack-ABC').resolve()
folder = project / 'profile_output/dense_resources_requested_20261015'
inventory_path = folder / 'closed_probe_weight_inventory.json'
assert hashlib.sha256(inventory_path.read_bytes()).hexdigest() == 'cbbc6ddb1d07c79a4e1cd04e0cfa066ad466975e7cf658d2adb2eed3d9740966'
inventory = json.loads(inventory_path.read_text())
assert inventory['file_count'] == 18 and inventory['total_bytes'] == 2076823422
protected = {
    Path('/home/gaob/CADTrack_ep0014.pth.tar'): 'cbcc96ba3a0cf1bfb997ee164174c2264cabf17ad013f8249b7f31a43815d731',
    project / 'outputs/r1_matched_fits_20261015/fit_search_actions/best.pth': 'a14130fafbc7eb49a17a3834fe8a43855f3ed0833ba576e737e74d1579965ad8',
    project / 'outputs/r1_matched_fits_20261015/fit_future_actions/best.pth': 'cffca0302460b60f505c2a09a5b651b10f1d1e2211178f68307a4d708080bf62',
}
for path, digest in protected.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
roots = {Path(item['path']).resolve(): item for item in inventory['roots']}
for root, item in roots.items():
    assert root.is_relative_to(project / 'profile_output')
    assert root.name in ('r2_dense_m0_20261015_attempt2', 'r2_dense_m0_capacity128_20261015')
    assert hashlib.sha256((root / 'state.json').read_bytes()).hexdigest() == item['state_sha256']
    assert hashlib.sha256((root / 'exit.json').read_bytes()).hexdigest() == item['exit_sha256']
    assert json.loads((root / 'state.json').read_text())['status'] == 'DENSE_M0_COMPLETE'
    assert json.loads((root / 'exit.json').read_text())['actual_controller_exit_code'] == 0
weights = []
for item in inventory['files']:
    path = Path(item['path']).resolve()
    assert any(path.is_relative_to(root) for root in roots)
    assert not path.is_relative_to(project / 'outputs')
    assert path not in protected
    assert path.name in ('candidate_epoch00.pth', 'candidate_epoch01.pth', 'last.pth')
    assert item['probe'] and not item['formal_or_selected_best']
    assert path.stat().st_size == item['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    complete = json.loads((path.parent / 'fit_complete.json').read_text())
    assert complete['probe'] and complete['updates'] == 2 and complete['strict_dense_reload_passed']
    weights.append(path)
assert len(set(weights)) == 18
before = shutil.disk_usage(project).free
started = datetime.now(timezone.utc).isoformat()
for path in weights:
    path.unlink()
    print(json.dumps(dict(deleted_probe_checkpoint=str(path))), flush=True)
assert not any(path.exists() for path in weights)
for path, digest in protected.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
result = dict(status='CLOSED_PROBE_WEIGHTS_CLEANED', started_utc=started,
    completed_utc=datetime.now(timezone.utc).isoformat(), deleted_files=inventory['files'],
    deleted_count=18, removed_file_bytes=inventory['total_bytes'],
    disk_free_bytes_before=before, disk_free_bytes_after=shutil.disk_usage(project).free,
    protected_pretrain_and_selected_best_sha256={str(path): digest for path, digest in protected.items()},
    current_full_training_outputs_excluded=True, all_metadata_logs_receipts_preserved=True,
    audit_verdict='WARN', audit_blocking_findings=0,
    audit_report_sha256='63cb905b61e3242e639d514e23988beb18d01eae2693538d50eeb3ed251965b1',
    audit_sealed_outputs_sha256='a8b0494895c8b6994cee3470c2cf30a66d4c69db4a9b2a1315ac099ef6b2a470',
    audit_input_weight_files=0, new_neural_jobs=0, training_interrupted=False,
    goal_achieved=False)
(folder / 'closed_probe_weight_cleanup.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
