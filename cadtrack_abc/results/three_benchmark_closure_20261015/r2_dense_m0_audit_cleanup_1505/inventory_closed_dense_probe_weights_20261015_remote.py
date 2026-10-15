from pathlib import Path
import hashlib
import json

project = Path('/home/gaob/CADTrack-ABC').resolve()
roots = [project / 'profile_output/r2_dense_m0_20261015_attempt2',
         project / 'profile_output/r2_dense_m0_capacity128_20261015']
result = dict(scope='Closed qualification probe checkpoints only; current full-fit candidates and required dependencies are excluded.',
              files=[], roots=[], deleted_files=[], deletion_performed=False)
for root in roots:
    root = root.resolve()
    assert root.is_relative_to(project / 'profile_output')
    state = json.loads((root / 'state.json').read_text())
    terminal = json.loads((root / 'exit.json').read_text())
    assert state['status'] == 'DENSE_M0_COMPLETE'
    assert terminal['actual_controller_exit_code'] == 0
    result['roots'].append(dict(path=str(root), state_sha256=hashlib.sha256((root / 'state.json').read_bytes()).hexdigest(),
                               exit_sha256=hashlib.sha256((root / 'exit.json').read_bytes()).hexdigest()))
    folders = ['H_batch32', 'HA_batch32', 'H_batch64', 'HA_batch64'] if 'attempt2' in root.name else ['H_batch128', 'HA_batch128']
    for name in folders:
        folder = (root / name).resolve()
        assert folder.is_relative_to(root)
        manifest = json.loads((folder / 'manifest.json').read_text())
        complete = json.loads((folder / 'fit_complete.json').read_text())
        assert manifest['config']['probe']
        assert complete['status'] == 'DENSE_QUALIFICATION_COMPLETE' and complete['probe']
        assert complete['updates'] == 2 and complete['strict_dense_reload_passed']
        weights = sorted(folder.glob('*.pth'))
        assert {f.name for f in weights} == {'candidate_epoch00.pth', 'candidate_epoch01.pth', 'last.pth'}
        for weight in weights:
            resolved = weight.resolve()
            assert resolved.is_relative_to(folder)
            result['files'].append(dict(path=str(resolved), bytes=resolved.stat().st_size,
                sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(), probe=True, formal_or_selected_best=False))
result['total_bytes'] = sum(item['bytes'] for item in result['files'])
result['file_count'] = len(result['files'])
assert result['file_count'] == 18
result['protected_paths'] = ['/home/gaob/CADTrack_ep0014.pth.tar',
    str(project / 'outputs/r1_matched_fits_20261015/fit_search_actions/best.pth'),
    str(project / 'outputs/r1_matched_fits_20261015/fit_future_actions/best.pth'),
    str(project / 'outputs/r2_dense_pair_20261015')]
output = project / 'profile_output/dense_resources_requested_20261015/closed_probe_weight_inventory.json'
output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
