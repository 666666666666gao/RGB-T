"""Explicit CPU program fixtures using actual M0 arrays; never an experiment result."""
import copy
import hashlib
import importlib.util
import json
import sys
import subprocess
import tempfile
from pathlib import Path
import numpy as np

candidate = Path('/data/gb/setup/current_policy_fit_review/merge_current_policy_caches.py')
spec = importlib.util.spec_from_file_location('_review_future_merge', candidate)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
source_roots = {p: Path('/data/gb/outputs/recoverability_current_policy_m0_20261004')/p for p in ('c1','own')}
source_configs = {p: json.loads((root/'config.json').read_text()) for p,root in source_roots.items()}
source_arrays = {}
for policy,root in source_roots.items():
    with np.load(root/'samples.npz') as archive:
        source_arrays[policy] = {key:archive[key].copy() for key in archive.files}

results = {}
def rejection(name, call):
    try:
        call()
    except AssertionError:
        results[name] = 'REJECTED_AS_REQUIRED'
    else:
        results[name] = 'UNEXPECTED_ACCEPTANCE'

with tempfile.TemporaryDirectory(prefix='current_policy_merge_program_fixture_',dir='/data/gb/setup/current_policy_fit_review') as temporary:
    base = Path(temporary)
    # Synthetic fixture IDs deliberately distinguish these copied rows from real jobs.
    jobs = [{'sequence':'CPU_FIXTURE_ONLY_'+str(i), 'query_frame':source_configs['c1']['jobs'][i%2]['query_frame']} for i in range(4)]
    for policy in ('c1','own'):
        for shard in range(4):
            root = base/f'recoverability_current_policy_{policy}_train_shard{shard}_20261004'
            root.mkdir()
            values = {key:array[shard%2:shard%2+1].copy() for key,array in source_arrays[policy].items()}
            config = copy.deepcopy(source_configs[policy])
            config.update(jobs=[jobs[shard]], clips=1, output=str(root), program_fixture_only=True)
            receipt = {'partition':'train','completed':True,'clips':1,
                       'decision_input_contains_future':False,'official_tracking_accuracy':False,
                       'valid_actions':int(values['action_valid'].sum()),'program_fixture_only':True}
            (root/'config.json').write_text(json.dumps(config))
            (root/'completion.json').write_text(json.dumps(receipt))
            (root/'job_completed.txt').write_text('CPU PROGRAM FIXTURE ONLY; not a collection\n')
            np.savez_compressed(root/'samples.npz',**values)
    read = lambda policy: module.read_partition(base,policy,'train',jobs)
    reference, configs, receipts, roots = read('c1')
    own, _, _, _ = read('own')
    assert len(reference['valid'])==4 and len(configs)==len(receipts)==len(roots)==4
    for key in reference:
        expected = np.concatenate([source_arrays['c1'][key][i%2:i%2+1] for i in range(4)])
        assert reference[key].dtype==expected.dtype and np.array_equal(reference[key],expected),key
    results['four_shards_ordered_exact_concatenation']='PASS'
    results['actual_m0_future_only_difference_accepted']=module.verify_pair(reference,own)
    changed = {key:value.copy() for key,value in own.items()}
    changed['wrong_update_fraction'].flat[0]=.25
    module.verify_pair(reference,changed)
    results['both_future_label_fields_may_differ']='PASS'
    changed = {key:value.copy() for key,value in own.items()}
    changed['anchor_features'].flat[0] += 1
    rejection('changed_decision_input',lambda:module.verify_pair(reference,changed))
    changed = {key:value.copy() for key,value in own.items()}
    changed['future_iou']=changed['future_iou'].astype(np.float64)
    rejection('mismatched_future_dtype',lambda:module.verify_pair(reference,changed))

    root = Path(roots[1])
    marker = root/'job_completed.txt'
    marker.unlink()
    rejection('missing_actual_completion_marker',lambda:read('c1'))
    proposed_output = base / 'CPU_FIXTURE_OUTPUT_NEVER_ACCEPTED'
    command = [sys.executable, str(candidate), '--base', str(base), '--output', str(proposed_output)]
    partial = subprocess.run(command, capture_output=True, text=True)
    assert partial.returncode != 0 and 'AssertionError' in partial.stderr and not proposed_output.exists()
    results['actual_main_partial_input_writes_no_output'] = 'PASS'
    marker.write_text('CPU PROGRAM FIXTURE ONLY\n')
    wrong_jobs = subprocess.run(command, capture_output=True, text=True)
    assert wrong_jobs.returncode != 0 and 'AssertionError' in wrong_jobs.stderr and not proposed_output.exists()
    results['actual_main_wrong_ordered_jobs_writes_no_output'] = 'PASS'
    rejection('different_expected_job_order',lambda:module.read_partition(base,'c1','train',list(reversed(jobs))))
    config_path=root/'config.json'
    original_config=json.loads(config_path.read_text())
    changed=copy.deepcopy(original_config)
    changed['forward_batch']=2
    config_path.write_text(json.dumps(changed))
    rejection('different_serial_execution_config',lambda:read('c1'))
    config_path.write_text(json.dumps(original_config))

    with np.load(root/'samples.npz') as archive:
        original_values={key:archive[key].copy() for key in archive.files}
    changed={key:value.copy() for key,value in original_values.items()}
    selected=int(changed['original_choice'][0])
    changed['action_valid'][0,0,selected,0]=False
    np.savez_compressed(root/'samples.npz',**changed)
    rejection('invalid_regular_action_mask',lambda:read('c1'))
    np.savez_compressed(root/'samples.npz',**original_values)

    # These fields are existing trainer config invariants, not speculative hardening.
    for key, replacement in [('root','/CPU_FIXTURE_OTHER_ROOT'),
                             ('cache','CPU_FIXTURE_OTHER_CACHE'),
                             ('split','CPU_FIXTURE_OTHER_SPLIT'),
                             ('regions',list(reversed(original_config['regions'])))]:
        changed=copy.deepcopy(original_config)
        changed[key]=replacement
        config_path.write_text(json.dumps(changed))
        rejection('trainer_required_config_'+key,lambda:read('c1'))
    config_path.write_text(json.dumps(original_config))

unexpected = [name for name,value in results.items() if value=='UNEXPECTED_ACCEPTANCE']
receipt = {'status':'FAIL' if unexpected else 'PASS','candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
           'unexpected_acceptances':unexpected,'checks':results,
           'fixture_scope':'Actual two-row CLI M0 NPZ payloads copied into four one-row shards with synthetic CPU_FIXTURE_ONLY job IDs. Tests program behavior only; no collection/benchmark completion is claimed.',
           'source_array_roots':{p:str(root) for p,root in source_roots.items()},
           'full_merge_main_executed':'negative incomplete/wrong-job fixtures only; positive complete collection pending','nn_imports':0,'nn_forward':0,'gpu_work':0,
           'existing_outputs_changed':False,'process_signals':0}
Path('/data/gb/setup/current_policy_fit_review/merge_cpu_fixture_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
