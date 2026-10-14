from pathlib import Path
import json
import os
import runpy
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
project = Path('/home/gaob/CADTrack-ABC')
sys.path.insert(0, str(project))
source = project / 'outputs/actual_batch64_collector_20261014/packets'
out = project / 'outputs/search_action_cli_CPU_sanity_20261014'
assert not out.exists()
sys.argv = ['train_future_actions', '--config', str(project / 'cadtrack_abc/config.json'),
            '--teacher', str(project / 'outputs/cad_abc_20261013/best.pth'),
            '--manifest', str(project / 'outputs/cad_abc_20261013/manifest.json'),
            '--train-index', str(source / 'collection_complete.json'),
            '--output', str(out), '--batch', '2', '--workers', '0', '--device', 'cpu',
            '--policy', 'search_actions', '--sanity']
runpy.run_module('cadtrack_abc.train_future_actions', run_name='__main__')
receipt = json.loads((out / 'fit_complete.json').read_text())
assert receipt['status'] == 'SEARCH_ACTION_SANITY_COMPLETE' and receipt['updates'] == 2
assert receipt['strict_head_reload'] and receipt['native_abc_frozen'] and receipt['motion_region_coordinates_frozen']
assert receipt['policy_family'] == 'search_actions' and receipt['trainable_parameters'] == 76678
import torch
from cadtrack_abc.future_actions import FutureActionHead
from cadtrack_abc.search_actions import SearchActionHead
teacher = torch.load(project / 'outputs/cad_abc_20261013/best.pth', map_location='cpu', weights_only=False)
payload = torch.load(receipt['candidates'][-1]['checkpoint'], map_location='cpu', weights_only=False)
assert all(torch.equal(value, teacher['abc'][key]) for key, value in payload['abc'].items())
FutureActionHead().load_state_dict(payload['future_action'], strict=True)
SearchActionHead().load_state_dict(payload['search_action'], strict=True)
assert not torch.cuda.is_initialized()
deleted = []
for candidate in receipt['candidates']:
    path = Path(candidate['checkpoint']).resolve()
    assert path.parent == out.resolve() and path.name.startswith('candidate_epoch') and path.suffix == '.pth'
    path.unlink()
    deleted.append(str(path))
result = dict(status='ACTUAL_SEARCH_ACTION_CLI_CPU_TWO_UPDATE_STRICT_RELOAD_PASS',
              fit_receipt=receipt, strict_both_head_reload=True, native_abc_byte_values_unchanged=True,
              cuda_initialized=False, cpu_sanity_weights_cleaned=deleted,
              full_fit_started=False, actual_visual_runtime_verified=False, new_formal_metrics=None)
(out / 'sanity_checked.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
