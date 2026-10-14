from pathlib import Path
import json
import sys
sys.path.insert(0,'/home/gaob/CADTrack-ABC')
import torch

root=Path('/home/gaob/CADTrack-ABC/outputs')
old=root/'batched_actions_probe_20261014/serial'
new=root/'actual_batch64_collector_20261014/packets'
left=json.loads((old/'collection_complete.json').read_text())
right=json.loads((new/'collection_complete.json').read_text())
assert left['completed'] and right['completed'] and left['queries']==right['queries']==2
assert left['action_batch']==1 and right['action_batch']==64
assert left['teacher_sha256']==right['teacher_sha256']
maximum={}
for record_a,record_b in zip(left['records'],right['records']):
    assert record_a['file']==record_b['file'] and record_a['frame']==record_b['frame']
    a=torch.load(old/record_a['file'],map_location='cpu',weights_only=False)
    b=torch.load(new/record_b['file'],map_location='cpu',weights_only=False)
    assert len(a['labels'])==len(b['labels'])==210
    assert a['metadata']['reference_action']==b['metadata']['reference_action']
    assert a['metadata']['native_state_reconstruction_equal'] and b['metadata']['native_state_reconstruction_equal']
    for first,second in zip(a['labels'],b['labels']):
        assert set(first)==set(second)
        for key in ('context','candidate','mode','harm'):
            assert first[key]==second[key],(record_a['frame'],key)
        assert torch.equal(first['future_valid'],second['future_valid'])
        for key in set(first)-{'context','candidate','mode','harm','future_valid'}:
            x,y=torch.as_tensor(first[key]),torch.as_tensor(second[key])
            maximum[key]=max(maximum.get(key,0.),float((x-y).abs().max()))
            torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-4)
    for first,second in zip(a['decision']['contexts'],b['decision']['contexts']):
        assert set(first)==set(second)
        for key in first:
            torch.testing.assert_close(first[key],second[key],rtol=1e-6,atol=1e-6)
    reference=a['metadata']['reference_action']
    for packet in (a,b):
        target=next(row for row in packet['labels'] if (row['context'],row['candidate'],row['mode'])==reference)
        assert float(target['advantage'])==0. and not target['harm']
assert not torch.cuda.is_initialized()
result=dict(status='ACTUAL_COLLECTOR_420_LABELS_SERIAL_BATCH64_PARITY_PASS',queries=2,actions=420,horizon=3,
            maximum_error=maximum,rtol=1e-4,atol=1e-4,teacher_sha256=left['teacher_sha256'],
            exact_harm=True,reference_advantage_zero=True,decision_cache_rtol=1e-6,
            original_strict_mixed_failure_preserved=True,speedup_claim=False,cuda_initialized=False,updates=0)
(new.parent/'comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
