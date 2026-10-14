from pathlib import Path
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

root = Path('/home/gaob/CADTrack-ABC/outputs/actual_batch64_collector_20261014/packets')
receipt = json.loads((root / 'collection_complete.json').read_text())
rows = []
for record in receipt['records']:
    decision = torch.load(root / record['file'], map_location='cpu', weights_only=False)['decision']
    tokens = decision['contexts'][0]['tokens']
    rebuilt = torch.cat((tokens.mean(1), tokens.max(1).values, decision['motion']['means'].flatten(1), decision['motion']['log_weights']), -1)
    expected = decision['search_input']
    sections = {}
    for name, start, end in [('mean', 0, 128), ('max', 128, 256), ('means', 256, 292), ('log_weights', 292, 295)]:
        delta = (rebuilt[:, start:end] - expected[:, start:end]).abs()
        sections[name] = dict(max_absolute=float(delta.max()), differing=int((delta != 0).sum()))
    rows.append(dict(sequence=record['sequence'], frame=record['frame'], sections=sections,
                     all_finite=bool(torch.isfinite(expected).all()), maximum_absolute=float((rebuilt - expected).abs().max())))
print(json.dumps(dict(rows=rows, cuda_initialized=torch.cuda.is_initialized(), models_run=0), indent=2))
