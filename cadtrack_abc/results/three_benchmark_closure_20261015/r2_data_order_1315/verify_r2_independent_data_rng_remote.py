"""CPU-only witness for the observed model-init/DataLoader RNG coupling."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from torch import nn
from torch.utils.data import DataLoader

torch.set_num_threads(1)
source = Path('/home/gaob/CADTrack-ABC/cadtrack_abc/train.py')
before = hashlib.sha256(source.read_bytes()).hexdigest()

def order(extra_projection, independent_generator):
    torch.manual_seed(42)
    base = nn.Linear(128,128)
    prompt = nn.Linear(128,128) if extra_projection else None
    generator = torch.Generator().manual_seed(42) if independent_generator else None
    loader = DataLoader(list(range(100)),batch_size=16,shuffle=True,num_workers=0,generator=generator)
    return [int(index) for batch in loader for index in batch]

records = {name:order(extra,generator) for name,extra,generator in (
    ('global_rng_head',False,False),('global_rng_head_and_prompt',True,False),
    ('independent_rng_head',False,True),('independent_rng_head_and_prompt',True,True))}
assert records['global_rng_head'] != records['global_rng_head_and_prompt']
assert records['independent_rng_head'] == records['independent_rng_head_and_prompt']
assert all(sorted(values)==list(range(100)) for values in records.values())
assert hashlib.sha256(source.read_bytes()).hexdigest() == before
assert not torch.cuda.is_initialized()
result = dict(status='CPU_DATA_RNG_COUPLING_WITNESS_PASS',observed_utc=datetime.now(timezone.utc).isoformat(),
    torch_version=torch.__version__,seed=42,source=str(source),source_sha256=before,
    records=records,order_sha256={name:hashlib.sha256(json.dumps(values).encode()).hexdigest() for name,values in records.items()},
    global_seed_alone_keeps_paired_order=False,separate_loader_generator_keeps_paired_order=True,
    scope='Actual Torch DataLoader CPU behavior with controlled extra module initialization. Not an integrated R2 video-data run or a tracking accuracy result.',
    neural_gpu_work_started=False,optimizer_updates=0,production_source_changed=False,temperature_or_power_queried=False)
path=Path('/home/gaob/CADTrack-ABC/profile_output/r2_data_rng_cpu_witness_20261015.json')
assert not path.exists()
path.write_text(json.dumps(result,indent=2))
print(json.dumps({key:value for key,value in result.items() if key!='records'},indent=2),flush=True)
