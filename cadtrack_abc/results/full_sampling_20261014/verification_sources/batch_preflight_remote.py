from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import subprocess

project = Path('/home/gaob/CADTrack-ABC')
parent = project / 'outputs/cad_abc_20261013'
assert (parent / 'pipeline_complete.json').is_file()
assert not Path('/proc/486671').exists()
manifest = json.loads((parent / 'manifest.json').read_text())
assert manifest['physical_gpus'] == '2,3' and manifest['baseline_rerun'] is False
report = json.loads((parent / 'lasher_report/full_report.json').read_text())
assert report['completed'] and report['sequences'] == 245 and report['frames'] == 220703
counts = {}
for split, key in [('train', 'train_sequences'), ('validation', 'validation_sequences')]:
    lengths = {name: len((Path(manifest['train_root']) / name / 'init.txt').read_text().splitlines()) for name in manifest[key]}
    counts[split] = {'sequences': len(lengths), 'frames': sum(lengths.values()),
                     'shard_frames': [sum(lengths[name] for name in sorted(lengths)[shard::2]) for shard in range(2)]}
gpu = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=index,name,memory.used,memory.total',
                      '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
rows = [row.split(',') for row in gpu.splitlines()]
assert [int(row[0]) for row in rows] == [2, 3]
assert all(int(row[2]) < 500 for row in rows), gpu
own = []
for process in Path('/proc').iterdir():
    command = process / 'cmdline'
    if process.name.isdecimal() and command.exists():
        raw = command.read_bytes()
        if b'cadtrack_abc.' in raw and b'python' in raw and b'python3 -' not in raw:
            own.append({'pid': process.name, 'command': raw.replace(b'\x00', b' ').decode()})
assert not own, own
state = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'gpus_memory_only': gpu,
         'split_frame_inventory': counts, 'python': '/data/wangwj/envs/CADTrack/bin/python',
         'packages': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'numpy', 'timm']},
         'home_free_bytes': shutil.disk_usage(project).free,
         'teacher_checkpoint': str(parent / 'best.pth'),
         'teacher_sha256': hashlib.sha256((parent / 'best.pth').read_bytes()).hexdigest(),
         'live_own_neural_processes': own, 'baseline_rerun': False,
         'model_executed': False, 'cuda_initialized': False}
assert state['teacher_sha256'] == '5c6b6fcbf18b11918ae3495ccfa927cf56a3f5c7f3cecd76aabaaa94d80cc988'
print(json.dumps(state))
