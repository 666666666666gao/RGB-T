"""Intake the three already exited full fits once, without NN or metric replay."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
base = parent / 'GOLA-source/refine-logs/runs/post_search_relation_control'
destination = base / 'closed_three_full_fit_metadata'
receipt_path = base / 'actual_three_closed_full_fit_intake.json'
assert not destination.exists() and not receipt_path.exists()
snapshot = json.loads((base / 'pipeline_observation.json').read_text())
assert snapshot['owner_pid'] == 1311071 and snapshot['progress']['stage'] == 'FIT_FULL'
assert all(not snapshot['children_alive'][gpu]['alive'] for gpu in (0, 2, 3))
payload = '''import hashlib,json,tarfile
from pathlib import Path
root=Path('/data/gb/outputs/post_search_relation_control_20261007')
archive=Path('/data/gb/setup/post_search_three_closed_full_fits_20261007.tar.gz')
assert not archive.exists()
files=[];fits={}
for name in ('one_way_lr5','one_way_lr3','post_search_lr3'):
 folder=root/'fit_full'/name
 done=json.loads((folder/'completion.json').read_text())
 config=json.loads((folder/'config.json').read_text())
 history=json.loads((folder/'metrics.json').read_text())
 assert done['completed'] and done['epochs']==64 and done['optimizer_steps']==576
 assert done['modules_changed']=={'A':True,'B':True,'C':True}
 assert done['relation_parameters_changed'] and done['max_relation_gradient_norm']>0
 assert all(v>0 for v in done['max_module_gradient_norms'].values())
 assert done['strict_reload_metrics_equal'] and done['last_strict_reload_metrics_equal']
 assert done['frozen_c1_gradients_absent'] and config['initial_checkpoint_weights_exact']
 assert config['train_clips']==2832 and config['validation_clips']==196
 assert [row['epoch'] for row in history]==list(range(65))
 fits[name]=dict(completion=done,initial_cached_validation=history[0],
  best_cached_validation=history[done['best_epoch']],last_cached_validation=history[-1])
 files.extend(folder/name for name in ('config.json','completion.json','metrics.json','train.jsonl'))
manifest=[dict(path=p.relative_to(root/'fit_full').as_posix(),bytes=p.stat().st_size,
 sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
with tarfile.open(archive,'w:gz') as stream:
 for p in files:stream.add(p,arcname=p.relative_to(root/'fit_full').as_posix(),recursive=False)
print(json.dumps(dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
 manifest=manifest,full_fits=fits,total_actual_optimizer_updates=1728,
 fourth_full_fit_complete=False,full98_complete=False,native_complete=False,
 scope='Three closed 64-epoch full-fit receipts; cached validation is not whole-video or native accuracy')))
'''
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                        input=payload, text=True, encoding='utf-8', capture_output=True)
(base / 'three_closed_full_fit_intake_stdout.txt').write_text(result.stdout, encoding='utf-8')
(base / 'three_closed_full_fit_intake_stderr.txt').write_text(result.stderr, encoding='utf-8')
result.check_returncode()
receipt = json.loads(result.stdout)
assert shutil.disk_usage(base).free > sum(row['bytes'] for row in receipt['manifest']) * 2
archive = parent / Path(receipt['archive']).name
assert not archive.exists()
subprocess.run(['scp', '2027:' + receipt['archive'], str(archive)], capture_output=True, check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['archive_sha256']
destination.mkdir()
with tarfile.open(archive) as stream: stream.extractall(destination)
for row in receipt['manifest']:
    data = (destination / row['path']).read_bytes()
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(total_actual_optimizer_updates=1728,
 best_epochs={name: value['completion']['best_epoch'] for name, value in receipt['full_fits'].items()},
 fourth_full_fit_complete=False, full98_complete=False, native_complete=False)), flush=True)
