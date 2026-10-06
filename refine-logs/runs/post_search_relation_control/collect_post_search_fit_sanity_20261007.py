"""Collect four already-closed fit sanities; do not query or replay active NN."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
base = parent / 'GOLA-source/refine-logs/runs/post_search_relation_control'
destination = base / 'closed_fit_sanity_metadata'
assert not destination.exists()
observed = json.loads((base / 'pipeline_observation.json').read_text())
assert observed['owner_alive'] and observed['progress']['stage'] == 'FIT_FULL'
payload = '''import hashlib,json,tarfile
from pathlib import Path
root=Path('/data/gb/outputs/post_search_relation_control_20261007')
arms=[('one_way_lr5',False),('post_search_lr5',True),('one_way_lr3',False),('post_search_lr3',True)]
files=[];summary={}
for name,bidirectional in arms:
 folder=root/'fit_sanity'/name
 done=json.loads((folder/'completion.json').read_text());config=json.loads((folder/'config.json').read_text())
 assert done['completed'] and done['epochs']==2 and done['optimizer_steps']==18
 assert config['train_clips']==2832 and config['validation_clips']==196 and config['observed_pair_training']
 assert config['post_search_bidirectional']==bidirectional and config['initial_checkpoint_weights_exact'] and config['initial_checkpoint_epoch']==5
 assert done['modules_changed']=={'A':True,'B':True,'C':True} and all(v>0 for v in done['max_module_gradient_norms'].values())
 assert done['relation_parameters_changed'] and done['max_relation_gradient_norm']>0
 assert done['strict_reload_metrics_equal'] and done['last_strict_reload_metrics_equal'] and done['frozen_c1_gradients_absent']
 summary[name]=dict(epochs=2,actual_optimizer_updates=18,ABC_relation_gradients_and_updates_reload_verified=True,
                   best_epoch=done['best_epoch'],peak_cuda_mib=done['peak_cuda_mib'],elapsed_seconds=done['elapsed_seconds'])
 files.extend(folder/name for name in ('completion.json','config.json','metrics.json','train.jsonl'))
check=root/'causal_pair_check.json';assert json.loads(check.read_text())['completed'];files.append(check)
archive=Path('/data/gb/setup/post_search_closed_fit_sanity_20261007.tar.gz');assert not archive.exists()
manifest=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
with tarfile.open(archive,'w:gz') as stream:
 for path in files:stream.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
print(json.dumps(dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                     manifest=manifest,arms=summary,total_actual_optimizer_updates=72,scope='closed fit_sanity and causal check only; no active NN/progress/metric query')))
'''
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'], input=payload, text=True, encoding='utf-8', capture_output=True)
(base / 'sanity_intake_stdout.txt').write_text(result.stdout, encoding='utf-8')
(base / 'sanity_intake_stderr.txt').write_text(result.stderr, encoding='utf-8')
result.check_returncode()
receipt = json.loads(result.stdout)
archive = base / Path(receipt['archive']).name
subprocess.run(['scp', '2027:' + receipt['archive'], str(archive)], capture_output=True, check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['archive_sha256']
destination.mkdir()
with tarfile.open(archive) as stream: stream.extractall(destination)
for item in receipt['manifest']:
    raw = (destination / item['path']).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
(base / 'actual_closed_fit_sanity_summary.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: receipt[k] for k in ('arms', 'total_actual_optimizer_updates', 'scope')}), flush=True)
