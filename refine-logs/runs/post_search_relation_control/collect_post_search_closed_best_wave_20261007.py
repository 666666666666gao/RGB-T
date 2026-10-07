"""Intake four closed best-video receipts; compare controls without new inference."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
base = parent / 'GOLA-source/refine-logs/runs/post_search_relation_control'
destination = base / 'closed_best_wave_metadata'
receipt_path = base / 'actual_closed_best_video_wave_intake.json'
assert not destination.exists() and not receipt_path.exists()
snapshot = json.loads((base / 'pipeline_observation.json').read_text())
assert snapshot['owner_pid'] == 1311071 and snapshot['progress']['stage'] == 'FULL98_LAST'
payload = '''import hashlib,json,tarfile
from pathlib import Path
root=Path('/data/gb/outputs/post_search_relation_control_20261007')
reference=Path('/data/gb/outputs/candidate_relation_raw_ABC_20261006_v2/full98/relations_lr5_best/predictions')
ref=json.loads((reference/'inference_completion.json').read_text())
archive=Path('/data/gb/setup/post_search_closed_best_wave_20261007.tar.gz');assert not archive.exists()
files=[];summaries={};identity={}
for name in ('one_way_lr5','post_search_lr5','one_way_lr3','post_search_lr3'):
 folder=root/'full98'/(name+'_best')/'predictions'
 done=json.loads((folder/'inference_completion.json').read_text());config=json.loads((folder/'inference_config.json').read_text())
 assert done['completed'] and (done['sequences'],done['frames'])==(98,49418) and not done['gt_scoring_completed']
 assert config['max_frames']==0 and config['validation_split']=='/data/gb/outputs/c1_initial_seed42/split.json'
 assert config['model']==str(root/'fit_full'/name/'best.pth')
 assert [(r['sequence'],r['frames']) for r in done['records']]==[(r['sequence'],r['frames']) for r in ref['records']]
 summaries[name]={key:value for key,value in done.items() if key!='records'}
 files.extend((folder/'inference_config.json',folder/'inference_completion.json'))
 if name.startswith('one_way'):
  comparisons=[]
  for row in done['records']:
   sequence=row['sequence'];actual=(folder/(sequence+'.txt')).read_bytes();original=(reference/(sequence+'.txt')).read_bytes()
   comparisons.append(dict(sequence=sequence,actual_sha256=hashlib.sha256(actual).hexdigest(),reference_sha256=hashlib.sha256(original).hexdigest(),byte_equal=actual==original))
  identity[name]=dict(comparisons=comparisons,byte_equal_sequences=sum(r['byte_equal'] for r in comparisons),sequences=98)
manifest=[dict(path=p.relative_to(root/'full98').as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
with tarfile.open(archive,'w:gz') as stream:
 for p in files:stream.add(p,arcname=p.relative_to(root/'full98').as_posix(),recursive=False)
print(json.dumps(dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest=manifest,
 inference=summaries,one_way_prediction_identity_to_protected_parent=identity,
 total_completed_sequences=392,total_completed_frames=197672,
 scope='Four best full98 inference runs complete; controls compared as original bytes; no GT metric replay or new native scores')))
'''
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                        input=payload, text=True, encoding='utf-8', capture_output=True)
(base / 'closed_best_wave_intake_stdout.txt').write_text(result.stdout, encoding='utf-8')
(base / 'closed_best_wave_intake_stderr.txt').write_text(result.stderr, encoding='utf-8')
result.check_returncode()
receipt = json.loads(result.stdout)
assert shutil.disk_usage(base).free > sum(row['bytes'] for row in receipt['manifest']) * 2
archive = parent / Path(receipt['archive']).name; assert not archive.exists()
subprocess.run(['scp', '2027:' + receipt['archive'], str(archive)], capture_output=True, check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['archive_sha256']
destination.mkdir()
with tarfile.open(archive) as stream: stream.extractall(destination)
for row in receipt['manifest']:
    data = (destination / row['path']).read_bytes()
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
receipt['original_observed_successor'] = snapshot
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(completed_sequences=392,completed_frames=197672,
 one_way_byte_equal_sequences={name:value['byte_equal_sequences'] for name,value in receipt['one_way_prediction_identity_to_protected_parent'].items()},
 native_completed=False)), flush=True)
