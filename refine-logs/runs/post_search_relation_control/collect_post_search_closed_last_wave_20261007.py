"""Collect closed endpoint inference evidence once, without replay or GT scoring."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
base = parent / 'GOLA-source/refine-logs/runs/post_search_relation_control'
destination = base / 'closed_last_wave_metadata'
receipt_path = base / 'actual_closed_last_video_wave_intake.json'
assert not destination.exists() and not receipt_path.exists()
snapshot = json.loads((base / 'pipeline_observation.json').read_text())
assert snapshot['owner_pid'] == 1311071 and snapshot['progress']['stage'] == 'FULL98_POST_SEARCH_INITIAL'
payload = '''import hashlib,json,tarfile
from pathlib import Path
root=Path('/data/gb/outputs/post_search_relation_control_20261007')
archive=Path('/data/gb/setup/post_search_closed_last_wave_20261007.tar.gz');assert not archive.exists()
files=[];summaries={};identity={}
for name in ('one_way_lr5','post_search_lr5','one_way_lr3','post_search_lr3'):
 folder=root/'full98'/(name+'_last')/'predictions';reference=root/'full98'/(name+'_best')/'predictions'
 done=json.loads((folder/'inference_completion.json').read_text());config=json.loads((folder/'inference_config.json').read_text())
 ref=json.loads((reference/'inference_completion.json').read_text())
 assert done['completed'] and (done['sequences'],done['frames'])==(98,49418) and not done['gt_scoring_completed']
 assert config['max_frames']==0 and config['validation_split']=='/data/gb/outputs/c1_initial_seed42/split.json'
 assert config['model']==str(root/'fit_full'/name/'last.pth')
 assert [(r['sequence'],r['frames']) for r in done['records']]==[(r['sequence'],r['frames']) for r in ref['records']]
 summaries[name]={key:value for key,value in done.items() if key!='records'}
 files.extend((folder/'inference_config.json',folder/'inference_completion.json'));comparisons=[]
 for row in done['records']:
  sequence=row['sequence'];actual=(folder/(sequence+'.txt')).read_bytes();best=(reference/(sequence+'.txt')).read_bytes()
  comparisons.append(dict(sequence=sequence,last_sha256=hashlib.sha256(actual).hexdigest(),best_sha256=hashlib.sha256(best).hexdigest(),byte_equal=actual==best))
 identity[name]=dict(comparisons=comparisons,byte_equal_sequences=sum(r['byte_equal'] for r in comparisons),sequences=98)
manifest=[dict(path=p.relative_to(root/'full98').as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
with tarfile.open(archive,'w:gz') as stream:
 for p in files:stream.add(p,arcname=p.relative_to(root/'full98').as_posix(),recursive=False)
print(json.dumps(dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest=manifest,
 inference=summaries,last_prediction_identity_to_best=identity,total_completed_sequences=392,total_completed_frames=197672,
 scope='Four endpoint full98 inference runs complete; original bytes compared to best; no GT metric replay or new native scores')))
'''
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                        input=payload, text=True, encoding='utf-8', capture_output=True)
(base / 'closed_last_wave_intake_stdout.txt').write_text(result.stdout, encoding='utf-8')
(base / 'closed_last_wave_intake_stderr.txt').write_text(result.stderr, encoding='utf-8')
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
 last_best_byte_equal_sequences={name:value['byte_equal_sequences'] for name,value in receipt['last_prediction_identity_to_best'].items()},
 initial_and_GT_report_and_native_complete=False)), flush=True)
