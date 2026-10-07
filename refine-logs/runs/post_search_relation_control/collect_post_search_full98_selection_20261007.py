"""Intake the closed nine-candidate developer report and pre-TEST selection once."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
base = parent / 'GOLA-source/refine-logs/runs/post_search_relation_control'
destination = base / 'closed_full98_and_selection'
receipt_path = base / 'actual_full98_selection_intake.json'
assert not destination.exists() and not receipt_path.exists()
snapshot = json.loads((base / 'pipeline_observation.json').read_text())
assert snapshot['owner_pid'] == 1311071 and snapshot['progress']['stage'] == 'ONE_MODEL_LOCKED_BEFORE_NATIVE'
payload = '''import hashlib,json,tarfile
from pathlib import Path
root=Path('/data/gb/outputs/post_search_relation_control_20261007')
archive=Path('/data/gb/setup/post_search_full98_selection_text_20261007.tar.gz');assert not archive.exists()
report=json.loads((root/'full98_report/full_recoverability_report.json').read_text())
selection=json.loads((root/'selected_model.json').read_text())
assert report['completed'] and report['sequences']==98 and not report['native_accuracy_completed']
assert report['args']['root']=='/data/wangwj/dataset/LasHeR/traingset' and report['args']['split']=='/data/gb/outputs/c1_initial_seed42/split.json'
candidates=report['args']['labels'];assert len(candidates)==9 and 'post_search_initial' in candidates
assert len(report['variants'])==14 and len(report['paired'])==45
assert all(v['frames']==49418 for v in report['variants'].values())
assert all(p['bootstrap_resamples']==5000 for p in report['paired'].values())
assert selection['same_checkpoint_both_native_datasets'] and not selection['native_metrics_completed']
assert selection['selected_candidate'] in candidates and selection['state_commit_model'] is None
assert report['variants'][selection['selected_candidate']]['sequence_mean_iou']==max(report['variants'][c]['sequence_mean_iou'] for c in candidates)
assert selection['sequence_mean_iou']=={c:v['sequence_mean_iou'] for c,v in report['variants'].items()}
files=[root/'selected_model.json']+sorted(p for p in (root/'full98_report').rglob('*') if p.is_file())
initial=root/'full98/post_search_initial/predictions'
done=json.loads((initial/'inference_completion.json').read_text());config=json.loads((initial/'inference_config.json').read_text())
assert done['completed'] and done['full_shard_union_verified'] and (done['sequences'],done['frames'])==(98,49418)
assert config['max_frames']==0 and config['model']==str(root/'fit_full/post_search_lr5/initial.pth')
files.extend((initial/'inference_completion.json',initial/'inference_config.json'))
shards=[]
for gpu,offset,count in ((0,0,25),(1,25,25),(2,50,24),(3,74,24)):
 folder=root/'initial_shards'/('gpu'+str(gpu))
 shard=json.loads((folder/'inference_completion.json').read_text());cfg=json.loads((folder/'inference_config.json').read_text())
 assert shard['completed'] and shard['sequences']==count and cfg['sequence_offset']==offset and cfg['limit_sequences']==count and cfg['max_frames']==0
 assert cfg['model']==str(root/'fit_full/post_search_lr5/initial.pth')
 shards.append(dict(gpu=gpu,sequences=count,frames=shard['frames'],wall_seconds=shard['wall_seconds_including_initialization_and_diagnostic_serialization']))
 files.extend((folder/'inference_completion.json',folder/'inference_config.json'))
assert sum(r['frames'] for r in shards)==49418
manifest=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
with tarfile.open(archive,'w:gz') as stream:
 for p in files:stream.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
print(json.dumps(dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest=manifest,
 full98_sequence_mean_iou=selection['sequence_mean_iou'],selection=selection,initial_shards=shards,
 paired=report['paired'],full98_complete=True,native_complete=False,
 scope='Nine original full98 developer runs and actual-GT report, pre-TEST one-checkpoint selection; not native benchmark metrics')))
'''
result = subprocess.run(['ssh', '-T', '2027', '/data/gb/envs/gola/bin/python -'],
                        input=payload, text=True, encoding='utf-8', capture_output=True)
(base / 'full98_selection_intake_stdout.txt').write_text(result.stdout, encoding='utf-8')
(base / 'full98_selection_intake_stderr.txt').write_text(result.stderr, encoding='utf-8')
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
receipt['original_native_observation'] = snapshot
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(full98_complete=True,selected_candidate=receipt['selection']['selected_candidate'],
 selected_epoch=receipt['selection']['selected_epoch'],sequence_mean_iou=receipt['full98_sequence_mean_iou'],
 native_complete=False)), flush=True)
