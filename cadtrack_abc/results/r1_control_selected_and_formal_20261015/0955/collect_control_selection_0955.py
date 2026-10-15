from pathlib import Path
import io
import json
import subprocess
import tarfile

private = Path(__file__).parent
destination = private / 'control_selection_inputs_0955'
destination.mkdir(exist_ok=False)
names = ['outputs/r1_matched_fits_20261015/fit_future_actions/selection.json',
    'outputs/r1_matched_fits_20261015/fit_future_actions/fit_complete.json']
for epoch in (0, 1, 2, 5, 10, 20):
    names += [f'outputs/r1_control_full98_20261015/candidate_epoch{epoch:02d}/development_shard{shard}.json'
        for shard in (0, 1)]
code = '''from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path('/home/gaob/CADTrack-ABC')
names=NAMES
selection=json.loads((root/names[0]).read_bytes())
assert selection['full98_development_selection_complete']
assert selection['selected']['epoch']==selection['trained_selected']['epoch']==20
best=Path(selection['best_checkpoint'])
candidate=Path(selection['selected']['checkpoint'])
digest=hashlib.sha256(best.read_bytes()).hexdigest()
assert digest==hashlib.sha256(candidate.read_bytes()).hexdigest()
assert selection['best_checkpoint']==selection['trained_best_checkpoint']
meta=json.dumps(dict(selected_epoch=20,best_checkpoint=str(best),candidate=str(candidate),
    best_sha256=digest,best_bytes=best.stat().st_size,selected_and_positive_best_same=True,
    neural_jobs_queried=False,checkpoint_modified=False)).encode()
data=io.BytesIO()
with tarfile.open(fileobj=data,mode='w:gz') as archive:
    for name in names: archive.add(root/name,arcname=name)
    info=tarfile.TarInfo('control_best_sha256.json'); info.size=len(meta)
    archive.addfile(info,io.BytesIO(meta))
sys.stdout.buffer.write(data.getvalue())
'''.replace('names=NAMES', 'names=' + repr(names))
response = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
    '-o', 'ServerAliveInterval=60', '-o', 'ConnectTimeout=10', '2028', 'python3 -'],
    input=code.encode(), capture_output=True, check=True)
with tarfile.open(fileobj=io.BytesIO(response.stdout), mode='r:gz') as archive:
    assert set(archive.getnames()) == set(names + ['control_best_sha256.json'])
    for member in archive.getmembers():
        assert member.isfile()
        target = destination / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())
print((destination / 'control_best_sha256.json').read_text())
