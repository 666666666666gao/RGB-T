from pathlib import Path
import hashlib
import json

project=Path('/home/gaob/CADTrack-ABC')
root=project/'outputs/r1_control_reports_20261015'
state=json.loads((root/'recovery_1308_state.json').read_text())
exit_receipt=json.loads((root/'recovery_1308_exit.json').read_text())
assert state['status']=='CONTROL_R1_THREE_BENCHMARKS_COMPLETE'
assert exit_receipt['controller_exit_code']==0
files={}
for name in ['recovery_1308_state.json','recovery_1308_exit.json','observation_resume_1350.json','observation_resume_1355.json']:
    files[name]=(root/name).read_text()
for dataset,count,frames in [('lasher',245,220703),('rgbt234',234,116649),('rgbt210',210,104706)]:
    path=root/'selected'/f'{dataset}_report'/'full_report.json'
    report=json.loads(path.read_text())
    assert report['completed'] and report['sequences']==count and report['frames']==frames
    assert report['epoch']==20 and report['checkpoint']==state['checkpoint']
    files[f'{dataset}_full_report.json']=path.read_text()
    for shard in (0,1):
        files[f'{dataset}_complete_shard{shard}.json']=(root/'selected'/f'{dataset}_predictions'/f'complete_shard{shard}.json').read_text()
files['selection.json']=(project/'outputs/r1_matched_fits_20261015/fit_future_actions/selection.json').read_text()
assert hashlib.sha256(Path(state['checkpoint']).read_bytes()).hexdigest()==state['checkpoint_sha256']
print(json.dumps(dict(files=files,source_root=str(root),actual_exit=exit_receipt,checkpoint_sha256=state['checkpoint_sha256'])),flush=True)
