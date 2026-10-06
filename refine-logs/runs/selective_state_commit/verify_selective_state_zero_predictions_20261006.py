"""Compare completed full98 prediction text with the frozen parent, once."""
import json
import subprocess
from pathlib import Path
base=Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source/refine-logs/runs/selective_state_commit')
payload='''import json
from pathlib import Path
root=Path('/data/gb/outputs/selective_state_training_endpoint_20261006')
parent=Path('/data/gb/outputs/recoverability_search_gross_control_20261006/predictions')
names={p.name for p in parent.glob('*.txt')}
assert len(names)==98
results=[]
for label in ('H3_lost0_best','H3_lost01_best'):
 folder=root/'full98'/label/'predictions'
 done=json.loads((folder/'inference_completion.json').read_text())
 assert done['completed'] and (done['sequences'],done['frames'])==(98,49418)
 assert {p.name for p in folder.glob('*.txt')}==names
 different=[name for name in sorted(names) if (folder/name).read_bytes()!=(parent/name).read_bytes()]
 assert not different
 results.append({'variant':label,'prediction_files':98,'different_prediction_files':different,
                 'all_prediction_text_bytes_equal_parent':True})
print(json.dumps({'status':'ZERO_HEAD_FULL98_PREDICTION_TEXT_PARITY_PASS','results':results,
 'NN_replays':0,'active_process_GPU_queries':0}))
'''
run=subprocess.run(['ssh','-T','2027','/data/gb/envs/gola/bin/python -'],input=payload,
                   text=True,encoding='utf-8',capture_output=True)
(base/'zero_prediction_parity_stdout.txt').write_text(run.stdout,encoding='utf-8')
(base/'zero_prediction_parity_stderr.txt').write_text(run.stderr,encoding='utf-8')
run.check_returncode()
result=json.loads(run.stdout)
(base/'zero_prediction_parity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
