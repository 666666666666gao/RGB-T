from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

project = Path('/home/gaob/CADTrack-ABC')
prepared = project / 'prepared/r1_fits_admission_20261015'
manifest = json.loads((prepared / 'source_manifest.json').read_text())
for name, digest in manifest.items():
    assert hashlib.sha256((prepared / name).read_bytes()).hexdigest() == digest
collection = project / 'outputs/native_action_collection_20261014'
launch = json.loads((collection / 'launch.json').read_text())
assert all(not (Path('/proc') / str(job['pid'])).exists() for job in launch['jobs'])
for split in ['train', 'validation']:
    for shard in range(2):
        receipt = json.loads((collection / f'collection_{split}/shard{shard}/collection_complete.json').read_text())
        assert receipt['completed']
baseline = json.loads((project / 'outputs/pure_cad_rgbt234_20261015/pipeline_complete.json').read_text())
assert baseline['completed'] and baseline['inference_exit_codes'] == [0, 0] and baseline['score_exit_code'] == 0
assert not Path('/proc/4027100').exists()
guards = ['baseline.py', 'data.py', 'model.py', 'collect_actions.py', 'future_actions.py',
          'pure_baseline_tracker.py', 'infer_baseline.py', 'baseline_pipeline.py']
before = {name: hashlib.sha256((project / 'cadtrack_abc' / name).read_bytes()).hexdigest() for name in guards}
for name in ['future_pipeline.py', 'train_future_actions.py', 'r1_pair_pipeline.py',
             'select_future_actions.py', 'infer.py', 'score.py']:
    shutil.copyfile(prepared / 'cadtrack_abc' / name, project / 'cadtrack_abc' / name)
    assert (project / 'cadtrack_abc' / name).read_bytes() == (prepared / 'cadtrack_abc' / name).read_bytes()
assert before == {name: hashlib.sha256((project / 'cadtrack_abc' / name).read_bytes()).hexdigest() for name in guards}
python = '/data/wangwj/envs/CADTrack/bin/python'
with (prepared / 'visual_sanity.log').open('w') as stream:
    sanity = subprocess.run([python, str(prepared / 'search_action_visual_sanity_remote.py')],
        cwd=project, stdout=stream, stderr=subprocess.STDOUT, check=True)
with (prepared / 'matched_fit_controller.log').open('w') as stream:
    controller = subprocess.Popen([python, str(prepared / 'run_r1_matched_fits_only_remote.py')],
        cwd=project, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
admission = dict(status='R1_MATCHED_FIT_CONTROLLER_ADMITTED_AFTER_REAL_VISUAL_SANITY',
    observed_utc=datetime.now(timezone.utc).isoformat(), controller_pid=controller.pid,
    controller_command=[python, str(prepared / 'run_r1_matched_fits_only_remote.py')],
    sanity_exit_code=sanity.returncode, original_core_sources_unchanged=True,
    completed_r1_training=False, full_video_and_formal_results_pending=True,
    rgbt210_download_not_required_for_training=True, physical_gpus=[2, 3])
(prepared / 'admission.json').write_text(json.dumps(admission, indent=2))
print(json.dumps(admission, indent=2), flush=True)
