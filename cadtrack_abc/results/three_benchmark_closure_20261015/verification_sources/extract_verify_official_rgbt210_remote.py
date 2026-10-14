from datetime import datetime, timezone
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys

import numpy as np

project = Path('/home/gaob/CADTrack-ABC')
dataset = project / 'dataset'
receipt = json.loads((dataset / 'rgbt210_download_complete.json').read_text())
assert receipt['status'] == 'OFFICIAL_RGBT210_DOWNLOAD_COMPLETE'
archive = dataset / 'RGBT210.rar'
assert archive.stat().st_size == receipt['expected_bytes'] == 6857321678
tool = project / 'tools/7zip26.04/7zzs'
assert tool.is_file()
listing = subprocess.run([str(tool), 'l', '-slt', '-ba', str(archive)], capture_output=True, text=True, check=True).stdout
paths = [Path(value.replace('\\', '/')) for value in re.findall(r'^Path = (.+)$', listing, re.MULTILINE)]
assert paths and all(not path.is_absolute() and '..' not in path.parts for path in paths)
expanded = sum(int(value) for value in re.findall(r'^Size = (\d+)$', listing, re.MULTILINE))
assert shutil.disk_usage(dataset).free > expanded + 1024**3
sys.path.insert(0, '/data/wangwj/RGBT_toolkit-master/src')
from rgbt import RGBT210
official = RGBT210()
names = set(official.ALL)
assert len(names) == 210
prefixes = {path.parent.parent for path in paths if path.name == 'init.txt' and path.parent.name in names}
assert len(prefixes) == 1, list(map(str, prefixes))
destination = dataset / 'rgbt210'
destination.mkdir(exist_ok=False)
output = project / 'outputs/official_rgbt210_preparation_20261015'
output.mkdir(exist_ok=False)
(output / 'archive_listing.txt').write_text(listing)
with (output / 'extraction.log').open('w') as stream:
    subprocess.run([str(tool), 'x', str(archive), '-o' + str(destination), '-y', '-bsp0'], stdout=stream,
                   stderr=subprocess.STDOUT, check=True)
root = destination / next(iter(prefixes))
assert {path.name for path in root.iterdir() if path.is_dir()} == names
records = []
for name in sorted(names):
    sequence = root / name
    labels = np.loadtxt(sequence / 'init.txt', delimiter=',', ndmin=2)
    rgb = sorted((sequence / 'visible').iterdir())
    tir = sorted((sequence / 'infrared').iterdir())
    assert labels.shape == (len(rgb), 4) and len(rgb) == len(tir), name
    assert np.isfinite(labels).all() and (labels[0, 2:] > 0).all(), name
    installed = np.asarray(official.seqs_gt[name]['visible'])
    same_gt = labels.shape == installed.shape and np.array_equal(labels, installed)
    records.append(dict(sequence=name, frames=len(rgb), installed_gt_frames=len(installed),
                        same_as_installed_toolkit_gt=bool(same_gt)))
result = dict(completed=True, observed_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
              sequences=len(records), frames=sum(row['frames'] for row in records), records=records,
              installed_toolkit_gt_frames=sum(row['installed_gt_frames'] for row in records),
              installed_toolkit_gt_differing_sequences=[row['sequence'] for row in records if not row['same_as_installed_toolkit_gt']],
              actual_dataset_gt_used_for_future_scoring=True, archive_bytes=archive.stat().st_size,
              archive_sha256=receipt['sha256'], expanded_file_bytes=expanded,
              source_index='https://github.com/ahucv/Homepage/blob/master/public/source/dataset.md',
              source_url=receipt['url'], existing_datasets_moved=False, neural_inference_started=False,
              warm_conda_modified=False, temperature_or_power_queried=False)
(dataset / 'rgbt210_complete.json').write_text(json.dumps(result, indent=2))
(output / 'inventory.json').write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2), flush=True)
