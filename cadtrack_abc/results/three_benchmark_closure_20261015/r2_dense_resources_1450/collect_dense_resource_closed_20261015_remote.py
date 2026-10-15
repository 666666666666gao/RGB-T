from pathlib import Path
import base64
import hashlib
import json

root = Path('/home/gaob/CADTrack-ABC')
names = ['outputs/r2_dense_pair_20261015/first600_observation.json',
         'outputs/r2_dense_pair_20261015/followup1442_observation.json',
         'profile_output/dense_resources_requested_20261015/resource_report.json',
         'profile_output/dense_resources_requested_20261015/gpu_samples.csv']
result = {}
for name in names:
    value = (root / name).read_bytes()
    result[name] = dict(base64=base64.b64encode(value).decode(),
                        sha256=hashlib.sha256(value).hexdigest(), bytes=len(value))
print(json.dumps(result))
