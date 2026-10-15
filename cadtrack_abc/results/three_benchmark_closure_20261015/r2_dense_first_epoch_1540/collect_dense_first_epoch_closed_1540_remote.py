from pathlib import Path
import base64
import hashlib
import json

root = Path('/home/gaob/CADTrack-ABC')
names = ['outputs/r2_dense_pair_20261015/first_epoch_1540_observation.json',
         'outputs/r2_dense_pair_20261015/H/fit/manifest.json',
         'outputs/r2_dense_pair_20261015/HA/fit/manifest.json']
result = {}
for name in names:
    value = (root/name).read_bytes()
    result[name] = dict(base64=base64.b64encode(value).decode(), bytes=len(value),
                        sha256=hashlib.sha256(value).hexdigest())
print(json.dumps(result))
