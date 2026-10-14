"""Real TRAIN prefixes against original CAD tracker methods, no official test."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

project = Path('/home/gaob/CADTrack-ABC')
native = Path('/data/wangwj/CADTrack-main')
sys.path.insert(0, str(project))
from cadtrack_abc.pure_baseline_tracker import PureCADTracker
from cadtrack_abc.baseline import cfg
from cadtrack_abc.data import read_pair

assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
torch.cuda.set_device(0)
torch.set_num_threads(4)
torch.manual_seed(42)
output = project / 'outputs/pure_baseline_protocol_sanity_20261015_r2'
output.mkdir(exist_ok=False)
namespace = dict(torch=torch, np=np, math=math, cv=cv2, cv2=cv2, os=os, F=F,
                 BaseTracker=object)
sources = {}
for path, names in [
    ('lib/train/data/processing_utils.py', ['sample_target']),
    ('lib/test/tracker/data_utils.py', ['PreprocessorMM']),
    ('lib/test/utils/hann.py', ['hann1d', 'hann2d']),
    ('lib/utils/box_ops.py', ['clip_box']),
    ('lib/test/tracker/cadtrack.py', ['CADTrack'])]:
    source = (native / path).read_text()
    sources[path] = hashlib.sha256(source.encode()).hexdigest()
    nodes = [node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(native / path), 'exec'), namespace)

class OriginalNetworkAdapter(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    @property
    def box_head(self):
        return self.base.box_head

    def forward(self, template, search, track_query_before=None):
        assert track_query_before is None and len(search) == 1
        result = self.base(template, search[0])
        return [dict(score_map=result['score'], size_map=result['size'],
                     offset_map=result['offset'], track_query_before=None)]

checkpoint = '/home/gaob/CADTrack_ep0014.pth.tar'
tracker = PureCADTracker(checkpoint)
assert tracker.network.loaded_tensors == 250 and tracker.network.loaded_epoch == 14
reference = namespace['CADTrack'].__new__(namespace['CADTrack'])
reference.params = SimpleNamespace(template_factor=cfg.TEST.TEMPLATE_FACTOR,
    template_size=cfg.TEST.TEMPLATE_SIZE, search_factor=cfg.TEST.SEARCH_FACTOR,
    search_size=cfg.TEST.SEARCH_SIZE)
reference.cfg = cfg
reference.network = OriginalNetworkAdapter(tracker.network)
reference.preprocessor = namespace['PreprocessorMM'](cfg.DATA.MEAN, cfg.DATA.STD)
reference.num_template = cfg.DATA.TEMPLATE.NUMBER
reference.feat_sz = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
reference.output_window = namespace['hann2d'](torch.tensor([reference.feat_sz] * 2).long(), centered=True).cuda()
reference.update_intervals = cfg.TEST.UPDATE_INTERVALS
reference.update_threshold = cfg.TEST.UPDATE_THRESHOLD
reference.save_all_boxes = False
reference.debug = 0
assert torch.equal(reference.output_window, tracker.window)
train_names = set(json.loads((project / 'outputs/cad_abc_20261013/manifest.json').read_text())['train_sequences'])
rows = []
with torch.inference_mode():
    for name in ['10crosswhite', '1strowrightgirl']:
        assert name in train_names
        sequence = Path('/data/wangwj/dataset/LasHeR/trainingset') / name
        rgb, tir = sorted((sequence / 'visible').iterdir()), sorted((sequence / 'infrared').iterdir())
        initial = np.loadtxt(sequence / 'init.txt', delimiter=',', ndmin=2)[0].tolist()
        image = read_pair(rgb[0], tir[0])
        tracker.initialize(image, initial)
        reference.initialize(image, {'init_bbox': initial})
        assert all(torch.equal(a, b) for a, b in zip(tracker.templates, reference.z_dict))
        updates = 0
        for frame in range(1, 41):
            image = read_pair(rgb[frame], tir[frame])
            actual = tracker.track(image)
            expected = reference.track(image)
            assert np.array_equal(actual['box'], np.asarray(expected['target_bbox'])), (name, frame, actual['box'], expected)
            assert actual['confidence'] == float(expected['best_score']), (name, frame)
            assert all(torch.equal(a, b) for a, b in zip(tracker.templates, reference.z_dict)), (name, frame, 'templates')
            assert tracker.frame == reference.frame_id == frame
            updates += actual['template_updated']
        rows.append(dict(sequence=name, frames=41, template_updates=updates,
                         boxes_confidence_templates_exact=True))
assert sum(row['template_updates'] for row in rows) > 0
result = dict(status='REAL_TRAIN_NATIVE_PROTOCOL_PARITY_PASS', physical_gpu=2,
    checkpoint=checkpoint, checkpoint_epoch=14, strict_loaded_tensors=250,
    checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    source_files_sha256=sources, rows=rows, neural_tracking_steps=160,
    official_test_frames_used=0, optimizer_updates=0,
    peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
    scope='Original tracker methods with a field-name adapter around the same module-free supplied network; proves test-policy parity, not historical LasHeR score reproduction.')
(output / 'sanity.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
