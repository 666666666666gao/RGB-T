"""Bounded actual-input diagnosis for the observed first-frame parity failure."""
from pathlib import Path
import ast
import json
import math
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
torch.set_num_threads(4)
scope = dict(torch=torch, np=np, math=math, cv=cv2, F=F)
for path, name in [('lib/train/data/processing_utils.py', 'sample_target'),
                   ('lib/test/tracker/data_utils.py', 'PreprocessorMM')]:
    node = next(node for node in ast.parse((native / path).read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), scope)
tracker = PureCADTracker('/home/gaob/CADTrack_ep0014.pth.tar')
processor = scope['PreprocessorMM'](cfg.DATA.MEAN, cfg.DATA.STD)
sequence = Path('/data/wangwj/dataset/LasHeR/trainingset/10crosswhite')
rgb, tir = sorted((sequence / 'visible').iterdir()), sorted((sequence / 'infrared').iterdir())
initial = np.loadtxt(sequence / 'init.txt', delimiter=',', ndmin=2)[0].tolist()
with torch.inference_mode():
    tracker.initialize(read_pair(rgb[0], tir[0]), initial)
    image = read_pair(rgb[1], tir[1])
    own, own_factor = tracker.crop(image, tracker.box, tracker.search_factor, tracker.search_size)
    patch, native_factor, _ = scope['sample_target'](image, initial, tracker.search_factor,
                                                   output_sz=tracker.search_size)
    reference = processor.process(patch)
    outputs = [tracker.network(tracker.templates, own),
               tracker.network(tracker.templates, reference),
               tracker.network(tracker.templates, own)]
    rows = []
    for a, b in [(0, 1), (0, 2)]:
        rows.append(dict(pair=[a, b], fields={key: dict(equal=torch.equal(outputs[a][key], outputs[b][key]),
            max_abs_error=float((outputs[a][key] - outputs[b][key]).abs().max()))
            for key in ['score', 'size', 'offset', 'fused', 'modality']}))
    result = dict(scope='Two actual TRAIN images, three fixed-network forwards, no optimizer or official test.',
        own_resize_factor=own_factor, native_resize_factor=native_factor,
        inputs_equal=torch.equal(own, reference), input_max_abs_error=float((own-reference).abs().max()),
        input_mismatched_elements=int(torch.count_nonzero(own != reference)),
        own_strides=list(own.stride()), native_strides=list(reference.stride()),
        own_channels_last=own.is_contiguous(memory_format=torch.channels_last),
        native_channels_last=reference.is_contiguous(memory_format=torch.channels_last),
        forward_comparisons=rows, all_modules_evaluation=all(not module.training for module in tracker.network.modules()),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic)
(project / 'outputs/pure_baseline_protocol_sanity_20261015/parity_diagnosis.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
