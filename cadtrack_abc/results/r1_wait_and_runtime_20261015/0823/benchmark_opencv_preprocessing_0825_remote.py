from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
project = Path('/home/gaob/CADTrack-ABC')
sys.path.insert(0, str(project))
import cv2
import numpy as np
import torch
from cadtrack_abc.data import read_pair, crop_batch

torch.set_num_threads(4)
names = json.loads((project / 'outputs/cad_abc_20261013/manifest.json').read_text())['train_sequences']
assert len(names) == 881 and cv2.getNumThreads() == 112
inputs, reference, inventory = [], [], []
for index in (0, 293, 586, 880):
    name = names[index]
    directory = Path('/data/wangwj/dataset/LasHeR/trainingset') / name
    rgb, tir = sorted((directory / 'visible').iterdir())[0], sorted((directory / 'infrared').iterdir())[0]
    with (directory / 'init.txt').open() as stream:
        box = np.fromstring(stream.readline(), sep=',', dtype=np.float32)
    assert box.shape == (4,) and (box[2:] > 0).all()
    box_tensor = torch.tensor(box[None])
    image = read_pair(rgb, tir)
    patch, transform = crop_batch([image], box_tensor, 4., 256, 'cpu')
    inputs.append((name, rgb, tir, box_tensor))
    reference.append((image, patch, transform))
    inventory.append(dict(sequence=name, paired_shape=list(image.shape),
        raw_image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        cpu_normalized_crop_sha256=hashlib.sha256(patch.numpy().tobytes()).hexdigest()))
started = datetime.now(timezone.utc).isoformat()
records = []
for threads in (112, 4, 1, 1, 4, 112):
    cv2.setNumThreads(threads)
    assert cv2.getNumThreads() == threads
    seconds = 0.
    for repeat in range(8):
        for index, (name, rgb, tir, box) in enumerate(inputs):
            begin = time.perf_counter()
            image = read_pair(rgb, tir)
            patch, transform = crop_batch([image], box, 4., 256, 'cpu')
            seconds += time.perf_counter() - begin
            expected = reference[index]
            assert np.array_equal(image, expected[0])
            assert torch.equal(patch, expected[1]) and torch.equal(transform, expected[2])
    records.append(dict(opencv_threads=threads, frames=32, seconds=seconds,
        preprocessing_frames_per_second=32 / seconds, paired_pixels_and_crops_exact=True))
summaries = {threads: dict(frames=64,
    seconds=sum(row['seconds'] for row in records if row['opencv_threads'] == threads)) for threads in (112, 4, 1)}
result = dict(status='CPU_PREPROCESSING_BENCHMARK_COMPLETE', started_utc=started,
    observed_utc=datetime.now(timezone.utc).isoformat(), inventory=inventory, records=records,
    summary=summaries, speedup_vs_default112={str(threads): summaries[112]['seconds'] / summaries[threads]['seconds']
        for threads in (4, 1)}, causal_inference_claim=False,
    scope='Actual TRAIN first-frame decoding/cropping CPU-only; no network, GPU, official test or live-process changes.',
    new_neural_jobs_started=False, model_or_environment_modified=False)
(project / 'profile_output/opencv_preprocessing_benchmark_20261015_0825.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
