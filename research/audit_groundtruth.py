"""Verify the unchanged evaluator's GT against the two existing datasets."""
import json
from pathlib import Path

import numpy as np
from rgbt import LasHeR, RGBT234


def main():
    report = {}
    for name, evaluator, root, files in (
        ('lasher', LasHeR(), Path('/data/wangwj/dataset/LasHeR/testingset'), {'target': 'init.txt'}),
        ('rgbt234', RGBT234(), Path('/data/zhouy/DATASET/RGB-T234'),
         {'visible': 'visible.txt', 'infrared': 'infrared.txt'}),
    ):
        actual_sequences = {p.name for p in root.iterdir() if p.is_dir()}
        official_sequences = set(evaluator.ALL)
        differences = []
        frames = {modality: 0 for modality in files}
        for sequence in evaluator.ALL:
            for modality, filename in files.items():
                actual = np.loadtxt(root / sequence / filename, delimiter=',').astype(np.float32)
                packaged = (evaluator.seqs_gt[sequence] if name == 'lasher'
                            else evaluator.seqs_gt[sequence][modality])
                packaged = np.asarray(packaged, dtype=np.float32)
                frames[modality] += len(actual)
                if actual.shape != packaged.shape:
                    differences.append({'sequence': sequence, 'modality': modality,
                                        'actual_shape': list(actual.shape), 'official_shape': list(packaged.shape)})
                elif not np.array_equal(actual, packaged, equal_nan=True):
                    differences.append({'sequence': sequence, 'modality': modality,
                                        'max_absolute_difference_pixels': float(np.nanmax(np.abs(actual - packaged)))})
        report[name] = {'sequences': len(official_sequences), 'actual_annotation_frames': frames,
                        'sequence_names_match': actual_sequences == official_sequences,
                        'different_annotation_files': differences,
                        'annotations_match': not differences and actual_sequences == official_sequences,
                        'comparison_precision': 'float32, matching official rgbt GT loader'}
    Path('/data/gb/setup/official_groundtruth_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
