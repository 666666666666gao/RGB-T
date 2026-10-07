"""Trained projection transfer: four full-video controls and one native policy."""
import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.run_candidate_relation_training import environment, read, REPO, PYTHON, SPLIT, TRAIN_ROOT

PARENT = '/data/gb/outputs/post_search_relation_control_20261007/fit_full/one_way_lr5/best.pth'
PROJECTION = '/data/gb/outputs/identity_representation_probe_20261007/fit_full/encoded_lr4/best.pth'
REFERENCE = '/data/gb/outputs/post_search_relation_control_20261007/full98/one_way_lr5_best/predictions'
ARMS = [('identity_zero', 0.), ('identity_005', .05), ('identity_010', .1), ('identity_020', .2)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    review = read(args.review)
    assert review['status'] == 'PASS' and review['scope'] == 'COMPLETE_IDENTITY_EVIDENCE_PIPELINE_SOURCE'
    assert read('/data/gb/outputs/identity_representation_probe_20261007/progress.json')['stage'] == 'COMPLETE_IDENTITY_REPRESENTATION_PROBE'
    assert Path(PARENT).is_file() and Path(PROJECTION).is_file()
    free = shutil.disk_usage('/data/gb').free
    assert free > 20 * 1024**3
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    values = [tuple(int(part.strip()) for part in line.split(',')) for line in cards.splitlines()]
    assert [row[0] for row in values] == [0, 1, 2, 3] and all(row[1] < 500 and row[2] == 0 for row in values)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def record(stage, **fields):
        result = dict(stage=stage, at_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                      elapsed_seconds=time.perf_counter() - started, **fields)
        (root / 'progress.json').write_text(json.dumps(result, indent=2))
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)

    (root / 'plan.json').write_text(json.dumps(dict(parent_model=PARENT, identity_projection=PROJECTION,
        projection_epoch=9, arms=ARMS, current_stage_new_optimizer_updates=0,
        predecessor_real_epochs=32, predecessor_real_optimizer_updates=896,
        scope='Identity evidence transfer control; original ABC states retained, not new complete ABC retraining',
        search_budget='Five local candidates plus at most five from one executed extra visual forward',
        fixed_first_context_anchor=True, motion_dependency_pre_attention_unchanged=True,
        all_98_sequences_full_frames=True, developer_frames_per_arm=49418,
        native_selection='One dev-selected weight and same complete parent/projection on both full native datasets',
        reused_development=True, formal_TEST_not_used_for_selection=True, disk_free_before_bytes=free), indent=2))

    def wave(stage, commands):
        children, logs = [], []
        for gpu, command in enumerate(commands):
            log = (root / f'{stage}_gpu{gpu}.log').open('w')
            child = subprocess.Popen(command, cwd=REPO, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
            children.append(dict(gpu=gpu, pid=child.pid, command=command))
            logs.append((child, log))
        record(stage.upper(), children=children)
        codes = [child.wait() for child, _ in logs]
        for _, log in logs:
            log.close()
        assert all(code == 0 for code in codes), (stage, codes, 'Read primary logs; no unchanged retry')

    commands = []
    for (name, weight), offset in zip(ARMS, (0, 25, 50, 74)):
        commands.append([PYTHON, '-u', '-m', 'research.check_identity_evidence', '--model', PARENT,
            '--projection', PROJECTION, '--root', TRAIN_ROOT, '--split', SPLIT,
            '--sequence-offset', str(offset), '--weight', str(weight), '--output', str(root / 'sanity' / name)])
    wave('real_sanity', commands)
    assert all(read(root / 'sanity' / name / 'completion.json')['completed'] for name, _ in ARMS)
    record('REAL_GPU_ZERO_PARITY_AND_CAUSALITY_PASS')
    commands = []
    for name, weight in ARMS:
        commands.append([PYTHON, '-u', '-m', 'research.evaluate_recoverability', '--dataset', 'lasher',
            '--root', TRAIN_ROOT, '--validation-split', SPLIT, '--model', PARENT,
            '--identity-projection', PROJECTION, '--identity-weight', str(weight),
            '--search-value', 'gross', '--write-verification', 'action', '--output', str(root / 'full98' / name / 'predictions')])
    wave('full98', commands)
    expected = read(Path(REFERENCE) / 'inference_completion.json')['records']
    for name, weight in ARMS:
        folder = root / 'full98' / name / 'predictions'
        done, config = read(folder / 'inference_completion.json'), read(folder / 'inference_config.json')
        assert done['completed'] and (done['sequences'], done['frames']) == (98, 49418)
        assert config['max_frames'] == config['limit_sequences'] == config['sequence_offset'] == 0
        assert config['identity_projection'] == PROJECTION and config['identity_weight'] == weight
        assert config['model'] == PARENT and config['validation_split'] == SPLIT
        assert [(row['sequence'], row['frames']) for row in done['records']] == [(row['sequence'], row['frames']) for row in expected]
    zero = root / 'full98/identity_zero/predictions'
    for row in expected:
        name = row['sequence']
        assert (zero / (name + '.txt')).read_bytes() == (Path(REFERENCE) / (name + '.txt')).read_bytes(), name
        with np.load(zero / (name + '_recoverability_decisions.npz')) as actual, np.load(Path(REFERENCE) / (name + '_recoverability_decisions.npz')) as before:
            assert all(np.array_equal(actual[key], before[key]) for key in before.files), name
    record('FULL98_ZERO_RESIDUAL_PARENT_PREDICTION_AND_DECISION_PARITY_PASS')
    references = ['/data/gb/outputs/abc_internal_validation_v1/baseline/predictions',
                  '/data/gb/outputs/abc_internal_validation_v1/c1/predictions',
                  '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/predictions',
                  '/data/gb/outputs/recoverability_search_gross_control_20261006/predictions', REFERENCE]
    with (root / 'full98_CPU_report.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'research.collect_recoverability_metrics', '--dataset', 'lasher',
            '--root', TRAIN_ROOT, '--split', SPLIT, '--labels', *[name for name, _ in ARMS],
            '--runs', *[str(root / 'full98' / name / 'predictions') for name, _ in ARMS],
            '--reference-labels', 'baseline', 'c1', 'old4', 'gross_parent', 'current_parent',
            '--references', *references, '--output', str(root / 'full98_report')],
            cwd=REPO, env=environment(''), stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(root / 'full98_report/full_recoverability_report.json')
    selected, weight = max(ARMS, key=lambda arm: report['variants'][arm[0]]['sequence_mean_iou'])
    selection = dict(selected_candidate=selected, parent_model=PARENT, state_commit_model=None,
                     identity_projection=PROJECTION, identity_weight=weight,
                     same_checkpoint_both_native_datasets=True, native_metrics_completed=False,
                     selection_scope='Four fixed residual weights on reused full98, locked before both native TEST sets; not a new seed or complete ABC fit',
                     sequence_mean_iou={name: row['sequence_mean_iou'] for name, row in report['variants'].items()})
    (root / 'selected_model.json').write_text(json.dumps(selection, indent=2))
    record('ONE_COMPLETE_POLICY_LOCKED_BEFORE_NATIVE', **selection)
    with (root / 'native_controller.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'scripts.run_selective_state_native', '--selection', str(root / 'selected_model.json'),
            '--review', args.review, '--output', str(root / 'native')], cwd=REPO, env=environment(''),
            stdout=log, stderr=subprocess.STDOUT, check=True)
    native = read(root / 'native/complete_metrics.json')
    assert native['completed'] and native['same_ABC_model_both_datasets'] == PARENT
    assert native['same_identity_projection_both_datasets'] == PROJECTION and native['same_identity_weight_both_datasets'] == weight
    record('COMPLETE_IDENTITY_EVIDENCE_FULL98_AND_NATIVE', selection=selection, native=native,
           new_optimizer_updates=0, prior_projection_real_optimizer_updates=896, own_weights_created_or_removed=0)


if __name__ == '__main__':
    main()
