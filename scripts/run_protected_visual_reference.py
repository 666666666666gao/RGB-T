"""Four state-propagation controls, real sanity, full98, one complete native policy."""
import argparse
import io
import json
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scripts.run_candidate_relation_training import environment, read, REPO, PYTHON, SPLIT, TRAIN_ROOT

PARENT = '/data/gb/outputs/post_search_relation_control_20261007/fit_full/one_way_lr5/best.pth'
PARENT_PREDICTIONS = '/data/gb/outputs/post_search_relation_control_20261007/full98/one_way_lr5_best/predictions'
BASELINE = '/data/gb/outputs/abc_internal_validation_v1/baseline/predictions'
MODES = ('parent', 'geometry', 'visual', 'visual_motion')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    review = read(args.review)
    assert review['status'] == 'PASS' and review['scope'] == 'COMPLETE_PROTECTED_REFERENCE_PIPELINE_SOURCE'
    assert read('/data/gb/outputs/identity_evidence_transfer_20261007/progress.json')['stage'] == 'COMPLETE_IDENTITY_EVIDENCE_FULL98_AND_NATIVE'
    free = shutil.disk_usage('/data/gb').free; assert free > 20 * 1024**3
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    values = [tuple(int(part.strip()) for part in line.split(',')) for line in cards.splitlines()]
    assert [row[0] for row in values] == [0, 1, 2, 3] and all(row[1] < 500 and row[2] == 0 for row in values)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def record(stage, **fields):
        result = dict(stage=stage, at_cst=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                      elapsed_seconds=time.perf_counter() - started, **fields)
        (root / 'progress.json').write_text(json.dumps(result, indent=2))
        with (root / 'events.jsonl').open('a') as stream: stream.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)

    (root / 'plan.json').write_text(json.dumps(dict(parent_model=PARENT, modes=MODES, new_optimizer_updates=0,
        scope='Causal state-propagation controls; original A/B/C active, no new training gain claimed',
        shared_visual_budget='Five primary plus at most five from ONE actually executed extra region',
        added_reference_state='One bounded reference branch with online template; private ABC branch retained',
        geometry_only_not_native_GOLA=True, protected_reference_is_prediction_not_GT=True,
        visual_motion_changes_both_motion_history_and_motion_memory=True,
        private_ABC_template_in_visual_modes_not_used_by_primary_or_anchor_extra=True,
        selection='Highest full98 sequence mean IoU, parent wins ties; same preselected policy both formal datasets',
        reused_development=True, formal_TEST_not_used_for_selection=True, disk_free_before_bytes=free), indent=2))

    def wave(stage, commands):
        children, logs = [], []
        for gpu, command in enumerate(commands):
            log = (root / f'{stage}_gpu{gpu}.log').open('w')
            child = subprocess.Popen(command, cwd=REPO, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
            children.append(dict(gpu=gpu, pid=child.pid, command=command)); logs.append((child, log))
        record(stage.upper(), children=children)
        codes = [child.wait() for child, _ in logs]
        for _, log in logs: log.close()
        assert all(code == 0 for code in codes), (stage, codes, 'Read primary logs; no unchanged retry')

    wave('real_sanity', [[PYTHON, '-u', '-m', 'research.check_protected_visual_reference', '--model', PARENT,
        '--root', TRAIN_ROOT, '--split', SPLIT, '--sequence-offset', str(offset), '--mode', mode,
        '--output', str(root / 'sanity' / mode)] for mode, offset in zip(MODES, (0, 25, 50, 74))])
    assert all(read(root / 'sanity' / mode / 'completion.json')['completed'] for mode in MODES)
    record('REAL_PARENT_PARITY_AND_NATIVE_REFERENCE_INDEPENDENCE_PASS')
    wave('full98', [[PYTHON, '-u', '-m', 'research.evaluate_recoverability', '--dataset', 'lasher',
        '--root', TRAIN_ROOT, '--validation-split', SPLIT, '--model', PARENT,
        '--reference-mode', mode, '--search-value', 'gross', '--write-verification', 'action',
        '--output', str(root / 'full98' / mode / 'predictions')] for mode in MODES])
    expected = read(Path(PARENT_PREDICTIONS) / 'inference_completion.json')['records']
    for mode in MODES:
        folder = root / 'full98' / mode / 'predictions'
        done, config = read(folder / 'inference_completion.json'), read(folder / 'inference_config.json')
        assert done['completed'] and (done['sequences'], done['frames']) == (98, 49418)
        assert config['max_frames'] == config['limit_sequences'] == config['sequence_offset'] == 0
        assert config['reference_mode'] == mode and config['model'] == PARENT
        assert [(row['sequence'], row['frames']) for row in done['records']] == [(row['sequence'], row['frames']) for row in expected]
        for row in expected:
            name = row['sequence']
            with np.load(folder / (name + '_recoverability_decisions.npz')) as actual:
                if mode == 'parent':
                    assert (folder / (name + '.txt')).read_bytes() == (Path(PARENT_PREDICTIONS) / (name + '.txt')).read_bytes(), name
                    with np.load(Path(PARENT_PREDICTIONS) / (name + '_recoverability_decisions.npz')) as before:
                        assert all(np.array_equal(actual[key], before[key]) for key in before.files), name
                if mode in ('visual', 'visual_motion'):
                    references = np.concatenate((np.loadtxt(Path(BASELINE) / (name + '.txt'), ndmin=2)[:1],
                        actual['protected_reference_box']), 0)
                    references[1:, 2:] -= references[1:, :2]
                    text = io.StringIO(); np.savetxt(text, references, delimiter='\t', fmt='%.3f')
                    assert text.getvalue().encode() == (Path(BASELINE) / (name + '.txt')).read_bytes(), (mode, name)
    record('FULL98_PARENT_DECISION_PARITY_AND_SHARED_NATIVE_REFERENCE_PARITY_PASS')
    for mode in MODES:
        with (root / f'{mode}_reference_CPU_report.log').open('w') as log:
            subprocess.run([PYTHON, '-u', '-m', 'research.audit_protected_reference_metrics', '--dataset', 'lasher',
                '--root', TRAIN_ROOT, '--predictions', str(root / 'full98' / mode / 'predictions'),
                '--output', str(root / 'full98' / mode / 'reference_metrics')], cwd=REPO, env=environment(''),
                stdout=log, stderr=subprocess.STDOUT, check=True)
    references = [BASELINE, '/data/gb/outputs/abc_internal_validation_v1/c1/predictions',
        '/data/gb/outputs/recoverability_write_pair_reference_own_b384_full_20261004/predictions',
        '/data/gb/outputs/recoverability_search_gross_control_20261006/predictions', PARENT_PREDICTIONS]
    with (root / 'full98_CPU_report.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'research.collect_recoverability_metrics', '--dataset', 'lasher',
            '--root', TRAIN_ROOT, '--split', SPLIT, '--labels', *MODES,
            '--runs', *[str(root / 'full98' / mode / 'predictions') for mode in MODES],
            '--reference-labels', 'baseline', 'c1', 'old4', 'gross_parent', 'current_parent', '--references', *references,
            '--output', str(root / 'full98_report')], cwd=REPO, env=environment(''), stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(root / 'full98_report/full_recoverability_report.json')
    selected = max(MODES, key=lambda mode: report['variants'][mode]['sequence_mean_iou'])
    selection = dict(selected_candidate=selected, reference_mode=selected, parent_model=PARENT, state_commit_model=None,
        same_checkpoint_both_native_datasets=True, native_metrics_completed=False,
        new_optimizer_updates=0, selection_scope='State-propagation control, reused full98; not new ABC fit or independent confirmation',
        sequence_mean_iou={name: row['sequence_mean_iou'] for name, row in report['variants'].items()})
    (root / 'selected_model.json').write_text(json.dumps(selection, indent=2))
    record('ONE_COMPLETE_REFERENCE_POLICY_LOCKED_BEFORE_NATIVE', **selection)
    with (root / 'native_controller.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'scripts.run_selective_state_native', '--selection', str(root / 'selected_model.json'),
            '--review', args.review, '--output', str(root / 'native')], cwd=REPO, env=environment(''), stdout=log,
            stderr=subprocess.STDOUT, check=True)
    native = read(root / 'native/complete_metrics.json')
    assert native['completed'] and native['same_reference_mode_both_datasets'] == selected
    record('COMPLETE_PROTECTED_REFERENCE_FULL98_AND_NATIVE', selection=selection, native=native, new_optimizer_updates=0,
           own_weights_created_or_removed=0)


if __name__ == '__main__':
    main()
