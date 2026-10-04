"""Execute queue control flow with temporary paths and mocked external commands."""
import contextlib
import hashlib
import io
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

REVIEW = Path('/data/gb/setup/current_policy_fit_review')
SOURCE = REVIEW / 'queue_recoverability_current_policy_fit.py'
source = SOURCE.read_text()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
cells = [('oracle', 'c1', 0), ('selector', 'c1', 1), ('oracle', 'own', 2), ('selector', 'own', 3)]
stems = {'oracle': 'recoverability_current_policy_fit', 'selector': 'recoverability_current_policy_selector_fit'}
results = {}

for case in ('success', 'merge_failure', 'selector_audit_failure', 'selector_bad_receipt'):
    with tempfile.TemporaryDirectory(prefix='queue_CPU_FIXTURE_', dir=REVIEW) as temporary:
        base = Path(temporary)
        setup = base / 'setup'
        setup.mkdir()
        private = base / 'experiments/recoverability_current_policy_fit_20261004'
        fixture_review = setup / 'current_policy_fit_review'
        pipeline = setup / 'current_policy_four_fit_pipeline_20261004'
        (setup / 'current_policy_four_fit_queue_source_review_20261004.json').write_text(json.dumps({
            'status': 'PASS', 'review_independence': 'same-family', 'acceptance_status': 'provisional',
            'CPU_PROGRAM_FIXTURE_ONLY': True}))
        markers = [base / 'outputs' / f'recoverability_current_policy_{p}_{part}_shard{s}_20261004/job_completed.txt'
                   for p in ('c1', 'own') for part in ('train', 'validation') for s in range(4)]
        m0_markers = [base / 'outputs' / f'{stems[m]}_{p}_b384_m0_20261004/job_completed.txt'
                      for m, p, _ in cells]
        events, sleeps = [], []

        def fake_sleep(seconds):
            assert seconds == 240
            sleeps.append(seconds)
            assert len(sleeps) <= 2
            for marker in markers if len(sleeps) == 1 else m0_markers:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text('CPU PROGRAM FIXTURE ONLY\n')

        def fake_run(args, **kwargs):
            assert kwargs.get('check') is True
            if args[0] == 'tmux':
                assert args[1:4] == ['new-session', '-d', '-s']
                command = shlex.split(args[-1])
                assert command[0] == 'bash' and command[5] == '>' and command[7] == '2>&1'
                mode = 'selector' if 'selector_fit' in command[1] else 'oracle'
                gpu, policy, stage = int(command[2]), command[3], command[4]
                assert (mode, policy, gpu) in cells
                assert command[1] == str(private / 'scripts' / f'run_{stems[mode]}.sh')
                assert all(p.is_file() for p in markers)
                if stage == 'full':
                    assert all(p.is_file() for p in m0_markers)
                    assert [e['helper'] for e in events if e['kind'] == 'audit'] == [
                        'actual_m0_cpu_audit.py', 'selector_actual_m0_cpu_audit.py']
                events.append({'kind': 'launch', 'mode': mode, 'policy': policy, 'gpu': gpu, 'stage': stage})
                return subprocess.CompletedProcess(args, 0)
            assert args[0] == str(base / 'envs/gola/bin/python')
            helper = Path(args[1]).name
            if helper == 'merge_current_policy_caches.py':
                assert all(p.is_file() for p in markers) and not events
                events.append({'kind': 'merge'})
                if case == 'merge_failure':
                    raise subprocess.CalledProcessError(1, args)
                assert args[2] == '--output'
                out = Path(args[3])
                out.mkdir(parents=True)
                (out / 'paired_cache_gate.json').write_text(json.dumps({
                    'status': 'PASS', 'all_non_future_arrays_exact': True,
                    'matched_partitions': {'train': 902, 'validation': 128}, 'CPU_PROGRAM_FIXTURE_ONLY': True}))
            else:
                assert helper in ('actual_m0_cpu_audit.py', 'selector_actual_m0_cpu_audit.py')
                assert args[1] == str(fixture_review / helper) and kwargs['cwd'] == private
                assert all(p.is_file() for p in m0_markers)
                assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
                assert os.environ['PYTHONPATH'] == str(private) + ':' + str(base / 'GOLA')
                assert len([e for e in events if e['kind'] == 'launch']) == 4
                events.append({'kind': 'audit', 'helper': helper})
                selector = helper.startswith('selector_')
                if selector and case == 'selector_audit_failure':
                    raise subprocess.CalledProcessError(1, args)
                name = 'current_policy_selector_fit' if selector else 'current_policy_fit'
                (setup / (name + '_m0_acceptance_20261004.json')).write_text(json.dumps({
                    'status': 'FAIL' if selector and case == 'selector_bad_receipt' else 'PASS',
                    'both_teacher_arms_passed': True, 'batch_size': 384, 'optimizer_steps_per_arm': 9,
                    'parent_epoch': 4, 'CPU_PROGRAM_FIXTURE_ONLY': True}))
            return subprocess.CompletedProcess(args, 0)

        def fake_pane(args, **kwargs):
            assert args[:2] == ['tmux', 'list-panes'] and kwargs == {'text': True}
            assert args[-2:] == ['-F', '#{pane_pid}']
            return '12345\n'

        outcome = 'success'
        with patch('subprocess.run', fake_run), patch('subprocess.check_output', fake_pane), \
             patch('time.sleep', fake_sleep), patch.dict(os.environ, {
                 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(private) + ':' + str(base / 'GOLA')}), \
             contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(source.replace('/data/gb', str(base)), str(SOURCE), 'exec'), {'__name__': '__main__'})
            except subprocess.CalledProcessError:
                assert case in ('merge_failure', 'selector_audit_failure')
                outcome = 'blocked_before_full'
            except AssertionError:
                assert case == 'selector_bad_receipt'
                outcome = 'blocked_before_full'
        launches = [e for e in events if e['kind'] == 'launch']
        m0 = [e for e in launches if e['stage'] == 'm0']
        full = [e for e in launches if e['stage'] == 'full']
        if case == 'success':
            assert outcome == 'success' and len(m0) == len(full) == 4
            assert (pipeline / 'full_launch_completed.txt').is_file()
        else:
            assert outcome == 'blocked_before_full' and not full
            assert len(m0) == (0 if case == 'merge_failure' else 4)
            assert not (pipeline / 'full_launch_completed.txt').exists()
        results[case] = {'outcome': outcome, 'simulated_M0_launches': len(m0),
                         'simulated_full_launches': len(full), 'sleep_requests_seconds': sleeps,
                         'events': events}

record = {'status': 'PASS_CPU_QUEUE_CONTROL_FIXTURES', 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'cases': results,
          'scope': 'Actual source control flow with only /data/gb path constants redirected to temporary fixture roots. All subprocess and sleep calls mocked. Synthetic receipts are fixture-only and removed with temporary directories.',
          'actual_tmux_commands': 0, 'actual_merges': 0, 'actual_M0_audits': 0,
          'actual_training_launches': 0, 'CUDA_initialized': False, 'GPU_queries': 0}
(REVIEW / 'queue_cpu_fixture_receipt.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
