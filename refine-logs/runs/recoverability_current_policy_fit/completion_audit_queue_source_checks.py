"""Bounded mocked flow checks for the completion queue; never launch it."""
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUEUE = HERE.parents[2] / "scripts/queue_current_policy_completion_audits.py"
source = QUEUE.read_text(encoding="utf-8")
REMOTE = r'''
import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
tree = ast.parse(source, filename='queue_current_policy_completion_audits.py')
compile(tree, 'queue_current_policy_completion_audits.py', 'exec')
assert {node.module for node in tree.body if isinstance(node, ast.ImportFrom)} == {'datetime', 'pathlib'}
assert {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names} == {'json', 'os', 'subprocess', 'time'}
cases = {}

class EndFixture(Exception):
    pass

for case in ('successful_stage_order', 'missing_fourth_training_marker', 'oracle_fit_failure',
             'wrong_selector_fit_receipt', 'oracle_full98_failure', 'closure_failure',
             'missing_source_review', 'source_review_not_PASS', 'duplicate_pipeline', 'nonempty_CUDA_environment'):
    with tempfile.TemporaryDirectory(prefix='completion_queue_CPU_FIXTURE_ONLY_') as temporary:
        base = Path(temporary)
        mapping = {
            '/data/gb/experiments/recoverability_current_policy_fit_20261004': str(base/'private'),
            '/data/gb/setup/current_policy_completion_review': str(base/'setup/review'),
            '/data/gb/setup': str(base/'setup'),
            '/data/gb/outputs': str(base/'outputs'),
        }
        class Redirect(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value in mapping:
                    return ast.copy_location(ast.Constant(mapping[node.value]), node)
                return node
        program = compile(ast.fix_missing_locations(Redirect().visit(ast.parse(source))),
                          'queue_current_policy_completion_audits.py:CPU_FIXTURE_ONLY', 'exec')
        setup = base/'setup'; review = setup/'review'; review.mkdir(parents=True)
        private = base/'private'; private.mkdir()
        pipeline = setup/'current_policy_completion_pipeline_20261004'
        stems = ('current_policy_fit', 'current_policy_selector_fit')
        roots = [base/'outputs'/f'recoverability_{stem}_{policy}_b384_full_20261004'
                 for stem in stems for policy in ('c1', 'own')]
        for root in roots:
            root.mkdir(parents=True)
        gate = dict(status='PASS', review_independence='same-family', acceptance_status='provisional')
        if case == 'source_review_not_PASS':
            gate['status'] = 'PENDING'
        if case != 'missing_source_review':
            (review/'completion_audit_queue_source_review.json').write_text(json.dumps(gate))
        if case == 'duplicate_pipeline':
            pipeline.mkdir(); (pipeline/'retained.txt').write_text('CPU_FIXTURE_ONLY original marker')
        calls, sleeps, phase_sleeps = [], [], {}
        def phase():
            return json.loads((pipeline/'state.json').read_text())['phase']
        def fake_sleep(seconds):
            assert seconds == 240
            current = phase(); sleeps.append([current, seconds])
            phase_sleeps[current] = phase_sleeps.get(current, 0)+1
            if case == 'missing_fourth_training_marker':
                for root in roots[:3]:
                    (root/'training_completed.txt').write_text('CPU_FIXTURE_ONLY')
                raise EndFixture('fixture terminated without fourth marker')
            assert current in ('WAIT_ALL4_TRAINING_COMPLETIONS', 'WAIT_ALL4_FULL98_REPORTS')
            marker = 'training_completed.txt' if current == 'WAIT_ALL4_TRAINING_COMPLETIONS' else 'job_completed.txt'
            count = 3 if phase_sleeps[current] == 1 else 4
            assert phase_sleeps[current] <= 2
            for root in roots[:count]:
                (root/marker).write_text('CPU_FIXTURE_ONLY')
        def fake_run(command, cwd, check):
            assert command[0] == '/data/gb/envs/gola/bin/python' and cwd == private and check is True
            assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
            filename = Path(command[1]).name
            calls.append(filename)
            assert all((root/'training_completed.txt').is_file() for root in roots)
            if 'full_fit' in filename:
                assert phase() == 'ACTUAL_CPU_FULL_FIT_AUDITS'
                if case == 'oracle_fit_failure' and filename == 'oracle_actual_full_fit_cpu_audit.py':
                    raise subprocess.CalledProcessError(1, command)
                stem = stems[0] if filename.startswith('oracle_') else stems[1]
                receipt = dict(status='PASS', both_teacher_arms_passed=True, epochs_per_arm=60, optimizer_steps_per_arm=180)
                if case == 'wrong_selector_fit_receipt' and filename.startswith('selector_'):
                    receipt['optimizer_steps_per_arm'] = 179
                (setup/f'{stem}_actual_full_fit_cpu_audit_20261004.json').write_text(json.dumps(receipt))
            elif 'full98' in filename:
                assert all((root/'job_completed.txt').is_file() for root in roots)
                assert phase() == 'ACTUAL_CPU_FULL98_AUDITS'
                if case == 'oracle_full98_failure' and filename.startswith('oracle_'):
                    raise subprocess.CalledProcessError(1, command)
            else:
                assert filename == 'four_factor_closure.py' and phase() == 'ACTUAL_FOUR_CELL_COMPARISON'
                assert calls[-3:] == ['oracle_actual_full98_cpu_audit.py', 'selector_actual_full98_cpu_audit.py', filename]
                if case == 'closure_failure':
                    raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        failure = None
        environment = '' if case != 'nonempty_CUDA_environment' else 'CPU_FIXTURE_NONEMPTY'
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': environment}), \
             patch('subprocess.run', fake_run), patch('time.sleep', fake_sleep), \
             contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(program, {'__name__': '__CPU_FIXTURE_ONLY__'})
            except (AssertionError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError, EndFixture) as error:
                failure = type(error).__name__
        expected = ['oracle_actual_full_fit_cpu_audit.py', 'selector_actual_full_fit_cpu_audit.py',
                    'oracle_actual_full98_cpu_audit.py', 'selector_actual_full98_cpu_audit.py', 'four_factor_closure.py']
        if case == 'successful_stage_order':
            assert failure is None and calls == expected
            assert sleeps == [['WAIT_ALL4_TRAINING_COMPLETIONS', 240]]*2 + [['WAIT_ALL4_FULL98_REPORTS', 240]]*2
            assert phase() == 'ALL_ACTUAL_INTERNAL_AUDITS_COMPLETED' and (pipeline/'job_completed.txt').is_file()
        else:
            assert failure is not None and not (pipeline/'job_completed.txt').exists()
            counts = {'missing_fourth_training_marker': 0, 'oracle_fit_failure': 1, 'wrong_selector_fit_receipt': 2,
                      'oracle_full98_failure': 3, 'closure_failure': 5,
                      'missing_source_review': 0, 'source_review_not_PASS': 0, 'duplicate_pipeline': 0,
                      'nonempty_CUDA_environment': 0}
            assert calls == expected[:counts[case]]
            if case in ('missing_source_review', 'source_review_not_PASS', 'nonempty_CUDA_environment'):
                assert not pipeline.exists()
            if case == 'duplicate_pipeline':
                assert (pipeline/'retained.txt').read_text() == 'CPU_FIXTURE_ONLY original marker'
        cases[case] = {'status':'PASS', 'mocked_helper_calls':calls, 'mocked_sleeps':sleeps,
                       'expected_stop':failure, 'actual_completion_marker_written':False,
                       'synthetic_completion_marker_written':(pipeline/'job_completed.txt').is_file()}
assert 'torch' not in sys.modules
print(json.dumps({'status':'PASS_SOURCE_AND_MOCKED_QUEUE_FLOW_ONLY',
                  'queue_sha256':hashlib.sha256(source.encode()).hexdigest(), 'remote_python':sys.executable,
                  'remote_compile':True, 'cases':cases,
                  'scope':'Program executed only with synthetic temporary paths and mocked subprocess/sleep; no actual coordinator or helper launch.',
                  'execution':dict(actual_queue_launches=0, actual_helper_executions=0, actual_sleeps=0,
                                   NN_forward=0, optimizer_steps=0, torch_imported=False, GPU_queries=0,
                                   power_temperature_queries=0, process_signals=0, production_sources_modified=False,
                                   actual_outputs_changed=False, weights_read_or_modified=False)}, indent=2))
'''
result = subprocess.run(
    ["ssh", "2027", "env CUDA_VISIBLE_DEVICES= /data/gb/envs/gola/bin/python -"],
    input="source = " + repr(source) + "\n" + REMOTE, text=True, capture_output=True,
)
assert result.returncode == 0, result.stderr
receipt = json.loads(result.stdout)
assert receipt["queue_sha256"] == hashlib.sha256(QUEUE.read_bytes()).hexdigest()
(HERE / "completion_audit_queue_cpu_fixture_receipt.json").write_text(
    json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2, allow_nan=False))
