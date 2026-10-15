"""Host stdlib control-flow checks; subprocess/GPU/model execution is mocked."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


p = Path(__file__).parent
repo = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source')
source = repo / 'cadtrack_abc/continue_dense_main.py'
spec = importlib.util.spec_from_file_location('dependency_queue_check_target', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = json.loads((repo / 'cadtrack_abc/results/three_benchmark_closure_20261015/r3_complete_chain_source_1805/qualified_scope.json').read_text())
results = []


class FirstSleepReached(Exception):
    pass


class QueueChecks(unittest.TestCase):
    def scenario(self, kind):
        with tempfile.TemporaryDirectory(prefix='dependency_queue_fixture_', dir=p) as directory:
            root = Path(directory)
            parent, destination, output = root / 'parent', root / 'main', root / 'queue'
            parent.mkdir()
            (root / 'pretrained').write_bytes(b'fixture-not-a-model')
            (root / 'config.json').write_text(json.dumps(dict(pretrained=str(root / 'pretrained'))))
            command = [sys.executable, '-m', 'cadtrack_abc.dense_main_pipeline',
                '--config', str(root / 'config.json'), '--r2-pair-root', str(parent), '--output', str(destination)]
            scope = dict(original, prepared_command_not_executed=command)
            (root / 'scope.json').write_text(json.dumps(scope))
            if kind == 'before_deadline':
                # If this is read before the first sleep, it raises JSON decode
                # rather than the expected timer sentinel.
                (parent / 'pair_exit.json').write_text('must not be read yet')
            else:
                (parent / 'pair_exit.json').write_text(json.dumps(dict(actual_controller_exit_code=9 if kind == 'parent_failed' else 0)))
                (parent / 'pair_complete.json').write_text(json.dumps(dict(
                    status='DENSE_MATCHED_PAIR_FULL_TRAINING_AND_EVALUATION_COMPLETE',
                    actual_exit_codes=[0, 0], all20_actual_clip_orders_equal=True)))
            child = Mock(pid=123456)

            def wait_for_fake_child():
                if kind == 'child_failed':
                    return 7
                destination.mkdir()
                (destination / 'pipeline_complete.json').write_text(json.dumps(dict(
                    status='DENSE_R3_COMPLETE_TRAINING_AND_EVALUATION',
                    full_ABC_training_complete=True, seven_metric_plus_two=False)))
                return 0

            child.wait.side_effect = wait_for_fake_child
            capacities = [Mock(stdout='2, 12000, 24576\n3, 15000, 24576\n'),
                Mock(stdout='2, 1, 24576\n3, 1, 24576\n')] if kind == 'busy_then_free' else [Mock(stdout='2, 1, 24576\n3, 1, 24576\n')]
            argv = ['continue_dense_main.py', '--scope', str(root / 'scope.json'),
                '--r2-pair-root', str(parent), '--not-before', '2099-01-01T00:00:00+00:00',
                '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(module.time, 'time', return_value=0), \
                 patch.object(module.time, 'sleep') as sleep, \
                 patch.object(module.subprocess, 'run', side_effect=capacities) as capacity, \
                 patch.object(module.subprocess, 'Popen', return_value=child) as launch:
                if kind == 'before_deadline':
                    sleep.side_effect = FirstSleepReached
                    with self.assertRaises(FirstSleepReached):
                        module.main()
                    code = None
                else:
                    code = module.main()
                state = json.loads((output / 'queue_state.json').read_text())
                if kind == 'before_deadline':
                    self.assertEqual(state['status'], 'WAITING_UNTIL_NOT_BEFORE')
                    self.assertIsNone(state['last_dependency_check_utc'])
                    self.assertEqual(launch.call_count, 0)
                    self.assertEqual(capacity.call_count, 0)
                    self.assertGreater(sleep.call_args.args[0], 300)
                elif kind == 'parent_failed':
                    self.assertEqual((code, state['status']), (1, 'FAILED_R2_DEPENDENCY'))
                    self.assertEqual(launch.call_count, 0)
                    self.assertEqual(capacity.call_count, 0)
                else:
                    self.assertEqual(launch.call_count, 1)
                    self.assertEqual(state['launch_count'], 1)
                    self.assertEqual(state['launch_attempts'], 1)
                    self.assertTrue(state['main_controller_started'])
                    self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                    self.assertTrue(all(call.args[0][2] == '2,3' for call in capacity.call_args_list))
                    if kind == 'child_failed':
                        self.assertEqual((code, state['status']), (7, 'FAILED_R3_MAIN_CONTROLLER'))
                        self.assertEqual(state['actual_main_controller_exit_code'], 7)
                    else:
                        self.assertEqual((code, state['status']), (0, 'R3_COMPLETE_TRAINING_AND_EVALUATION_FINISHED'))
                        self.assertFalse(state['seven_metric_plus_two'])
                        self.assertFalse(state['goal_achieved'])
                    if kind == 'busy_then_free':
                        self.assertEqual(capacity.call_count, 2)
                        self.assertEqual(sleep.call_args.args[0], 300)
                results.append(dict(case=kind, passed=True, main_launches=launch.call_count,
                    mocked_capacity_calls=capacity.call_count, mocked_sleep_calls=sleep.call_count,
                    real_NN_or_GPU_processes_started=0))

    def test_not_before_gate(self):
        self.scenario('before_deadline')

    def test_parent_failure(self):
        self.scenario('parent_failed')

    def test_busy_cards_wait_then_one_launch(self):
        self.scenario('busy_then_free')

    def test_child_failure(self):
        self.scenario('child_failed')

    def test_success_is_not_goal_success(self):
        self.scenario('success')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QueueChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        actual_tests=result.testsRun, cases=results,
        scope='Executor host stdlib control-flow tests with mocked sleep/subprocess. Fixture success receipts are synthetic, not experiment evidence.',
        project_neural_modules_imported=False, torch_imported='torch' in sys.modules,
        actual_GPU_queries=0, actual_NN_jobs=0, formal_metrics=None, goal_achieved=False)
    (p / 'dense_dependency_queue_control_flow_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
