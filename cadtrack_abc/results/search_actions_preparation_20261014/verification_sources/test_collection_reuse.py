from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

repo = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source')
sys.path.insert(0, str(repo))
from cadtrack_abc import future_pipeline as pipeline


class CollectionReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sampling = self.root / 'sampling'
        self.collection = self.root / 'collection'
        self.parent = self.root / 'parent'
        self.parent.mkdir()
        self.teacher = self.parent / 'best.pth'
        self.teacher.write_bytes(b'not-a-model-cpu-fixture')
        self.sha = hashlib.sha256(self.teacher.read_bytes()).hexdigest()
        for split in ['train', 'validation']:
            sampling = self.sampling / ('queries_' + split)
            sampling.mkdir(parents=True)
            for shard in range(2):
                name = f'{split}_{shard}'
                queries = [dict(sequence=name, frame=4)]
                (sampling / f'queries_shard{shard}.json').write_text(json.dumps(queries))
                sample = dict(program_complete=True, full_shard_queries_complete=True, full_sequence_rollouts=True,
                              sanity=False, teacher_sha256=self.sha, model_family='CADTrackABC-native-current-quality',
                              horizon=3, sequences=1, shard=shard, shards=2, queries=1, records=[dict(sequence=name)])
                (sampling / f'sampling_shard{shard}.json').write_text(json.dumps(sample))
                folder = self.collection / ('collection_' + split) / f'shard{shard}'
                folder.mkdir(parents=True)
                (folder / 'packet.pt').write_bytes(b'not-a-model-cpu-fixture')
                record = dict(sequence=name, frame=4, horizon=3, action_labels=210,
                              native_state_reconstruction_equal=True, full_predicted_prefix_from_frame0=True, file='packet.pt')
                receipt = dict(completed=True, teacher_sha256=self.sha, model_family='CADTrackABC-native-current-quality',
                               action_batch=64, records=[record], queries=1)
                (folder / 'collection_complete.json').write_text(json.dumps(receipt))
        self.folders = [self.collection / 'collection_train' / f'shard{shard}' for shard in range(2)]

    def test_matching_collection_reuses_all_queries(self):
        receipts = pipeline.validate_collection(self.folders, self.sampling / 'queries_train', self.sha)
        self.assertEqual(sum(r['queries'] for r in receipts), 2)

    def test_different_teacher_rejected(self):
        with self.assertRaises(AssertionError):
            pipeline.validate_collection(self.folders, self.sampling / 'queries_train', 'other_teacher')

    def test_query_identity_mismatch_rejected(self):
        path = self.folders[0] / 'collection_complete.json'
        receipt = json.loads(path.read_text())
        receipt['records'][0]['frame'] = 5
        path.write_text(json.dumps(receipt))
        with self.assertRaises(AssertionError):
            pipeline.validate_collection(self.folders, self.sampling / 'queries_train', self.sha)

    def test_missing_dev_labels_block_before_new_output_or_process(self):
        (self.collection / 'collection_validation/shard1/collection_complete.json').unlink()
        manifest = dict(train_sequences=['train_0', 'train_1'], validation_sequences=['validation_0', 'validation_1'])
        (self.parent / 'manifest.json').write_text(json.dumps(manifest))
        config = self.root / 'config.json'
        config.write_text(json.dumps(dict(physical_gpus=[2, 3], evaluation_focus='lasher', output=str(self.parent))))
        output = self.root / 'new_fit'
        args = ['future_pipeline', '--config', str(config), '--output', str(output), '--sampling-root', str(self.sampling),
                '--collection-root', str(self.collection), '--policy', 'search_actions']
        with patch.object(sys, 'argv', args), patch.object(pipeline, 'validate_parent', return_value=(self.teacher, {})), \
                patch.object(pipeline.subprocess, 'Popen') as popen:
            with self.assertRaises(FileNotFoundError):
                pipeline.main()
            popen.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CollectionReuseTests))
    receipt = dict(status='CPU_COLLECTION_REUSE_FIXTURES_PASS' if result.wasSuccessful() else 'FAILED',
                   tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                   models_loaded=0, optimizer_updates=0, full_action_collection_complete=False)
    Path(__file__).with_suffix('.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps(receipt))
    sys.exit(0 if result.wasSuccessful() else 1)
