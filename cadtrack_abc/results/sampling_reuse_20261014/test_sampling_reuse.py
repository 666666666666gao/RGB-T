"""CPU fixture checks for reuse prerequisites, never dataset metrics or NN."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

repo=Path('C:/Users/gb/.codex_tmp/gola_setup_20261002/GOLA-source')
sys.path.insert(0,str(repo))
from cadtrack_abc import future_pipeline


class SamplingReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.sampling=self.root/'sampling'
        self.folder=self.sampling/'queries_train'
        self.folder.mkdir(parents=True)
        self.expected={'train_a','train_b'}
        for shard,name in enumerate(sorted(self.expected)):
            receipt=dict(program_complete=True,full_shard_queries_complete=True,full_sequence_rollouts=True,
                         sanity=False,teacher_sha256='fixed_teacher',model_family='CADTrackABC-native-current-quality',
                         horizon=3,sequences=1,shard=shard,shards=2,queries=2,records=[{'sequence':name}])
            self.write_receipt(shard,receipt)
            (self.folder/f'queries_shard{shard}.json').write_text(json.dumps([
                dict(sequence=name,frame=4),dict(sequence=name,frame=5)]))

    def write_receipt(self,shard,receipt):
        (self.folder/f'sampling_shard{shard}.json').write_text(json.dumps(receipt))

    def change(self,key,value):
        path=self.folder/'sampling_shard0.json'
        receipt=json.loads(path.read_text())
        receipt[key]=value
        self.write_receipt(0,receipt)

    def test_complete_two_shard_input_is_reusable(self):
        receipts=future_pipeline.validate_sampling(self.folder,self.expected,'fixed_teacher')
        self.assertEqual(len(receipts),2)
        self.assertEqual(sum(row['queries'] for row in receipts),4)

    def test_changed_teacher_rejected(self):
        self.change('teacher_sha256','other_teacher')
        with self.assertRaises(AssertionError):
            future_pipeline.validate_sampling(self.folder,self.expected,'fixed_teacher')

    def test_sanity_trajectories_rejected(self):
        self.change('sanity',True)
        with self.assertRaises(AssertionError):
            future_pipeline.validate_sampling(self.folder,self.expected,'fixed_teacher')

    def test_other_policy_family_rejected(self):
        self.change('model_family','CADTrackFuturePolicy-future-actions')
        with self.assertRaises(AssertionError):
            future_pipeline.validate_sampling(self.folder,self.expected,'fixed_teacher')

    def test_different_horizon_rejected(self):
        self.change('horizon',1)
        with self.assertRaises(AssertionError):
            future_pipeline.validate_sampling(self.folder,self.expected,'fixed_teacher')

    def test_duplicate_queries_rejected(self):
        (self.folder/'queries_shard0.json').write_text(json.dumps([dict(sequence='train_a',frame=4)]*2))
        with self.assertRaises(AssertionError):
            future_pipeline.validate_sampling(self.folder,self.expected,'fixed_teacher')

    def test_train_only_completion_blocks_before_new_output_or_process(self):
        parent=self.root/'parent'
        parent.mkdir()
        teacher=parent/'best.pth'
        teacher.write_bytes(b'fixture-not-a-model')
        from hashlib import sha256
        teacher_sha=sha256(teacher.read_bytes()).hexdigest()
        for shard in range(2):
            path=self.folder/f'sampling_shard{shard}.json'
            receipt=json.loads(path.read_text())
            receipt['teacher_sha256']=teacher_sha
            path.write_text(json.dumps(receipt))
        (parent/'manifest.json').write_text(json.dumps(dict(train_sequences=sorted(self.expected),validation_sequences=['dev_a','dev_b'])))
        config=self.root/'config.json'
        config.write_text(json.dumps(dict(physical_gpus=[2,3],evaluation_focus='lasher',output=str(parent))))
        output=self.root/'new_fit'
        args=['future_pipeline','--config',str(config),'--output',str(output),'--sampling-root',str(self.sampling)]
        with patch.object(sys,'argv',args),patch.object(future_pipeline,'validate_parent',return_value=(teacher,{})),patch.object(future_pipeline.subprocess,'Popen') as popen:
            with self.assertRaises(FileNotFoundError):
                future_pipeline.main()
            popen.assert_not_called()
        self.assertFalse(output.exists())


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SamplingReuseTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(status='CPU_REUSE_GUARD_FIXTURE_PASS' if result.wasSuccessful() else 'CPU_REUSE_GUARD_FIXTURE_FAILED',
                 tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                 actual_full_sampling_complete=False,models_loaded=0,neural_forwards=0,optimizer_updates=0,
                 scope='Temporary synthetic completion/query fixtures test pipeline prerequisite order only, not actual sampling or performance.')
    Path(__file__).with_suffix('.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
    print(json.dumps(receipt))
    sys.exit(0 if result.wasSuccessful() else 1)
