"""Batch planning/deployment contracts; these tests never submit jobs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import batch
import batch_adapter as adapter


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def profile(self):
        p=self.root/'profile';p.mkdir()
        (p/'config.yaml').write_text('executor: local\n')
        (p/'config.v8+.yaml').write_text(yaml.safe_dump({'executor':'cluster-generic',
            'cluster-generic-submit-cmd':'submit.py','cluster-generic-status-cmd':'status.py'}))
        return p
    def test_versioned_config_and_rejections(self):
        p=self.profile();self.assertEqual(batch.profile_config(p)[1]['executor'],'cluster-generic')
        (p/'config.v8+.yaml').write_text('executor: cluster-generic\n')
        with self.assertRaisesRegex(ValueError,'submit and status'):batch.profile_config(p)
        (p/'config.v8+.yaml').unlink()
        with self.assertRaisesRegex(ValueError,'Batch profile'):batch.profile_config(p)
    def test_selection_keeps_small_local_and_explicit_wins(self):
        saved=self.root/'saved';saved.write_text(str(self.root/'default'))
        with patch.object(batch,'SAVED',saved):
            self.assertIsNone(batch.choose_profile(None,'small'))
            self.assertEqual(batch.choose_profile(None,'full'),self.root/'default')
            self.assertEqual(batch.choose_profile(None,'small',True),self.root/'default')
            self.assertEqual(batch.choose_profile(self.root/'override','full'),self.root/'override')
    def test_repeat_preserves_custom_profile(self):
        p=self.profile();(p/'deployment.json').write_text('{}')
        saved=self.root/'saved';before=(p/'config.v8+.yaml').read_bytes()
        with patch.object(batch,'SAVED',saved),patch.object(batch,'check_profile'),patch.object(batch.subprocess,'run') as run:
            batch.setup(self.root/'env',profile=p,jobs=50)
            run.assert_not_called()
        self.assertEqual((p/'config.v8+.yaml').read_bytes(),before)
        self.assertEqual(saved.read_text().strip(),str(p))
    def test_unmanaged_profile_refused(self):
        with self.assertRaisesRegex(ValueError,'unmanaged'):batch.setup(self.root/'env',profile=self.profile())
    def test_resources_and_shared_paths(self):
        d=adapter.description(self.root/'job with space.sh',{'threads':2,'resources':{'runtime':3,'mem_mb':2048,'disk_mb':5000}},self.root)
        self.assertEqual(d['+MaxRuntime'],'180');self.assertEqual(d['request_memory'],'2048MB')
        self.assertEqual(d['request_disk'],'5000MB');self.assertEqual(d['request_cpus'],'2')
        self.assertEqual(d['should_transfer_files'],'NO');self.assertIn('job with space.sh',d['arguments'])
        self.assertEqual(d['log'],str(self.root/'condor.log'))
    def test_bad_resources_and_ids(self):
        for v in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):adapter.positive(v)
        for v in ['../../other_123','bad','a_12']:
            with self.assertRaises(ValueError):adapter.parse_id(v)
        self.assertEqual(adapter.parse_id('a'*32+'_123'),('a'*32,123))
    def test_job_event_status(self):
        types=SimpleNamespace(JOB_HELD=1,JOB_ABORTED=2,EXECUTABLE_ERROR=3,JOB_TERMINATED=4)
        class Event(dict):
            type=4
        self.assertEqual(adapter.event_status([Event(ReturnValue=0)],types),'success')
        self.assertEqual(adapter.event_status([Event(ReturnValue=0,TerminatedNormally=False)],types),'failed')
        self.assertEqual(adapter.event_status([Event(ReturnValue=1)],types),'failed')
        self.assertEqual(adapter.event_status([],types),'running')
    def test_property_header(self):
        p=self.root/'job.sh';p.write_text('#!/bin/bash\n# properties = {"threads": 1}\ntrue\n')
        self.assertEqual(adapter.properties(p),{'threads':1})
    def test_cancel_checks_receipt_and_targets_only_recorded_job(self):
        from unittest.mock import Mock
        import json
        token='b'*32; folder=self.root/token;folder.mkdir()
        (folder/'job.json').write_text(json.dumps({'cluster':123}))
        schedd=Mock();binding=SimpleNamespace(Schedd=lambda:schedd,JobAction=SimpleNamespace(Remove='remove'))
        with patch.dict(sys.modules,{'htcondor2':binding}),patch.object(sys,'argv',['adapter','cancel',str(self.root),token+'_123']):
            adapter.main()
        schedd.act.assert_called_once_with('remove',['123.0'])
        schedd.reset_mock()
        with patch.dict(sys.modules,{'htcondor2':binding}),patch.object(sys,'argv',['adapter','cancel',str(self.root),token+'_456']):
            with self.assertRaisesRegex(ValueError,'receipt mismatch'):adapter.main()
        schedd.act.assert_not_called()

    def test_check_does_not_write(self):
        p=self.profile();saved=self.root/'saved'
        with patch.object(batch,'SAVED',saved),patch.object(batch,'check_profile',return_value={'profile':str(p)}):
            batch.setup(self.root/'env',check=True,profile=p)
        self.assertFalse(saved.exists())


if __name__=='__main__':unittest.main()
