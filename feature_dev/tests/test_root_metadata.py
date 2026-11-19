"""ROOT metadata round trips, explicit missing values and tamper detection."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import uproot
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from root_metadata import write_metadata, validate_metadata
from exposure import validate_audit
from test_exposure import fixture, source


class RootMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'features.root'
        self.audit,self.events=fixture()
        self.plan=dict(sample='fixture',source_hashes={'fixture':'test-sha'},exposure_audit=self.audit,
                       normalization=validate_audit(self.audit,'test-sha',self.events,{}))
        with uproot.recreate(self.path) as f:
            f.mktree('baseline',{'eventId':'int64'});f['baseline'].extend({'eventId':np.array([1,2])})
    def tearDown(self):self.tmp.cleanup()
    def write(self):
        write_metadata(self.path,self.plan,self.events,'raw-sha')
        validate_metadata(self.path,self.plan,self.events,'raw-sha')
    def test_yield_roundtrip_includes_empty_and_reference(self):
        self.write()
        with uproot.open(self.path) as f:
            a=f['source_exposure'].arrays(library='np')
            self.assertEqual(set(a['source_uid']),{'ve','vm','empty','data'})
            by_uid={uid:i for i,uid in enumerate(a['source_uid'])}
            self.assertEqual(a['selected_entries'][by_uid['empty']],0)
            self.assertTrue(a['in_sample'][by_uid['empty']])
            self.assertFalse(a['in_sample'][by_uid['data']])
            self.assertEqual(a['effective_luminosity_fb'][by_uid['data']],2)
            audit=str(f['exposure_audit_json'])
            self.assertEqual(json.loads(audit),self.audit)
            self.assertEqual(str(f['exposure_audit_sha256']),hashlib.sha256(audit.encode()).hexdigest())
            self.assertEqual(json.loads(str(f['feature_metadata_json']))['exposure_unit'],'fb^-1')
            np.testing.assert_array_equal(f['baseline']['eventId'].array(library='np'),[1,2])
    def test_shape_unknown_is_not_zero(self):
        self.plan.pop('exposure_audit')
        self.plan['normalization']=[dict(group=e['group'],geometry='geo',mode='shape') for e in self.events]
        self.write()
        with uproot.open(self.path) as f:
            for name,field in [('source_exposure','luminosity_fb'),('normalization','event_weight')]:
                a=f[name].arrays(library='np')
                self.assertFalse(a['exposure_valid'].any())
                self.assertTrue(np.isnan(a[field]).all())
            self.assertEqual(str(f['exposure_audit_json']),'null')
    def test_corrupted_audit_rejected(self):
        self.write()
        with uproot.update(self.path) as f:f['exposure_audit_json']='{}'
        with self.assertRaisesRegex(ValueError,'metadata mismatch'):
            validate_metadata(self.path,self.plan,self.events,'raw-sha')
    def test_changed_normalization_rejected(self):
        self.write();self.plan['normalization'][0]['scale']*=2
        with self.assertRaises(AssertionError):
            validate_metadata(self.path,self.plan,self.events,'raw-sha')
    def test_data_effective_exposure_weight_one(self):
        self.events=[dict(source_uid='data',group='data',geometry='geo')]
        self.audit=dict(version=1,sample_sha256='test-sha',populations={'data':dict(role='data',coverage_evidence='fixture',sources=[source('data',20,1,.1)])},groups=[dict(group='data',geometry='geo',population='data')])
        self.plan['exposure_audit']=self.audit
        self.plan['normalization']=validate_audit(self.audit,'test-sha',self.events,{})
        self.write()
        with uproot.open(self.path) as f:
            a=f['normalization'].arrays(library='np')
            self.assertEqual(a['event_weight'][0],1)
            self.assertEqual(a['target_exposure_fb'][0],2)

if __name__=='__main__':unittest.main()
