import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets import prepare_sample, resolve_dataset, MAX_SAMPLE_BYTES
from engine import load_events, write_json
from selected_hits import SelectedHitWriter


def export(path, sample_type='MC_neutrino', group='CC_nue', uid='unit-test', count=5):
    writer=SelectedHitWriter(path,{'kind':'artificial integration-test hits','sample_type':sample_type})
    for i in range(count):
        writer.add(dict(runId=1,eventId=i,eventIndex=i),
                   [dict(station=1,isVertical=True,layer_channel=1,qdc=2.)],group,uid,'test-geometry')
    writer.finish(count)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.a=self.root/'a.jsonl';export(self.a)
        self.b=self.root/'b.jsonl';export(self.b,group='NC_nue',uid='second')
        self.out=self.root/'small.jsonl'
        self.config=self.root/'config.json'
        write_json(self.config,{'version':1,'datasets':{'mc-neutrino':{'samples':{'small':'small.jsonl','full':'full.jsonl'},'required_groups':['CC_nue','NC_nue']}}})
    def tearDown(self):self.tmp.cleanup()
    def prepare(self,inputs=None,cap=2):
        return prepare_sample('mc-neutrino',inputs or [self.a,self.b],self.out,'v1',cap)
    def test_balanced_sample(self):
        self.assertEqual(self.prepare(),4)
        rows,manifest=load_events(self.out)
        self.assertEqual({g:sum(e['group']==g for e in rows) for g in ['CC_nue','NC_nue']},{'CC_nue':2,'NC_nue':2})
        self.assertEqual(manifest['provenance']['sample_version'],'v1')
    def test_order_independence(self):
        self.prepare();other=self.root/'other.jsonl'
        prepare_sample('mc-neutrino',[self.b,self.a],other,'v1',2)
        self.assertEqual(self.out.read_bytes(),other.read_bytes())
    def test_full_keeps_every_event(self):self.assertEqual(self.prepare(cap=0),10)
    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.prepare([self.a,self.a])
        self.assertFalse(self.out.exists())
    def test_wrong_family(self):
        with self.assertRaisesRegex(ValueError,'does not match'):prepare_sample('testbeam',[self.a],self.out,'v1',2)
    def test_synthetic_rejected(self):
        from engine import ROOT
        with self.assertRaisesRegex(ValueError,'Synthetic'):self.prepare([ROOT/'feature_dev/samples/small.jsonl'])
    def test_no_overwrite(self):
        self.prepare()
        with self.assertRaises(FileExistsError):self.prepare()
    def test_resolve(self):
        self.prepare();path,identity=resolve_dataset('mc-neutrino','small',self.config,self.root)
        self.assertEqual(path,self.out);self.assertEqual(identity['name'],'mc-neutrino')
    def test_missing_sample(self):
        with self.assertRaisesRegex(ValueError,'No prepared'):resolve_dataset('mc-neutrino','small',self.config,self.root)
    def test_missing_group(self):
        self.prepare([self.a])
        with self.assertRaisesRegex(ValueError,'Missing configured groups'):resolve_dataset('mc-neutrino','small',self.config,self.root)
    def test_unregistered_sample(self):
        self.out.write_bytes(self.a.read_bytes())
        Path(str(self.out)+'.manifest.json').write_bytes(Path(str(self.a)+'.manifest.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'matching dataset'):resolve_dataset('mc-neutrino','small',self.config,self.root)
    def test_large_sample_guard(self):
        with self.out.open('wb') as f:f.truncate(MAX_SAMPLE_BYTES+1)
        with self.assertRaisesRegex(ValueError,'256 MiB'):resolve_dataset('mc-neutrino','small',self.config,self.root)
    def test_unsupported_normalization(self):
        write_json(self.config,{'version':1,'datasets':{'mc-neutrino':{'normalization':'19fb'}}})
        with self.assertRaisesRegex(ValueError,'unit_area'):resolve_dataset('mc-neutrino','small',self.config,self.root)

if __name__=='__main__':unittest.main()
