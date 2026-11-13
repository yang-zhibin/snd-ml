import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import ROOT, schema, load_algorithm, evaluate, load_events, validate_values
from selected_hits import SelectedHitWriter, merge_exports

class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.base=schema(ROOT/'feature_dev/schemas/baseline.yaml')['features']
        self.gap=schema(ROOT/'feature_dev/schemas/gap_clusters.yaml')['features']
        self.b=load_algorithm(ROOT/'feature_dev/algorithms/baseline.py')
        self.g=load_algorithm(ROOT/'feature_dev/algorithms/gap_clusters.py')
        self.hits=[dict(station=1,isVertical=True,layer_channel=c,qdc=q) for c,q in [(1,2.),(3,3.),(10,5.)]]
    def test_known_gap_feature(self):
        values=evaluate(self.g,self.hits,{'max_missing_channels':1},self.gap)
        self.assertEqual(values['scifi_cluster_total_count'],2)
        self.assertEqual(values['charge_fragmentation_v1'],.5)
        self.assertEqual(values['scifi_cluster_total_qdc'],10)
    def test_gap_zero_parity(self):
        b=evaluate(self.b,self.hits,{},self.base)
        g=evaluate(self.g,self.hits,{'max_missing_channels':0},self.gap)
        self.assertEqual(b,{k:g[k] for k in b})
    def test_empty_hits(self):
        g=evaluate(self.g,[],{},self.gap)
        self.assertEqual(g['charge_fragmentation_v1'],-999)
    def test_invalid_parameter(self):
        for p in [{'max_missing_channels':-1},{'max_missing_channels':True},{'typo':1}]:
            with self.assertRaises(ValueError):evaluate(self.g,self.hits,p,self.gap)
    def test_no_mutation(self):
        before=copy.deepcopy(self.hits);evaluate(self.g,self.hits,{},self.gap)
        self.assertEqual(self.hits,before)
    def test_reject_mutating_algorithm(self):
        def cluster(h,p):h.clear();return None
        m=types.SimpleNamespace(cluster_hits=cluster,calculate_features=lambda *a:{})
        with self.assertRaisesRegex(ValueError,'mutated'):evaluate(m,self.hits,{},self.base)
    def test_schema_mismatch(self):
        with self.assertRaisesRegex(ValueError,'schema mismatch'):validate_values({},self.base)
    def test_values(self):
        definition={'f':dict(dtype='int64',missing_value=-999,valid_range=[0,2])}
        for v in [True,1.5,float('nan'),float('inf'),3,2**64]:
            with self.assertRaises(ValueError):validate_values({'f':v},definition)
        validate_values({'f':-999},definition)
    def export(self,path,hits=None):
        w=SelectedHitWriter(path,{'kind':'test'})
        w.add(dict(runId=1,eventId=2,eventIndex=3),self.hits if hits is None else hits,'label','uid','geometry')
        w.finish(1)
    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hits.jsonl';self.export(p)
            events,m=load_events(p)
            self.assertEqual(events[0]['hits'],self.hits)
            self.assertEqual(m['events'],1)
            with self.assertRaises(FileExistsError):self.export(p)
    def test_modified_sample(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hits.jsonl';self.export(p);p.write_text(p.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'modified'):load_events(p)
    def test_duplicate_channel(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hits.jsonl';self.export(p,self.hits+self.hits[:1])
            with self.assertRaisesRegex(ValueError,'Duplicate hit'):load_events(p)
    def test_incomplete_export(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hits.jsonl';w=SelectedHitWriter(p,{})
            with self.assertRaises(ValueError):w.finish(1)
            self.assertFalse(Path(str(p)+'.manifest.json').exists())
    def test_fixture_groups(self):
        events,m=load_events(ROOT/'feature_dev/samples/small.jsonl')
        self.assertEqual(len(events),24)
        self.assertEqual(len({e['group'] for e in events}),3)
        self.assertEqual(m['provenance']['kind'],'synthetic')
    def test_merge_exports(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.jsonl';b=Path(d)/'b.jsonl';out=Path(d)/'out.jsonl'
            self.export(a)
            w=SelectedHitWriter(b,{'kind':'test'})
            w.add(dict(runId=1,eventId=2,eventIndex=3),self.hits,'second','different-source','other-geometry');w.finish(1)
            merge_exports(out,[a,b]);events,_=load_events(out)
            self.assertEqual(len(events),2)
            self.assertEqual(events[1]['geometry'],'other-geometry')
            with self.assertRaisesRegex(ValueError,'Duplicate event'):merge_exports(Path(d)/'bad.jsonl',[a,a])
    def test_export_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hits.jsonl';hits=copy.deepcopy(self.hits);hits[0].update(x=1.2,y=3.,z=10.,hitTimeNS=5.)
            self.export(p,hits);events,_=load_events(p)
            self.assertEqual(events[0]['hits'][0]['x_cm'],1.2)
            self.assertEqual(events[0]['hits'][0]['time_ns'],5.)
    def test_bad_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'schema.json'
            for feature in [dict(dtype='int64',description='f',missing_value=-999.5),dict(dtype='float64',description='f',missing_value=-999,histogram={'bins':0,'range':[0,1]})]:
                p.write_text(json.dumps({'version':'v1','features':{'f':feature}}))
                with self.assertRaises(ValueError):schema(p)
    def test_label_independence(self):
        events,_=load_events(ROOT/'feature_dev/samples/small.jsonl')
        e=events[1];a=evaluate(self.g,e['hits'],{},self.gap)
        e['group']='changed label';e['geometry']='other';e['event_id']['eventId']=999
        self.assertEqual(a,evaluate(self.g,e['hits'],{},self.gap))

if __name__=='__main__':unittest.main()
