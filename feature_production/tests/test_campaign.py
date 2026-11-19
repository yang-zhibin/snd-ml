"""Artificial catalogue/ROOT fixtures for production planning and coverage."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import uproot
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT, read_json, write_json, sha, canonical
from distributions import distributions
from worker import attach_exposure
module_spec=importlib.util.spec_from_file_location('production_cli',ROOT/'feature_production/cli.py')
production_cli=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(production_cli)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def catalogue(self):
        raw=self.root/'digi.root';raw.write_text('fixture')
        geo=self.root/'geo.root';geo.write_text('geometry')
        rows=[dict(family='mc_neutrino',source_uid=uid,sample_type='MC_neutrino',subfolder='ve',
            digi_path=str(raw),geo_path=str(geo),cutflow_path='',selection_mode='direct_input',input_tree='cbmsim',
            source_domain_id='1',source_geometry_id='geo',input_event_count=str(count),selected_event_count=str(count),
            lumi_per_file='15',exposure_kind='simulated_luminosity_fb') for uid,count in [('nonempty',2),('empty',0)]]
        path=self.root/'catalogue.csv'
        with path.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        config=self.root/'config.json'
        write_json(config,dict(version=1,datasets={'mc-neutrino':dict(sources=dict(catalogues=[str(path)],events_per_group=2))}))
        return config
    def args(self,sample='full'):
        return SimpleNamespace(plugin='silvia',parameters='{}',dataset='mc-neutrino',dataset_config=self.catalogue(),
            output_root=self.root/'output',feature_root=None,run='test',sample=sample,files_per_group=1,events_per_file=2,
            exposure_manifest=None,normalization='shape')
    def test_full_preserves_zero_sources(self):
        plan=production_cli.make_plan(self.args())
        self.assertEqual(len(plan['jobs']),2)
        self.assertTrue(all(j['max_events']==0 for j in plan['jobs']))
    def test_small_is_capped(self):
        plan=production_cli.make_plan(self.args('small'))
        self.assertEqual(len(plan['jobs']),1);self.assertEqual(plan['jobs'][0]['max_events'],2)
    def test_full_yield_needs_bound_audit(self):
        args=self.args();args.normalization='yield'
        with self.assertRaisesRegex(ValueError,'campaign_sha256'):production_cli.make_plan(args)
    def test_output_roots_are_per_run(self):
        args=self.args();args.feature_root=self.root/'eos-fixture'
        plan=production_cli.make_plan(args)
        self.assertEqual(plan['directory'],str(self.root/'output/mc-neutrino/test'))
        self.assertEqual(plan['source_directory'],str(self.root/'eos-fixture/mc-neutrino/test/sources'))
    def fixture(self,mode='shape'):
        plan=dict(directory=str(self.root/'run'),source_directory=str(self.root/'sources'),dataset='mc-neutrino',
                  sample='full',campaign_sha256='fixture-campaign',normalization=mode,
                  plugin=dict(name='silvia',features={'density':dict(dtype='float64',description='density',missing_value=-999,histogram=dict(bins=2,range=[0,4]))}),jobs=[])
        Path(plan['directory']).mkdir()
        for index,(uid,count,lumi) in enumerate([('nonempty',2,10),('empty',0,5)]):
            row=dict(source_uid=uid,family='mc_neutrino',source_geometry_id='geo')
            key='mc_neutrino:'+uid;plan['jobs'].append(dict(record=row,source_key=key))
            directory=Path(plan['source_directory'])/str(index);directory.mkdir(parents=True)
            path=directory/'features.root'
            with uproot.recreate(path) as f:
                f.mktree('sndData',dict(pdgCode='int32',silvia__density='float64'))
                if count:f['sndData'].extend(dict(pdgCode=np.array([12,12]),silvia__density=np.array([1.,3.])))
            context=dict(version=1,source_uid=uid,audit_source_id=key,parent_luminosity_fb=lumi,sampling_probability=1,
                         exposure_valid=True,effective_luminosity_fb=lumi,campaign_sha256=plan['campaign_sha256'])
            attach_exposure(path,context)
            write_json(directory/'validation.json',dict(validation_passed=True,feature_sha256=sha(path),written_event_count=count))
            write_json(directory/'complete.json',{p.name:sha(p) for p in directory.iterdir()})
        def source(uid,lumi,count):
            return dict(source_uid=uid,luminosity_fb=lumi,sampling_probability=1,selected_entries=count,processed_ranges=[[0,count]] if count else [])
        plan['exposure_audit']=dict(version=1,campaign_sha256=plan['campaign_sha256'],populations={
            've':dict(role='mc',coverage_evidence='fixture',sources=[source('mc_neutrino:nonempty',10,2),source('mc_neutrino:empty',5,0)]),
            'data':dict(role='reference_data',coverage_evidence='fixture',sources=[source('reference',2,0)])},
            groups=[dict(group='CC_nue',geometry='geo',population='ve',target_population='data')])
        return plan
    def test_streaming_yield_includes_empty_source(self):
        plan=self.fixture('yield')
        with patch('distributions.render'):distributions(plan)
        out=Path(plan['directory'])/'plots'
        hist=read_json(out/'histograms.json')['density']['curves'][0]
        self.assertEqual(hist['counts'],[1,1]);self.assertAlmostEqual(hist['normalization']['scale'],2/15)
        np.testing.assert_allclose(hist['sumw2'],[(2/15)**2]*2)
        self.assertEqual(len(read_json(out/'coverage.json')['sources']),2)
    def test_missing_source_receipt_fails(self):
        plan=self.fixture()
        (Path(plan['source_directory'])/'1/complete.json').unlink()
        with self.assertRaises(FileNotFoundError):distributions(plan)
    def test_audit_count_mismatch_fails(self):
        plan=self.fixture('yield')
        plan['exposure_audit']['populations']['ve']['sources'][0].update(selected_entries=3,processed_ranges=[[0,3]])
        with self.assertRaisesRegex(ValueError,'counts'):distributions(plan)
    def test_shape_fractions(self):
        plan=self.fixture()
        with patch('distributions.render'):distributions(plan)
        hist=read_json(Path(plan['directory'])/'plots/histograms.json')['density']['curves'][0]
        self.assertEqual(hist['normalized'],[.5,.5])

if __name__=='__main__':unittest.main()
