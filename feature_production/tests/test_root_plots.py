"""Physics/format acceptance for the shared ROOT plotting path."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
import uproot
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plot_settings import resolve_axis, geometry_key
from distributions import distributions, resolve, fill, render
import test_campaign as fixtures


class RangeTests(unittest.TestCase):
    def test_auto_negative_and_max_edge(self):
        r=resolve_axis({},3,-5,2)
        self.assertLess(r['x_min'],-5);self.assertGreater(r['x_max'],2)
    def test_constant_empty_integer(self):
        self.assertEqual(resolve_axis({},0,None,None)['range_reason'],'no_valid_values')
        self.assertLess(resolve_axis({},2,0,0)['x_min'],0)
        r=resolve_axis(dict(dtype='int64'),2,1,3)
        self.assertEqual((r['x_min'],r['x_max'],r['bins']),(.5,3.5,3))
    def test_explicit_and_invalid(self):
        r=resolve_axis(dict(range=[0,1],bins=10),2,-9,99)
        self.assertEqual(r['x_max'],1)
        for c in [dict(range=[1,0]),dict(range=[0,float('inf')]),dict(bins=0),dict(logx=True,range=[0,1])]:
            with self.assertRaises(ValueError):resolve_axis(c,1,0,0)


class BackendTests(unittest.TestCase):
    setUp=fixtures.CampaignTests.setUp
    tearDown=fixtures.CampaignTests.tearDown
    fixture=fixtures.CampaignTests.fixture
    def test_actual_root_plotters_and_two_page_pdf(self):
        import plot_comparison, plot_group_hist2d
        plan=self.fixture()
        plan['plot_settings']['histograms']['features2d']={'density_pair':dict(x='silvia__density',y='silvia__density',x_bins=2,y_bins=2)}
        with patch.object(plot_comparison,'draw_overlay',wraps=plot_comparison.draw_overlay) as one, \
             patch.object(plot_group_hist2d,'plot',wraps=plot_group_hist2d.plot) as two:
            distributions(plan)
            self.assertEqual(one.call_count,1);self.assertEqual(two.call_count,1)
        base=Path(plan['directory']);geo=geometry_key('geo')
        info=subprocess.check_output(['pdfinfo',str(base/'plots/2d/density_pair/selected'/'plot.pdf')],text=True)
        self.assertRegex(info,r'Pages:\s+2')
        with (base/'plots/1d/silvia__density/selected'/'summary.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        row=next(r for r in rows if r['group']=='CC_nue')
        self.assertEqual(float(row['scale_factor']),.5)
        self.assertEqual(float(row['scaled_integral']),1.)
        self.assertEqual(len(rows),4) # empty neutrino classes are retained
        self.assertFalse((base/'plots/report.pdf').exists())
    def test_2d_yield_does_not_scale_profiles(self):
        plan=self.fixture('yield')
        plan['plot_settings']['histograms']['features2d']={'density_pair':dict(x='silvia__density',y='silvia__density')}
        distributions(plan)
        base=Path(plan['directory']);geo=geometry_key('geo')
        with (base/'plots/2d/density_pair/selected'/'summary.csv').open() as stream:
            row=next(r for r in csv.DictReader(stream) if r['particle_group']=='CC_nue')
        self.assertEqual(row['scale_factor'],'')
        self.assertAlmostEqual(float(row['scaled_integral']),4/15)
        self.assertEqual(row['normalization'],'yield')
    def test_legacy_histogram_entry_points_match(self):
        from types import SimpleNamespace
        from make_group_hist import make_group_histogram
        from make_group_hist2d import make_histogram
        from common import write_json
        plan=self.fixture()
        plan['plot_settings']['histograms']['features2d']={'pair':dict(x='silvia__density',y='silvia__density',x_range=[0,4],y_range=[0,4],x_bins=2,y_bins=2)}
        resolve(plan);fill(plan)
        metadata=self.root/'metadata.csv'
        with metadata.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['region','particle_group','feature_partition_path','lumi_per_partition','n_events'])
            writer.writeheader()
            writer.writerow(dict(region='selected',particle_group='CC_nue',feature_partition_path=str(Path(plan['source_directory'])/'0/features.root'),lumi_per_partition=10,n_events=2))
        histcfg=self.root/'hist.json';evalcfg=self.root/'eval.json'
        write_json(histcfg,dict(features={'silvia__density':dict(x_min=0,x_max=4,bin_width=2)},features2d={'pair':dict(x_expression='silvia__density',y_expression='silvia__density',x_min=0,x_max=4,x_bins=2,y_min=0,y_max=4,y_bins=2,x_title='density',y_title='density')}))
        write_json(evalcfg,dict(cuts=dict(definitions={},base_cut_options={'none':[]},extra_cut_options={'none':''}),metadata=dict(tree_name='sndData')))
        args=SimpleNamespace(repo_root=str(self.root),metadata_csv=str(metadata),hist_config=str(histcfg),eval_config=str(evalcfg),region='selected',particle_group='CC_nue',feature='silvia__density',pair='pair',base_cut_key='none',extra_cut_key='none',tree=None,friend_tree=None,output=str(self.root/'legacy1.root'))
        make_group_histogram(args)
        args.output=str(self.root/'legacy2.root');make_histogram(args)
        base=Path(plan['directory'])/'histograms';geo=geometry_key('geo')
        for legacy,new,key in [('legacy1.root','1d/silvia__density','hist'),('legacy2.root','2d/pair','hist2d')]:
            with uproot.open(self.root/legacy) as old, uproot.open(base/new/'selected'/'CC_nue.root') as current:
                np.testing.assert_array_equal(old[key].values(flow=True),current[key].values(flow=True))
                np.testing.assert_array_equal(old[key].variances(flow=True),current[key].variances(flow=True))

    def test_testbeam_labels_and_geometry_are_preserved(self):
        plan=self.fixture()
        plan['dataset']='testbeam'
        for index,job in enumerate(plan['jobs']):
            job['record'].update(family='tb:2023',source_pdg_code='11',source_energy_gev=str(100+index),source_geometry_id='beam_geo_'+str(index))
        resolve(plan);fill(plan);render(plan)
        records=json.loads((Path(plan['directory'])/'histograms/summary.json').read_text())
        self.assertEqual(len(records),2)
        self.assertNotEqual(records[0]['geometries'],records[1]['geometries'])
        self.assertIn(':pdg11:',records[0]['group'])
        self.assertNotIn(':',Path(records[0]['path']).name)

    def test_pair_feature_cuts_are_selection_not_missing(self):
        plan=self.fixture()
        cfg=plan['plot_settings']['histograms']
        cfg['features']['other']=dict(expression='silvia__density',feature_cut='silvia__density > 2')
        cfg['features2d']={'pair':dict(x='silvia__density',y='other')}
        resolve(plan);fill(plan)
        records=json.loads((Path(plan['directory'])/'histograms/summary.json').read_text())
        row=next(r for r in records if r['path'].startswith('2d/') and r['group']=='CC_nue')
        self.assertEqual(row['selected_entries'],1)
        self.assertEqual(row['valid_entries'],1)
        self.assertEqual(row['missing_entries'],0)

    def test_geometry_combination_preserves_audited_weights(self):
        from common import write_json, sha
        from worker import attach_exposure
        plan=self.fixture('yield')
        second=Path(plan['source_directory'])/'1/features.root'
        with uproot.open(second) as f: context=json.loads(str(f['production_exposure_json']))
        with uproot.recreate(second) as f:
            f.mktree('sndData',dict(pdgCode='int32',silvia__density='float64'))
            f['sndData'].extend(dict(pdgCode=np.array([12,12]),silvia__density=np.array([7.,9.])))
        attach_exposure(second,context)
        write_json(second.parent/'validation.json',dict(validation_passed=True,feature_sha256=sha(second),written_event_count=2))
        write_json(second.parent/'complete.json',{p.name:sha(p) for p in second.parent.iterdir() if p.name!='complete.json'})
        plan['jobs'][1]['record']['source_geometry_id']='second_geometry'
        audit=plan['exposure_audit']; second_source=audit['populations']['ve']['sources'].pop()
        second_source.update(selected_entries=2,processed_ranges=[[0,2]])
        audit['populations']['ve2']=dict(role='mc',coverage_evidence='fixture',sources=[second_source])
        audit['groups'].append(dict(group='CC_nue',geometry='second_geometry',population='ve2',target_population='data'))
        plan['plot_settings']['histograms']['features']['silvia__density']={'bins':4}
        plan['plot_settings']['histograms']['features2d']={'pair':dict(x='silvia__density',y='silvia__density')}
        distributions(plan)
        base=Path(plan['directory'])
        resolved=json.loads((base/'config/resolved.json').read_text())['1d/silvia__density/selected']
        self.assertEqual((resolved['observed_min'],resolved['observed_max']),(1.,9.))
        for kind,name,raw,yielded in [('1d','silvia__density','hist','hist_yield'),('2d','pair','hist2d','hist2d_yield')]:
            with uproot.open(base/'histograms'/kind/name/'selected/CC_nue.root') as f:
                self.assertEqual(np.sum(f[raw].values(flow=True)),4)
                self.assertAlmostEqual(np.sum(f[yielded].values(flow=True)),1.2)
                self.assertAlmostEqual(np.sum(f[yielded].variances(flow=True)),.4)
                meta=json.loads(str(f['production_histogram_json']))
                self.assertEqual(meta['geometries'],['geo','second_geometry'])
        self.assertEqual(len(list((base/'plots/1d').rglob('plot.pdf'))),1)
        self.assertFalse(any(p.name.startswith('geo_') for p in (base/'plots').rglob('*')))

    def test_missing_plot_branch_rejected(self):
        plan=self.fixture()
        plan['plot_settings']['histograms']['features']['absent']={}
        with self.assertRaisesRegex(ValueError,'Missing plot branches'):resolve(plan)

    def test_filter_missing_finite_and_joint_pairs(self):
        plan=self.fixture()
        path=Path(plan['source_directory'])/'0/features.root'
        from common import sha, write_json
        from worker import attach_exposure
        context=json.loads(str(uproot.open(path)['production_exposure_json']))
        with uproot.recreate(path) as f:
            f.mktree('sndData',dict(pdgCode='int32',silvia__density='float64',other='float64'))
            f['sndData'].extend(dict(pdgCode=np.array([12,14,12,12,12]),
                 silvia__density=np.array([1,7,-999,np.nan,3]),other=np.array([2,4,8,9,-999])))
        attach_exposure(path,context)
        write_json(path.parent/'validation.json',dict(validation_passed=True,feature_sha256=sha(path),written_event_count=5))
        write_json(path.parent/'complete.json',{p.name:sha(p) for p in path.parent.iterdir() if p.name!='complete.json'})
        empty=Path(plan['source_directory'])/'1/features.root'
        with uproot.open(empty) as f: empty_context=json.loads(str(f['production_exposure_json']))
        with uproot.recreate(empty) as f:
            f.mktree('sndData',dict(pdgCode='int32',silvia__density='float64',other='float64'))
        attach_exposure(empty,empty_context)
        record=json.loads((empty.parent/'validation.json').read_text());record['feature_sha256']=sha(empty)
        write_json(empty.parent/'validation.json',record)
        write_json(empty.parent/'complete.json',{p.name:sha(p) for p in empty.parent.iterdir() if p.name!='complete.json'})
        cfg=plan['plot_settings']['histograms']
        cfg['features']={'silvia__density':{'bins':4},'other':{'bins':4}}
        cfg['features2d']={'pair':dict(x='silvia__density',y='other')}
        cfg['selections']={'selected':'true','cut':'silvia__density > 2'}
        resolve(plan);fill(plan)
        base=Path(plan['directory']);geo=geometry_key('geo')
        ranges=json.loads((base/'config/resolved.json').read_text())
        r=ranges['1d/silvia__density/selected']
        self.assertEqual(r['valid_values'],3);self.assertEqual(r['observed_max'],7)
        r=ranges['2d/pair/selected']
        self.assertEqual(r['x_resolution']['valid_values'],2)
        with uproot.open(base/'histograms/1d/silvia__density/selected'/'CC_numu.root') as f:
            self.assertEqual(sum(f['hist'].values()),1)
        with uproot.open(base/'histograms/1d/silvia__density/cut'/'CC_nue.root') as f:
            self.assertEqual(sum(f['hist'].values()),1)

if __name__=='__main__':unittest.main()
