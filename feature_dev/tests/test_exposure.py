"""Small audits: exposure accounting, shard coverage and histogram errors."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exposure import validate_audit, resolve_normalization, normalize_counts


def source(uid, lumi, count, probability=1):
    return dict(source_uid=uid, luminosity_fb=lumi, selected_entries=count,
                sampling_probability=probability, sampling_method='uniform_random',
                processed_ranges=[[0, count]] if count else [])


def fixture():
    events = [dict(source_uid='ve', group=g, geometry='geo') for g in ('CC_nue','NC_nue')]
    events += [dict(source_uid='vm', group='CC_numu', geometry='geo')]
    def population(role, sources):
        return dict(role=role, coverage_evidence='test fixture complete parent catalogue', sources=sources)
    audit = dict(version=1, sample_sha256='test-sha', populations={
        've':population('mc',[source('ve',10,2),source('empty',5,0)]),
        'vm':population('mc',[source('vm',30,1)]),
        'target':population('reference_data',[source('data',20,0,.1)])}, groups=[
        dict(group=e['group'],geometry='geo',population='vm' if e['source_uid']=='vm' else 've',target_population='target') for e in events])
    return audit, events


class ExposureTests(unittest.TestCase):
    def setUp(self):self.audit,self.events=fixture()
    def validate(self):return validate_audit(self.audit,'test-sha',self.events,{})
    def test_abcd_ratios_and_empty_parent(self):
        records={r['group']:r for r in self.validate()}
        self.assertAlmostEqual(records['CC_nue']['scale'],2/15)
        self.assertEqual(records['CC_nue']['scale'],records['NC_nue']['scale'])
        self.assertAlmostEqual(records['CC_numu']['scale'],2/30)
    def test_shards_do_not_repeat_exposure(self):
        self.audit['populations']['ve']['sources'][0]['processed_ranges']=[[1,2],[0,1]]
        self.assertEqual(self.validate()[0]['source_exposure_fb'],15)
    def test_duplicate_source_rejected(self):
        self.audit['populations']['ve']['sources'].append(copy.deepcopy(self.audit['populations']['ve']['sources'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.validate()
    def test_missing_and_overlapping_shards(self):
        for ranges in ([[0,1]],[[0,2],[1,2]],[[1,2]]):
            self.audit['populations']['ve']['sources'][0]['processed_ranges']=ranges
            with self.assertRaises(ValueError):self.validate()
    def test_sample_count_and_sha_mismatch(self):
        self.events.pop()
        with self.assertRaisesRegex(ValueError,'counts'):self.validate()
        self.audit['sample_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'SHA'):self.validate()
    def test_invalid_lumi(self):
        for value in (0,-1,float('nan'),float('inf'),True):
            self.audit['populations']['ve']['sources'][0]['luminosity_fb']=value
            with self.assertRaises(ValueError):self.validate()
    def test_unknown_and_wrong_population(self):
        self.audit['groups'][0]['population']='vm'
        with self.assertRaisesRegex(ValueError,'another parent'):self.validate()
    def test_capped_provenance(self):
        with self.assertRaisesRegex(ValueError,'Capped'):
            validate_audit(self.audit,'test-sha',self.events,{'inputs':[{'max_events':100}]})
    def test_data_keeps_weight_one_at_effective_exposure(self):
        audit=dict(version=1,sample_sha256='data',populations={'data':dict(role='data',coverage_evidence='fixture',sources=[source('data',20,1,.1)])},groups=[dict(group='data',geometry='geo',population='data')])
        record=validate_audit(audit,'data',[dict(source_uid='data',group='data',geometry='geo')],{})[0]
        self.assertEqual(record['scale'],1)
        self.assertEqual(record['target_exposure_fb'],2)
    def test_sampling_requires_uniform_data(self):
        self.audit['populations']['target']['sources'][0]['sampling_method']='first_n'
        with self.assertRaisesRegex(ValueError,'uniform_random'):self.validate()
    def test_policy(self):
        for dataset in ('testbeam','synthetic','real-data','mc-neutrino'):
            self.assertEqual(resolve_normalization('auto',dataset,'small',None,'test-sha',self.events,{})[0]['mode'],'shape')
            with self.assertRaises(ValueError):resolve_normalization('yield',dataset,'small',None,'test-sha',self.events,{})
        with self.assertRaisesRegex(ValueError,'exposure-manifest'):
            resolve_normalization('auto','mc-neutrino','full',None,'test-sha',self.events,{})
    def test_abcd_reference_values(self):
        self.audit['populations']['ve']['sources'][0]['luminosity_fb']=14635
        self.audit['populations']['vm']['sources'][0]['luminosity_fb']=13350
        target=self.audit['populations']['target']['sources'][0]
        target['sampling_probability']=1
        for lumi in (19.780680927948,1.960319252235):
            target['luminosity_fb']=lumi
            records={r['group']:r for r in self.validate()}
            self.assertAlmostEqual(records['CC_nue']['scale'],lumi/14640)
            self.assertAlmostEqual(records['CC_numu']['scale'],lumi/13350)
    def test_mc_sampling_and_wrong_dataset_role(self):
        with self.assertRaisesRegex(ValueError,'role'):
            resolve_normalization('yield','real-data','full',self.audit,'test-sha',self.events,{})
        self.audit['populations']['ve']['sources'][0]['sampling_probability']=.1
        with self.assertRaisesRegex(ValueError,'MC subsampling'):self.validate()
    def test_missing_reference_and_evidence(self):
        self.audit['groups'][0]['target_population']='absent'
        with self.assertRaisesRegex(ValueError,'target population'):self.validate()
        self.audit['populations']['ve'].pop('coverage_evidence')
        with self.assertRaisesRegex(ValueError,'coverage_evidence'):self.validate()
    def test_sumw2_flows_and_shape(self):
        result=normalize_counts([2,3],{'mode':'yield','scale':2},1,4)
        self.assertEqual(result['normalized'],[4,6]);self.assertEqual(result['sumw2'],[8,12])
        self.assertEqual(result['underflow_sumw2'],4);self.assertEqual(result['normalized_overflow'],8)
        self.assertEqual(normalize_counts([2,3],{'mode':'shape'})['normalized'],[.4,.6000000000000001])
        self.assertEqual(normalize_counts([0],{'mode':'shape'})['sumw2'],[0])

if __name__=='__main__':unittest.main()
