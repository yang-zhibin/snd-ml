import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import array
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT, plugin_spec, sha, canonical
from feature_plugin import Plugin, validate_root_plugin, validate_spec
import ROOT as R


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.spec=plugin_spec('silvia',{})
        self.hits=[dict(station=1,isVertical=True,layer_channel=i,qdc=2.,x=i*.1,y=0.,z=300.,hitTimeNS=2.) for i in (1,2,5)]
    def test_known_values_and_no_mutation(self):
        before=copy.deepcopy(self.hits);values=Plugin(self.spec).calculate(self.hits)
        self.assertEqual(values['scifi_vertical_cluster1_density'],2)
        self.assertAlmostEqual(values['leading_charge_fraction'],2/3)
        self.assertEqual(values['scifi_vertical_cluster1_station'],1)
        self.assertEqual(self.hits,before)
    def test_missing_coordinates_are_missing(self):
        for h in self.hits:h.pop('x')
        values=Plugin(self.spec).calculate(self.hits)
        self.assertEqual(values['scifi_vertical_cluster1_density'],-999)
        self.assertEqual(values['scifi_vertical_cluster1_density_fraction'],-999)
    def test_empty_event(self):
        values=Plugin(self.spec).calculate([])
        self.assertEqual(values['leading_charge_fraction'],-999)
    def test_code_tamper(self):
        self.spec['source_hashes'][self.spec['algorithm']]='wrong'
        with self.assertRaisesRegex(ValueError,'source changed'):Plugin(self.spec)
    def test_invalid_spec(self):
        for key,value in [('name','bad/name'),('parameters',[]),('features',{})]:
            bad=copy.deepcopy(self.spec);bad[key]=value
            with self.assertRaises(ValueError):validate_spec(bad)
    def test_root_round_trip_and_branch_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'features.root'
            f=R.TFile(str(path),'RECREATE');tree=R.TTree('sndData','test')
            original=array.array('d',[123.]);tree.Branch('scifi_vertical_cluster1_density',original,'scifi_vertical_cluster1_density/D')
            plugin=Plugin(self.spec);plugin.create_branches(tree)
            plugin.fill(self.hits);tree.Fill()
            plugin.fill([]);tree.Fill()
            tree.Write();plugin.write_metadata(R);f.Close()
            f=R.TFile.Open(str(path))
            self.assertTrue(validate_root_plugin(f,self.spec))
            tree=f.Get('sndData');tree.GetEntry(0)
            self.assertEqual(tree.scifi_vertical_cluster1_density,123)
            self.assertEqual(tree.silvia__scifi_vertical_cluster1_density,2)
            tree.GetEntry(1);self.assertEqual(tree.silvia__scifi_vertical_cluster1_density,-999)
            f.Close()
    def test_collision_rejected(self):
        tree=R.TTree('collision','test');buf=array.array('d',[0.])
        tree.Branch('silvia__leading_charge_fraction',buf,'silvia__leading_charge_fraction/D')
        with self.assertRaisesRegex(ValueError,'collision'):Plugin(self.spec).create_branches(tree)
    def test_embedded_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'bad.root';f=R.TFile(str(path),'RECREATE')
            R.TObjString(canonical(self.spec)).Write('feature_plugin_json')
            R.TObjString('wrong').Write('feature_plugin_sha256');f.Close()
            f=R.TFile.Open(str(path))
            with self.assertRaisesRegex(ValueError,'checksum'):validate_root_plugin(f)
            f.Close()
    def test_mutating_plugin_rejected(self):
        plugin=Plugin(self.spec)
        def mutate(h,p):h.clear();return {}
        plugin.module.cluster_hits=mutate
        plugin.module.calculate_features=lambda h,c,p:{k:v['missing_value'] for k,v in self.spec['features'].items()}
        with self.assertRaisesRegex(ValueError,'modified'):plugin.calculate(self.hits)
    def test_inherited_baseline_outputs_not_exported(self):
        plugin=Plugin(self.spec);values=plugin.calculate(self.hits)
        self.assertEqual(set(values),set(self.spec['features']))
        self.assertNotIn('scifi_cluster_total_nhits',values)

if __name__=='__main__':unittest.main()

class ShardPluginTests(unittest.TestCase):
    def test_incompatible_plugin_shards_rejected(self):
        from merge_unified_shower_feature_shards import validate_receipts
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'source.root';path.write_bytes(b'fixture')
            base=dict(validation_passed=True,validation_record='fixture',source_uid='same',
                      feature_path=str(path),feature_sha256=sha(path),available_selected_event_count=2,
                      conversion_max_events=0,written_event_count=1)
            a=dict(base,selected_entry_start=0,selected_entry_stop=1,feature_plugin_sha256='first')
            b=dict(base,selected_entry_start=1,selected_entry_stop=2,feature_plugin_sha256='second')
            with self.assertRaisesRegex(RuntimeError,'feature_plugin_sha256'):validate_receipts([a,b])
            b['feature_plugin_sha256']='first';b['production_exposure_sha256']='different'
            with self.assertRaisesRegex(RuntimeError,'production_exposure_sha256'):validate_receipts([a,b])
    def test_source_exposure_tree_copied_once(self):
        from merge_unified_shower_feature_shards import copy_metadata
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'source.root';target=Path(temporary)/'merged.root'
            source=R.TFile(str(path),'RECREATE');tree=R.TTree('source_exposure','exposure')
            lumi=array.array('d',[15.]);tree.Branch('luminosity_fb',lumi,'luminosity_fb/D');tree.Fill();tree.Write();source.Close()
            source=R.TFile.Open(str(path));output=R.TFile(str(target),'RECREATE');copy_metadata(source,output);output.Close();source.Close()
            output=R.TFile.Open(str(target));tree=output.Get('source_exposure')
            self.assertEqual(tree.GetEntries(),1);tree.GetEntry(0);self.assertEqual(tree.luminosity_fb,15)
            output.Close()
