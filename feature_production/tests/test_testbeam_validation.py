"""Regression checks for selected counts, beam energy and input preflight."""
import array
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import ROOT
import uproot
sys.path[:0] = [str(Path(__file__).resolve().parents[2]/p) for p in ('convertData','feature_dev')]
from source_provenance import beam_energy_matches
from validate_unified_shower_feature_file import validate_nominal_source_values, identity_cutflow_entries
from sources import check_selection_headers


class TestTestbeamValidation(unittest.TestCase):
    def tree(self, pdg=211, energy=100.000097):
        tree = ROOT.TTree('features', 'fixture'); tree.SetDirectory(0)
        values = [array.array('i', [pdg]), array.array('d', [energy])]
        tree.Branch('source_pdg_code', values[0], 'source_pdg_code/I')
        tree.Branch('source_energy_gev', values[1], 'source_energy_gev/D')
        tree.Fill()
        return tree, values

    def test_nominal_mc_energy_and_exact_pdg(self):
        args = SimpleNamespace(expected_source_domain_id=4, expected_source_pdg_code=211, expected_source_energy_gev=100.)
        for energy in (100.000097, 101., 110., float('nan')):
            tree, buffers = self.tree(energy=energy)
            if beam_energy_matches(energy,100.):
                validate_nominal_source_values(tree,args)
            else:
                with self.assertRaisesRegex(RuntimeError, 'beam energy'):
                    validate_nominal_source_values(tree,args)
        tree, buffers = self.tree(pdg=11)
        with self.assertRaisesRegex(RuntimeError, 'PDG'):
            validate_nominal_source_values(tree,args)
        args.expected_source_domain_id=5
        tree, buffers=self.tree()
        with self.assertRaisesRegex(RuntimeError, 'beam energy'):
            validate_nominal_source_values(tree,args)

    def test_selected_count_is_not_raw_count(self):
        tree = ROOT.TTree('cutFlowSummary','fixture'); tree.SetDirectory(0)
        value = array.array('q',[0]); tree.Branch('entry',value,'entry/L')
        for i in range(1200): value[0]=i;tree.Fill()
        self.assertEqual(len(identity_cutflow_entries(tree,1200)),1200)
        with self.assertRaisesRegex(RuntimeError,'1200 rows, expected 18000'):
            identity_cutflow_entries(tree,18000)

    def test_preflight_reports_all_bad_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'wrong.root'
            with uproot.recreate(path) as f:f['wrong']='not a cutflow'
            rows=[dict(source_uid='missing',selection_mode='indexed_cutflow',cutflow_path=str(path)+'absent'),
                  dict(source_uid='wrong-tree',selection_mode='identity_cutflow',cutflow_path=str(path))]
            with self.assertRaises(ValueError) as context:check_selection_headers(rows)
            self.assertIn('missing',str(context.exception));self.assertIn('wrong-tree',str(context.exception))
            check_selection_headers([dict(source_uid='direct',selection_mode='direct_input')])

if __name__=='__main__':unittest.main()

class TestCatalogueRepair(unittest.TestCase):
    def test_publish_only_after_all_sources_pass(self):
        import csv
        import json
        import numpy as np
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'snakemake/metadata'))
        from repair_testbeam_mc_selected_counts import repair
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); rows=[]; upstream=[]
            for i in range(2):
                raw=root/f'raw{i}.root'; cut=root/f'cut{i}.root'
                with uproot.recreate(raw) as f:
                    f.mktree('cbmsim',{'value':'int32'}).extend({'value':np.arange(18,dtype=np.int32)})
                with uproot.recreate(cut) as f:
                    f.mktree('cutFlowSummary',{'entry':'int64'}).extend({'entry':np.arange(12,dtype=np.int64)})
                row=dict(source_uid=str(i),family='testbeam_mc_2024',selection_mode='identity_cutflow',
                         digi_path=str(raw),cutflow_path=str(cut),geo_path='geo',input_tree='cbmsim',input_event_count='18')
                rows.append(row);upstream.append(dict(row,n_events='12'))
            def write(path, data):
                with path.open('w') as f:
                    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
            manifest=root/'manifest.csv';master=root/'master.csv';write(manifest,rows)
            original=manifest.read_bytes();upstream[1]['n_events']='13';write(master,upstream)
            with self.assertRaisesRegex(ValueError,'selected count differs'):
                repair(manifest,master,root/'report.json',True)
            self.assertEqual(manifest.read_bytes(),original)
            upstream[1]['n_events']='12';write(master,upstream)
            report=repair(manifest,master,root/'report.json',True)
            self.assertEqual(Path(report['backup']).read_bytes(),original)
            actual=list(csv.DictReader(manifest.open()))
            for before,after in zip(rows,actual):
                self.assertEqual(after.pop('selected_event_count'),'12');self.assertEqual(before,after)
