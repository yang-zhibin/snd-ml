"""Preparation acceptance: bounded source selection, parity and cache integrity."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import write_json, read_json, sha
from sources import build_plan, converter_arguments, resolve_prepared, verify_inputs
from prepare import validate_export, assemble, export_source
from selected_hits import export_group
from test_datasets import export


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalogue = self.root/'catalogue.csv'
        self.config = self.root/'datasets.json'
        self.rows = []
        for i in range(6):
            digi = self.root/f'digi{i}.root'; digi.write_text('input')
            geo = self.root/'geo.root'; geo.write_text('geometry')
            cut = self.root/f'cut{i}.root'; cut.write_text('selection')
            self.rows.append(dict(source_uid=f'source-{i}', family='mc_neutrino',
                subfolder='nue' if i % 2 else 'numu', digi_path=str(digi), geo_path=str(geo),
                cutflow_path=str(cut), sample_type='MC_neutrino', source_domain_id='1',
                selection_mode='external_cutflow', input_tree='cbmsim', selected_event_count='10',
                source_geometry_id='', source_geometry_checksum=''))
        self.save()
    def tearDown(self):
        self.temp.cleanup()
    def save(self, dataset='mc-neutrino', required=('CC_nue',)):
        with self.catalogue.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader(); writer.writerows(self.rows)
        write_json(self.config, {'version': 1, 'datasets': {dataset: {
            'required_groups': list(required), 'sources': {'catalogues': [str(self.catalogue)],
            'files_per_group': 1, 'events_per_file': 5, 'events_per_group': 3}}}})
    def plan(self, dataset='mc-neutrino', **kwargs):
        return build_plan(dataset, self.config, self.root/'out', **kwargs)
    def test_bounded_flavour_coverage(self):
        p = self.plan()
        self.assertEqual(len(p['jobs']), 2)
        self.assertEqual({j['record']['subfolder'] for j in p['jobs']}, {'nue', 'numu'})
        self.assertTrue(all(j['max_events'] == 5 for j in p['jobs']))
        self.assertFalse((self.root/'out').exists())
        chosen = [j['record']['source_uid'] for j in p['jobs']]
        self.rows.reverse(); self.save()
        self.assertEqual(chosen, [j['record']['source_uid'] for j in self.plan()['jobs']])
    def test_inputs_and_sampling_invalidate_cache(self):
        p = self.plan()
        self.assertNotEqual(p['signature'], self.plan(events_per_file=6)['signature'])
        Path(p['jobs'][0]['record']['digi_path']).write_text('changed input')
        self.assertNotEqual(p['signature'], self.plan()['signature'])
        with self.assertRaisesRegex(ValueError, 'changed after planning'):
            verify_inputs(p['jobs'][0])
    def test_missing_input_is_error(self):
        p = self.plan()
        Path(p['jobs'][0]['record']['digi_path']).unlink()
        with self.assertRaisesRegex(ValueError, 'missing or unreadable'):
            self.plan()
    def test_duplicate_sources_rejected(self):
        self.rows.append(self.rows[0]); self.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.plan()
    def test_zero_event_rows_excluded(self):
        self.rows[0]['selected_event_count'] = '0'; self.save()
        self.assertNotIn('source-0', [j['record']['source_uid'] for j in self.plan()['jobs']])
    def test_oversized_request_rejected(self):
        for options in ({'files_per_group': 0}, {'events_per_file': 1001}):
            with self.assertRaises(ValueError): self.plan(**options)
    def test_truth_labels_and_signs(self):
        for code, label in [(12, 'CC_nue'), (14, 'CC_numu'), (112, 'NC_nue'), (114, 'NC_numu')]:
            for sign in (-1, 1): self.assertEqual(export_group('__neutrino_truth__', sign*code), label)
        with self.assertRaisesRegex(ValueError, 'Unsupported'): export_group('__neutrino_truth__', 16)
        self.assertEqual(export_group('testbeam', 0), 'testbeam')
    def test_converter_preserves_selection(self):
        job = self.plan()['jobs'][0]
        args = converter_arguments(job, self.root)
        self.assertEqual(args[args.index('--selection-mode')+1], 'external_cutflow')
        self.assertEqual(args[args.index('--feature-dev-group')+1], '__neutrino_truth__')
        self.assertEqual(args[args.index('--source-uid')+1], job['record']['source_uid'])
        self.assertNotIn('--source-energy-gev', args)
    def test_muon_adapter(self):
        for row in self.rows:
            row.update(sample_type='MC_muon', source_domain_id='2', family='mc_muon_up', selection_mode='direct_input')
        self.save('mc-muon', ['mc_muon_up'])
        args = converter_arguments(self.plan('mc-muon')['jobs'][0], self.root)
        self.assertTrue(args[0].endswith('digi_2_features_mc_muon.py'))
    def test_testbeam_strata(self):
        for i, row in enumerate(self.rows):
            row.update(sample_type='real_testbeam', family='testbeam_data_2024', source_domain_id='5',
                       source_pdg_code='211', source_energy_gev=str(100 if i%2 else 200),
                       source_geometry_id='test-geometry', source_geometry_checksum='catalogue-checksum')
        self.save('testbeam', [])
        p = self.plan('testbeam')
        self.assertEqual(len(p['jobs']), 2)
        args = converter_arguments(p['jobs'][0], self.root)
        self.assertIn('--source-energy-gev', args)
    def test_real_data_uses_frozen_groups(self):
        for i, row in enumerate(self.rows):
            row.update(sample_type='real_data', family='real_data', source_domain_id='3',
                       selection_mode='indexed_cutflow',
                       selection_profile='real_data_2024_no_veto' if i%2 else 'real_data_2024_veto_10pct')
        self.save('real-data', ['real_data_2024_no_veto', 'real_data_2024_veto_10pct'])
        p = self.plan('real-data')
        self.assertEqual(len({j['group'] for j in p['jobs']}), 2)
        self.rows[0]['selection_mode'] = 'direct_input'; self.save('real-data')
        with self.assertRaisesRegex(ValueError, 'frozen'): self.plan('real-data')
    def make_export(self, directory, uid='test-source', mismatch=None):
        import numpy as np
        import uproot
        from engine import load_events
        from algorithms import baseline
        directory.mkdir(parents=True, exist_ok=True)
        export(directory/'hits.jsonl', uid=uid, count=2)
        rows, _ = load_events(directory/'hits.jsonl')
        values = [baseline.calculate_features(e['hits'], baseline.cluster_hits(e['hits'], {}), {}) for e in rows]
        arrays = {key: np.array([e['event_id'][key] for e in rows]) for key in rows[0]['event_id']}
        arrays.update({key: np.array([v[key] for v in values]) for key in values[0]})
        if mismatch: arrays[mismatch][0] += 1
        with uproot.recreate(directory/'features.root') as out: out['sndData'] = arrays
        return directory
    def test_parity_detects_identity_and_feature_mismatch(self):
        for i, mismatch in enumerate((None, 'eventIndex', 'scifi_cluster_total_count')):
            d = self.make_export(self.root/f'export{i}', mismatch=mismatch)
            if mismatch:
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    validate_export(d/'hits.jsonl', d/'features.root')
            else:
                self.assertEqual(validate_export(d/'hits.jsonl', d/'features.root')['events'], 2)
    def test_snakemake_precreated_parent_and_reuse(self):
        p = self.plan()
        destination = Path(p['directory'])/'exports/0'
        destination.mkdir(parents=True)
        def converter(command, **kwargs):
            output = Path(command[command.index('--feature-dev-export')+1]).parent
            self.make_export(output)
        with patch('prepare.subprocess.run', side_effect=converter) as call:
            export_source(p, 0)
            export_source(p, 0)
            self.assertEqual(call.call_count, 1)
        self.assertTrue((destination/'complete.json').exists())
    def test_failed_export_is_retryable(self):
        p = self.plan()
        import subprocess
        with patch('prepare.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'converter')):
            with self.assertRaisesRegex(ValueError, 'Converter failed'): export_source(p, 0)
        base = Path(p['directory'])
        self.assertFalse((base/'exports/0').exists())
        self.assertTrue((base/'logs/export-0.log').exists())

    def prepared_cache(self, required=('CC_nue',)):
        p = self.plan(); p['required_groups'] = list(required)
        base = Path(p['directory']); base.mkdir(parents=True)
        write_json(base/'plan.json', p)
        for i in range(len(p['jobs'])):
            d = self.make_export(base/'exports'/str(i), uid=f'source-{i}')
            write_json(d/'validation.json', validate_export(d/'hits.jsonl', d/'features.root'))
            write_json(d/'complete.json', {f.name: sha(f) for f in d.iterdir()})
        return p
    def test_assembly_cache_reuse_and_tamper(self):
        p = self.prepared_cache(); assemble(p)
        path, info = resolve_prepared(p)
        self.assertEqual(info['groups'][0]['events'], 3)
        self.assertEqual(resolve_prepared(p)[0], path)
        path.write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'cache changed'): resolve_prepared(p)
    def test_missing_group_never_publishes_sample(self):
        p = self.prepared_cache(('CC_nue', 'NC_nue'))
        with self.assertRaisesRegex(ValueError, 'Missing configured groups'): assemble(p)
        self.assertFalse((Path(p['directory'])/'complete.json').exists())
        self.assertFalse((Path(p['directory'])/'small.jsonl').exists())


if __name__ == '__main__': unittest.main()
