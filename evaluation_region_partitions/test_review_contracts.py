"""Adversarial regressions for review findings CR-001 through CR-005."""
import copy
import json
import tempfile
import unittest
from array import array
from pathlib import Path

import pandas as pd
import yaml
from feature_integrity import sha256, verify_feature
from snapshot_contract import validate_membership, validate_selection_coverage
from build_unified_partial_feature_snapshots import build_snapshots, _check_row
from unified_feature_distributions import load_validated_rows
from sample_selection import sample_kind, MC_SAMPLE_TYPES, REAL_SAMPLE_TYPES
from build_region_partitions import check_entry_identity
from prepare_unified_real_data_feature_manifests import build_manifests
from test_prepare_unified_real_data_feature_manifests import _write_source, _write_config


class ReviewContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def feature(self):
        feature = self.root / 'feature.root'; feature.write_bytes(b'original')
        marker = self.root / 'receipt.json'
        receipt = dict(validation_passed=True, source_uid='a', feature_path=str(feature),
                       feature_sha256=sha256(feature), feature_size_bytes=feature.stat().st_size)
        marker.write_text(json.dumps(receipt))
        row = dict(family='family', source_uid='a', feature_scifi_cluster_v1_path=str(feature),
                   validation_record=str(marker), lumi_per_file=1, exposure_kind='event_count_only',
                   selected_event_count=2)
        return feature, marker, receipt, row

    def snapshot(self):
        feature, marker, receipt, row = self.feature()
        source = self.root / 'source.csv'; pd.DataFrame([row]).to_csv(source,index=False)
        config = self.root / 'config.yaml'
        config.write_text(yaml.safe_dump(dict(version='test',source_manifests={'family':str(source)})))
        record = build_snapshots(config,self.root / 'snapshot',workers=1)[0]
        return feature, marker, receipt, row, record

    def test_normalized_supported_types(self):
        for group, kind in ((MC_SAMPLE_TYPES,'mc'),(REAL_SAMPLE_TYPES,'real')):
            for value in group:
                for spelling in (value, value.upper(), ' '+value+' '):
                    self.assertEqual(sample_kind(spelling),kind)
        for value in ('', 'unknown', 'notMC', 'mc_typo'):
            with self.assertRaises(ValueError): sample_kind(value)

    def test_same_size_and_changed_size_corruption_never_falls_back(self):
        feature,marker,receipt,row = self.feature()
        verify_feature(row)
        for data in (b'replaced',b'longer replacement'):
            feature.write_bytes(data)
            self.assertEqual(_check_row(row,str(self.root/'shards'))[0],'invalid_feature_integrity')
            manifest=self.root/'manifest.csv';pd.DataFrame([row]).to_csv(manifest,index=False)
            with self.assertRaisesRegex(RuntimeError,'checksum'):
                load_validated_rows(manifest,'family',allow_missing_validated_sources=True)

    def test_missing_digest_and_invalid_size(self):
        _,marker,receipt,row=self.feature()
        for key,value in (('feature_sha256',''),('feature_sha256','x'*64),('feature_size_bytes',-1),('feature_size_bytes',True)):
            altered={**receipt,key:value};marker.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):verify_feature(row)

    def test_receipt_replacement_cannot_rebind_frozen_bytes(self):
        feature,marker,receipt,row,record=self.snapshot()
        feature.write_bytes(b'replaced')
        receipt['feature_sha256']=sha256(feature);marker.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(RuntimeError,'frozen'):
            load_validated_rows(record['snapshot_manifest'],'family')

    def test_membership_hash_and_provenance(self):
        _,_,_,_,record=self.snapshot();validate_membership(record)
        snapshot=Path(record['snapshot_manifest']);original=snapshot.read_bytes()
        for column,value in (('source_uid','unknown'),('feature_scifi_cluster_v1_path','wrong.root'),('family','wrong')):
            frame=pd.read_csv(snapshot,keep_default_na=False);frame[column]=value;frame.to_csv(snapshot,index=False)
            with self.assertRaisesRegex(ValueError,'checksum'):validate_membership(record)
            refreshed={**record,'snapshot_manifest_sha256':sha256(snapshot)}
            with self.assertRaises(ValueError):validate_membership(refreshed)
            snapshot.write_bytes(original)
        bad=copy.deepcopy(record);bad['source_statuses']['a']='missing_feature'
        with self.assertRaises(ValueError):validate_membership(bad)

    def test_duplicate_unit_and_mixed_layout(self):
        _,_,_,_,record=self.snapshot();snapshot=Path(record['snapshot_manifest'])
        frame=pd.read_csv(snapshot,keep_default_na=False)
        pd.concat([frame,frame]).to_csv(snapshot,index=False)
        record['snapshot_manifest_sha256']=sha256(snapshot);record['available_feature_file_count']=2
        with self.assertRaisesRegex(ValueError,'duplicate'):validate_membership(record)

    def test_canonical_range_and_layout_status(self):
        _, _, _, _, record = self.snapshot()
        snapshot = Path(record['snapshot_manifest'])
        original = pd.read_csv(snapshot, keep_default_na=False)
        for key, value in (('selected_entry_start', 1), ('selected_entry_stop', 1), ('shard_index', 0)):
            frame = original.copy()
            frame[key] = value
            frame.to_csv(snapshot, index=False)
            record['snapshot_manifest_sha256'] = sha256(snapshot)
            with self.assertRaisesRegex(ValueError, 'canonical'):
                validate_membership(record)
        original.to_csv(snapshot, index=False)
        record['snapshot_manifest_sha256'] = sha256(snapshot)
        record['source_statuses']['a'] = 'available_shards'
        record['status_counts'] = {'available_shards': 1}
        with self.assertRaisesRegex(ValueError, 'availability status'):
            validate_membership(record)

    def test_selection_missing_receipt_and_profile_zero(self):
        a=_write_source(self.root,'a',2,3);b=_write_source(self.root,'b',0,4)
        config=_write_config(self.root,[a,b]);frames,coverage=build_manifests(config,workers=1)
        ledger=self.root/'coverage.json';ledger.write_text(json.dumps(coverage))
        record=dict(family='real_data_2024_no_veto',selection_coverage=str(ledger),selection_coverage_sha256=sha256(ledger))
        universe={row['source_uid']:row for row in frames[record['family']].to_dict('records')}
        checked=validate_selection_coverage(record,universe)
        self.assertEqual(checked['zero_selected_source_count'],1)
        Path(b['validation_record']).unlink()
        frames,coverage=build_manifests(config,workers=1);ledger.write_text(json.dumps(coverage));record['selection_coverage_sha256']=sha256(ledger)
        with self.assertRaisesRegex(ValueError,'incomplete upstream'):
            validate_selection_coverage(record,universe)

    def test_explicit_exclusion_and_changed_policy(self):
        a=_write_source(self.root,'a',2,3);b=_write_source(self.root,'b',4,4)
        config=_write_config(self.root,[a,b]);policy=self.root/'policy.yaml'
        policy.write_text(yaml.safe_dump({'excluded_source_uids':['b'],'exclusion_reason':'synthetic unavailable input'}))
        Path(b['validation_record']).unlink()
        frames,coverage=build_manifests(config,workers=1)
        ledger=self.root/'coverage.json';ledger.write_text(json.dumps(coverage))
        record=dict(family='real_data_2024_no_veto',selection_coverage=str(ledger),selection_coverage_sha256=sha256(ledger))
        universe={row['source_uid']:row for row in frames[record['family']].to_dict('records')}
        self.assertEqual(validate_selection_coverage(record,universe)['excluded_source_count'],1)
        policy.write_text('excluded_source_uids: []\n')
        with self.assertRaisesRegex(ValueError,'changed selection_policy'):
            validate_selection_coverage(record,universe)

    def root_tree(self,path,identities,missing=None):
        import ROOT
        handle=ROOT.TFile(str(path),'RECREATE');tree=ROOT.TTree('events','events')
        values={key:array('q',[0]) for key in ('runId','eventId','eventIndex') if key!=missing}
        for key,value in values.items():tree.Branch(key,value,key+'/L')
        for identity in identities:
            for key,value in zip(('runId','eventId','eventIndex'),identity):
                if key in values:values[key][0]=value
            tree.Fill()
        tree.Write();handle.Close()

    def test_entrywise_identity_not_just_count(self):
        left=self.root/'left.root';right=self.root/'right.root';keys=[(1,2,0),(1,3,1)]
        self.root_tree(left,keys);self.root_tree(right,keys)
        check_entry_identity(left,'events',right,'events')
        for values in (keys[::-1],[(1,2,0),(1,4,1)],keys[:1]):
            self.root_tree(right,values)
            with self.assertRaises(ValueError):check_entry_identity(left,'events',right,'events')
        self.root_tree(right,keys,missing='eventId')
        with self.assertRaisesRegex(ValueError,'Missing identity'):check_entry_identity(left,'events',right,'events')
        self.root_tree(left,[]);self.root_tree(right,[])
        check_entry_identity(left,'events',right,'events')


    def test_full_partition_combined_and_split_preserve_order(self):
        import ROOT
        from unittest.mock import patch
        import build_region_partitions as builder
        left=self.root/'input_feature.root'; right=self.root/'input_hit.root'
        keys=[(1,2,0),(1,3,1),(1,4,2)]
        self.root_tree(left,keys);self.root_tree(right,keys)
        rows=[dict(feature_path=str(left),hit3d_path=str(right),lumi_per_file=1,
                   metadata_path='synthetic.csv',row_number=0)]
        rule=dict(particle_family='test',particle_id=1,event_filter='true')
        config=dict(partition_version='test',input=dict(feature_tree='events',hit3d_tree='events'),
                    output=dict(feature_prefix='feature',hit3d_prefix='hit',write_cutflow=False),
                    grouping=dict(output_partition_id_template='{region}_{particle_group}_{part_index}'),
                    particle_group_rules={'test':rule},
                    metadata_output_columns=['feature_partition_path','n_events','selection_expression',
                                             'particle_filter_expression','source_feature_paths'])
        regions=dict(region_version='test',regions={'selected':{'expression':'eventIndex != 1'}})
        feature=self.root/'selected_feature.root';hit=self.root/'selected_hit.root'
        metadata=self.root/'feature.csv'
        args=dict(metadata_paths=['synthetic.csv'],region_config=regions,partition_config=config,
                  output_dir=str(self.root),output_metadata=str(metadata),only_regions={'selected'},
                  only_particles={'test'},part_index=1,feature_output=str(feature),hit3d_output=str(hit),
                  rdf_threads=2)
        with patch.object(builder,'select_lumi_metadata_rows',return_value=([],{'fraction':1})), \
             patch.object(builder,'metadata_paths_to_rows',return_value=rows), \
             patch.object(builder,'infer_particle_rule_for_rows',return_value=('test',rule)):
            ROOT.EnableImplicitMT(2)
            builder.run_rdf_partition(**args,product_mode='both')
            self.assertFalse(ROOT.IsImplicitMTEnabled())
            check_entry_identity(feature,'events',hit,'events')
            output=ROOT.TFile.Open(str(feature));tree=output.Get('events')
            self.assertEqual([(int(e.runId),int(e.eventId),int(e.eventIndex)) for e in tree],[keys[0],keys[2]])
            output.Close()
            builder.run_rdf_partition(**args,product_mode='feature')
            split={**args,'output_metadata':str(self.root/'hit.csv'),
                   'metadata_feature_output':str(feature),'feature_reference_metadata':str(metadata)}
            builder.run_rdf_partition(**split,product_mode='hit3d')
            check_entry_identity(feature,'events',hit,'events')
            with patch.object(builder, 'metadata_paths_to_rows', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'empty Hit3D source list'):
                    builder.run_rdf_partition(**split, product_mode='hit3d')
            self.root_tree(feature,[keys[2],keys[0]])
            with self.assertRaisesRegex(ValueError,'checksum'):
                builder.run_rdf_partition(**split,product_mode='hit3d')


    def test_complete_shards_and_corrupt_or_overlapping_shard(self):
        canonical,marker,_,row=self.feature();canonical.unlink();marker.unlink()
        shard_root=self.root/'shards';directory=shard_root/'a';directory.mkdir(parents=True)
        shards=[]
        for index in range(2):
            feature=directory/f'{index}.root';feature.write_bytes(b'unit')
            marker=directory/f'{index}.json'
            receipt=dict(validation_passed=True,source_uid='a',feature_path=str(feature),
                         feature_sha256=sha256(feature),feature_size_bytes=4,
                         selected_entry_start=index,selected_entry_stop=index+1)
            marker.write_text(json.dumps(receipt))
            shards.append(dict(shard_index=index,selected_entry_start=index,selected_entry_stop=index+1,
                               written_event_count=1,feature_path=str(feature),validation_record=str(marker),
                               feature_sha256=sha256(feature)))
        complete=dict(validation_passed=True,source_uid='a',selected_event_count=2,shards=shards)
        (directory/'shards.complete.json').write_text(json.dumps(complete))
        self.assertEqual(_check_row(row,str(shard_root))[0],'available_shards')
        source=self.root/'source.csv';pd.DataFrame([row]).to_csv(source,index=False)
        config=self.root/'config.yaml'
        config.write_text(yaml.safe_dump(dict(version='test',source_manifests={'family':str(source)},
                                              shard_receipt_roots={'family':str(shard_root)})))
        record=build_snapshots(config,self.root/'snapshot',workers=1)[0]
        validate_membership(record)
        shards[1]['selected_entry_start']=0
        (directory/'shards.complete.json').write_text(json.dumps(complete))
        self.assertEqual(_check_row(row,str(shard_root))[0],'invalid_shard_order')
        shards[1]['selected_entry_start']=1
        (directory/'shards.complete.json').write_text(json.dumps(complete))
        Path(shards[1]['feature_path']).write_bytes(b'evil')
        with self.assertRaisesRegex(RuntimeError,'integrity'):
            build_snapshots(config,self.root/'snapshot2',workers=1)

    def test_converter_timing_policy_uses_normalized_dispatch(self):
        import ast
        from types import SimpleNamespace
        source=Path(__file__).resolve().parents[1]/'convertData/digi_2_features.py'
        tree=ast.parse(source.read_text())
        names={'is_mc_sample','applies_mc_25ns_hit_filter'}
        module=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
        scope={'sample_kind':sample_kind};exec(compile(module,str(source),'exec'),scope)
        for spelling in ('MC_muon','mc_muon',' MC_MUON '):
            args=SimpleNamespace(type=spelling,out_path='',digi_path='',preSelect_path='')
            self.assertTrue(scope['is_mc_sample'](args))
            self.assertTrue(scope['applies_mc_25ns_hit_filter'](args))
            args.digi_path='/legacy/mc_neutrino/input.root'
            self.assertFalse(scope['applies_mc_25ns_hit_filter'](args))


if __name__=='__main__':unittest.main()
