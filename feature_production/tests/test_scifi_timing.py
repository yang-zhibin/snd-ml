"""Timing policy integration tests; artificial hits, no detector input scans."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import ROOT
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'convertData'))
import scifi_timing as timing
from hit_time_selection import select_calibrated_scifi_hits, apply_first_arrival_time0
import test_campaign as fixtures


class TimingTests(unittest.TestCase):
    def test_arguments_and_sample_routing(self):
        parser=argparse.ArgumentParser();timing.add_arguments(parser)
        args=parser.parse_args(['--scifi-data-window-cycles','-0.5','2.0'])
        cfg=timing.settings(args)
        self.assertEqual(timing.effective_policy('real_testbeam',cfg)['upper_cycles'],2.)
        self.assertEqual(timing.effective_policy('MC_testbeam',cfg)['upper_cycles'],1.2)
        self.assertEqual(timing.effective_policy('MC_muon',cfg)['clock_period_ns'],6.25)
        for bad in ([2,1],[1,1],[float('nan'),2],[-1,float('inf')]):
            with self.assertRaises(ValueError):timing.window(bad)

    def test_boundaries_planes_and_ties(self):
        # Peak centre 10.25 ns: inclusive boundaries 7.125 and 17.75.
        times=[10.1]*5+[7.125,17.75,7.124,17.751,22.75]
        hits=[dict(station=1,isVertical=False,hitTimeNS=t) for t in times]
        other=[dict(station=2,isVertical=True,hitTimeNS=t) for t in [40.1]*5+[47.75]]
        kept=select_calibrated_scifi_hits(hits+other)
        self.assertEqual(len(kept),13)
        wider=select_calibrated_scifi_hits(hits+other,upper_clock_cycles=2.)
        self.assertIn(hits[-1],wider)
        tied=[dict(station=1,isVertical=False,hitTimeNS=t) for t in [10.1,40.1]]
        self.assertEqual(select_calibrated_scifi_hits(tied),tied[:1])
        args=SimpleNamespace(type='MC_muon',scifi_mc_window_cycles=[-.5,2.])
        self.assertEqual(timing.calibrated_kwargs(args),dict(lower_clock_cycles=.5,upper_clock_cycles=2.))

    def test_first_arrival_unchanged(self):
        a=[dict(hitTimeNS=105.)];b=[dict(hitTimeNS=100.)]
        self.assertEqual(apply_first_arrival_time0('MC_muon',[a,b],time_key='hitTimeNS'),(100.,True))
        self.assertEqual(a[0]['hitTimeNS'],5.)

    def test_root_policy_roundtrip_and_mismatch(self):
        cfg={'data':[-.5,2.],'mc':[-.5,1.2]}
        policy=timing.effective_policy('real_testbeam',cfg)
        policy.update(additional_max_time_ns=None,selector_source='fixture.py',selector_sha256='a'*64)
        args=SimpleNamespace(_scifi_timing=dict(requested=cfg,effective=policy))
        with tempfile.TemporaryDirectory() as tmp:
            f=ROOT.TFile(str(Path(tmp)/'metadata.root'),'RECREATE')
            timing.write_record(ROOT,args);f.Close()
            f=ROOT.TFile.Open(str(Path(tmp)/'metadata.root'))
            self.assertEqual(timing.validate_record(f,'real_testbeam',cfg,True),args._scifi_timing)
            with self.assertRaisesRegex(ValueError,'upper_cycles'):
                timing.validate_record(f,'real_testbeam',{'data':[-.5,1.2],'mc':[-.5,1.2]},True)
            f.Close()


class TimingPlanTests(unittest.TestCase):
    setUp=fixtures.CampaignTests.setUp
    tearDown=fixtures.CampaignTests.tearDown
    args=fixtures.CampaignTests.args
    catalogue=fixtures.CampaignTests.catalogue

    def test_changed_effective_window_blocks_reuse(self):
        args=self.args();before=fixtures.production_cli.make_plan(args)
        prev=self.root/'previous';prev.mkdir();fixtures.write_json(prev/'plan.json',before)
        args.scifi_mc_window_cycles=[-.5,2.]
        after=fixtures.production_cli.make_plan(args)
        self.assertNotEqual(before['conversion_signature'],after['conversion_signature'])
        args.reuse_sources=prev
        with self.assertRaisesRegex(ValueError,'campaign_sha256'):
            fixtures.production_cli.make_plan(args)

    def test_unused_data_setting_allows_mc_reuse_and_forwarding(self):
        from sources import converter_arguments
        from worker import validator_arguments
        args=self.args();before=fixtures.production_cli.make_plan(args)
        prev=self.root/'previous';prev.mkdir();fixtures.write_json(prev/'plan.json',before)
        args.scifi_data_window_cycles=[-.5,2.];args.reuse_sources=prev
        plan=fixtures.production_cli.make_plan(args)
        job=dict(plan['jobs'][0],group='test')
        cmd=converter_arguments(job,self.root)
        pos=cmd.index('--scifi-data-window-cycles');self.assertEqual(cmd[pos+1:pos+3],['-0.5','2.0'])
        cmd=validator_arguments(plan,job,'a','b','c')
        self.assertIn('--require-scifi-timing',cmd)
        self.assertEqual(before['conversion_signature'],plan['conversion_signature'])
