"""Run with feature_dev/run_export.sh; tiny artificial hits through real SNDSW."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import ROOT
from analysis.analyses.snd_analysis_2024_0mu import sciFiTools as snd
import scifi_timing


class Hit:
    def __init__(self, time, vertical=False, valid=True):
        self.time, self.vertical, self.valid = time, vertical, valid
    def GetTime(self): return self.time
    def isVertical(self): return self.vertical
    def GetStation(self): return 1
    def GetDetectorID(self): return 1000000
    def isValid(self): return self.valid


class SndTimingTests(unittest.TestCase):
    def test_default_parity_wider_window_and_data_correction(self):
        clock=1e9/160.316e6
        times=[20.1]*5+[20.25-.5*clock,20.25+1.2*clock,20.25+1.2*clock+.001,31.]
        original=(snd.min_clock_cycle,snd.max_clock_cycle)
        # Fake only geometry correction; histogram/selection is the installed SNDSW code.
        detector=SimpleNamespace(GetCorrectedTime=lambda det,time,offset:time-1.)
        globals_=SimpleNamespace(FindObject=lambda name:detector)
        root=SimpleNamespace(TH1D=ROOT.TH1D,gROOT=SimpleNamespace(GetListOfGlobals=lambda:globals_))
        try:
            for mc in (True,False):
                hits=[Hit(t if mc else (t+1)/clock) for t in times]+[Hit(20.,valid=False)]
                event=SimpleNamespace(Digi_ScifiHits=hits)
                args=SimpleNamespace(type='MC_testbeam' if mc else 'real_testbeam')
                with patch.object(snd,'ROOT',root):
                    snd.min_clock_cycle,snd.max_clock_cycle=-.5,1.2
                    baseline=snd.selectHits(event,MC=mc)
                    scifi_timing.configure(args,lambda _:mc)
                    self.assertEqual(snd.selectHits(event,MC=mc),baseline)
                    self.assertNotIn(hits[-1],baseline)
                    self.assertNotIn(hits[-2],baseline)
                    if mc:self.assertEqual(len(baseline),7)
                    setattr(args,'scifi_mc_window_cycles' if mc else 'scifi_data_window_cycles',[-.5,2.])
                    policy=scifi_timing.configure(args,lambda _:mc)
                    wide=snd.selectHits(event,MC=mc)
                    self.assertIn(hits[-2],wide)
                    self.assertEqual(len(wide),9)
                    self.assertEqual(policy['upper_cycles'],2.)
        finally:
            snd.min_clock_cycle,snd.max_clock_cycle=original


if __name__=='__main__':unittest.main()
