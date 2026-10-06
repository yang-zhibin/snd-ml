#!/usr/bin/env python3

from __future__ import annotations

from argparse import Namespace
from array import array
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import ROOT


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_feature_distributions import make_histogram, make_plot  # noqa: E402


class UnifiedFeatureDistributionRootTest(unittest.TestCase):
    def test_raw_flow_and_unit_area_plot(self):
        config = REPO_ROOT / "snakemake/metadata/configs/unified_shower_distributions_v1.yaml"
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            directory = Path(directory)
            feature = directory / "feature.root"
            marker = directory / "feature.root.validated.json"
            manifest = directory / "manifest.csv"
            histogram = directory / "hist.root"
            plot = directory / "plot.pdf"
            summary = directory / "summary.csv"

            root_file = ROOT.TFile(str(feature), "RECREATE")
            tree = ROOT.TTree("sndData", "")
            values = {
                "count_scifi": array("i", [0]),
                "pdgCode": array("i", [0]),
                "density_sndsw_scifi": array("d", [0.0]),
                "scifi_cluster_qdc_per_hit": array("d", [0.0]),
                "scifi_qdc_per_hit_direct": array("d", [0.0]),
            }
            tree.Branch("count_scifi", values["count_scifi"], "count_scifi/I")
            tree.Branch("pdgCode", values["pdgCode"], "pdgCode/I")
            for name in (
                "density_sndsw_scifi",
                "scifi_cluster_qdc_per_hit",
                "scifi_qdc_per_hit_direct",
            ):
                tree.Branch(name, values[name], f"{name}/D")
            for pdg, density in ((12, 750.0), (14, 1250.0), (12, 45000.0)):
                values["count_scifi"][0] = 50
                values["pdgCode"][0] = pdg
                values["density_sndsw_scifi"][0] = density
                values["scifi_cluster_qdc_per_hit"][0] = 3.0
                values["scifi_qdc_per_hit_direct"][0] = 3.0
                tree.Fill()
            tree.Write()
            root_file.Close()

            source_uid = "unified_src_0123456789abcdef0123"
            marker.write_text(
                json.dumps(
                    {
                        "validation_passed": True,
                        "source_uid": source_uid,
                        "feature_path": str(feature),
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [{
                    "family": "mc_neutrino",
                    "subfolder": "2024_ve",
                    "source_uid": source_uid,
                    "feature_scifi_cluster_v1_path": str(feature),
                    "validation_record": str(marker),
                    "lumi_per_file": 1.0,
                    "exposure_kind": "simulated_luminosity_fb",
                }]
            ).to_csv(manifest, index=False)

            make_histogram(
                Namespace(
                    config=str(config), manifest=str(manifest), group="CC_nue",
                    feature="density_sndsw_scifi", output=str(histogram)
                )
            )
            source = ROOT.TFile.Open(str(histogram))
            raw = source.Get("hist")
            self.assertEqual(raw.Integral(0, raw.GetNbinsX() + 1), 2.0)
            self.assertEqual(raw.GetBinContent(raw.GetNbinsX() + 1), 1.0)
            source.Close()

            make_plot(
                Namespace(
                    config=str(config), feature="density_sndsw_scifi",
                    normalization="unit_area", hist=[f"CC_nue={histogram}"],
                    output=str(plot), summary=str(summary)
                )
            )
            self.assertTrue(plot.is_file())
            plotted = pd.read_csv(summary)
            self.assertEqual(plotted.loc[0, "normalization"], "unit_area")
            self.assertTrue(
                math.isclose(
                    plotted.loc[0, "scaled_in_range_integral"] * 500.0,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )


if __name__ == "__main__":
    unittest.main()
