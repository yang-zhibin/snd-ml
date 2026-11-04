#!/usr/bin/env python3

from __future__ import annotations

from argparse import Namespace
from array import array
import json
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import ROOT
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_feature_distributions import (  # noqa: E402
    make_energy_summary_plot,
    make_faceted_plot,
    make_histogram,
    make_plot,
)


class UnifiedFeatureDistributionRootTest(unittest.TestCase):
    def test_cross_geometry_plot_requires_explicit_unit_area_policy(self):
        config = (
            REPO_ROOT
            / "snakemake/metadata/configs/unified_testbeam_distributions_v3.yaml"
        )
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            directory = Path(directory)

            def write_histogram(path, group, geometry_id, checksum, charge, energy):
                root_file = ROOT.TFile(str(path), "RECREATE")
                histogram = ROOT.TH1D("hist", "", 160, 0.0, 40000.0)
                histogram.Fill(1000.0)
                histogram.Fill(2000.0)
                histogram.Write()
                for name, value in {
                    "feature": "density_sndsw_scifi",
                    "particle_group": group,
                    "selection_base_expression": "count_scifi > 35",
                    "selection_expression": "count_scifi > 35",
                    "sampling_policy": "deterministic_chain_prefix_v1",
                    "source_geometry_id": geometry_id,
                    "source_geometry_checksum": checksum,
                    "particle_charge": charge,
                    "beam_energy_gev": str(energy),
                    "exposure_kind": "event_count_only",
                }.items():
                    ROOT.TNamed(name, value).Write()
                for name, value in {
                    "selected_entries": 2,
                    "available_selected_entries": 2,
                    "source_count": 1,
                    "event_cap": 50000,
                }.items():
                    ROOT.TParameter("Long64_t")(name, value).Write()
                for name, value in {
                    "underflow_raw": 0.0,
                    "in_range_raw": 2.0,
                    "overflow_raw": 0.0,
                    "integral_all_raw": 2.0,
                    "sample_exposure": 0.0,
                }.items():
                    ROOT.TParameter("double")(name, value).Write()
                root_file.Close()

            pion = directory / "pion.root"
            electron = directory / "electron.root"
            write_histogram(
                pion,
                "data_pion_scan_100",
                "Fe_3walls_H8_2023",
                "fe-checksum",
                "pi+",
                100,
            )
            write_histogram(
                electron,
                "data_electron_scan_100",
                "W_2walls_28plates",
                "w-checksum",
                "e-",
                100,
            )
            arguments = dict(
                config=str(config),
                feature="density_sndsw_scifi",
                normalization="unit_area",
                hist=[
                    f"data_pion_scan_100={pion}",
                    f"data_electron_scan_100={electron}",
                ],
                output=str(directory / "cross.pdf"),
                summary=str(directory / "cross.csv"),
            )
            with self.assertRaisesRegex(RuntimeError, "mixed geometry"):
                make_plot(Namespace(**arguments, geometry_policy="same_geometry"))
            make_plot(
                Namespace(
                    **arguments, geometry_policy="cross_geometry_diagnostic"
                )
            )
            summary = pd.read_csv(directory / "cross.csv")
            self.assertEqual(set(summary["draw_option"]), {"E1 P"})
            self.assertEqual(
                set(summary["geometry_policy"]), {"cross_geometry_diagnostic"}
            )
            self.assertEqual(
                set(summary["source_geometry_id"]),
                {"Fe_3walls_H8_2023", "W_2walls_28plates"},
            )
            self.assertTrue(
                all(
                    math.isclose(value * 250.0, 1.0, abs_tol=1e-12)
                    for value in summary["scaled_in_range_integral"]
                )
            )
            styled_arguments = dict(
                arguments,
                output=str(directory / "styled.pdf"),
                summary=str(directory / "styled.csv"),
            )
            make_plot(Namespace(
                **styled_arguments,
                geometry_policy="cross_geometry_diagnostic",
                plot_set="data_combined_particle_energy",
            ))
            styled = pd.read_csv(directory / "styled.csv")
            self.assertEqual(styled["line_color"].nunique(), 1)
            self.assertEqual(
                dict(zip(styled["particle_charge"], styled["line_style"])),
                {"pi+": 1, "e-": 2},
            )
            self.assertEqual(
                set(styled["style_policy"]), {"paired_particle_energy"}
            )
            self.assertEqual(
                dict(zip(styled["particle_charge"], styled["marker_style"])),
                {"pi+": 20, "e-": 24},
            )
            self.assertEqual(set(styled["draw_option"]), {"E1 P"})
            self.assertEqual(set(styled["unit_area_mode"]), {"bin_fraction"})
            self.assertTrue(all(
                math.isclose(value, 1.0, abs_tol=1e-12)
                for value in styled["scaled_in_range_integral"]
            ))
            self.assertEqual(
                set(styled["legend_label"]),
                {"Data pi+ 100 GeV - Fe 3-wall, H8 2023", "Data e- 100 GeV - W 2-wall, H4 2024"},
            )
            derived_arguments = Namespace(
                **arguments,
                geometry_policy="cross_geometry_diagnostic",
                plot_set="data_combined_particle_energy",
            )
            derived_arguments.output = str(directory / "faceted.pdf")
            derived_arguments.summary = str(directory / "faceted.csv")
            make_faceted_plot(derived_arguments)
            faceted = pd.read_csv(derived_arguments.summary)
            self.assertTrue(Path(derived_arguments.output).is_file())
            self.assertEqual(set(faceted["draw_option"]), {"E1 P"})
            self.assertEqual(set(faceted["panel"]), {1})
            derived_arguments.output = str(directory / "energy_summary.pdf")
            derived_arguments.summary = str(directory / "energy_summary.csv")
            make_energy_summary_plot(derived_arguments)
            energy_summary = pd.read_csv(derived_arguments.summary)
            self.assertTrue(Path(derived_arguments.output).is_file())
            self.assertEqual(len(energy_summary), 2)
            self.assertTrue((energy_summary["q16"] <= energy_summary["median"]).all())
            self.assertTrue((energy_summary["median"] <= energy_summary["q84"]).all())

    def test_raw_flow_and_unit_area_plot(self):
        config = REPO_ROOT / "snakemake/metadata/configs/unified_shower_distributions_v1.yaml"
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            directory = Path(directory)
            feature = directory / "feature.root"
            marker = directory / "feature.root.validated.json"
            manifest = directory / "manifest.csv"
            histogram = directory / "hist.root"
            histogram_nc = directory / "hist_nc.root"
            histogram_capped = directory / "hist_capped.root"
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
            for pdg, density in ((12, 750.0), (112, 1250.0), (12, 45000.0)):
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
                            "feature_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
                            "feature_size_bytes": feature.stat().st_size,
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

            capped_config = directory / "capped.yaml"
            with config.open("r", encoding="utf-8") as handle:
                capped_settings = yaml.safe_load(handle)
            capped_settings["groups"]["CC_nue"]["max_events"] = 1
            capped_settings["groups"]["CC_nue"]["sampling_policy"] = (
                "deterministic_chain_prefix_v1"
            )
            capped_config.write_text(
                yaml.safe_dump(capped_settings, sort_keys=False), encoding="utf-8"
            )
            make_histogram(
                Namespace(
                    config=str(capped_config), manifest=str(manifest), group="CC_nue",
                    feature="density_sndsw_scifi", output=str(histogram_capped)
                )
            )
            capped_source = ROOT.TFile.Open(str(histogram_capped))
            self.assertEqual(int(capped_source.Get("available_selected_entries").GetVal()), 2)
            self.assertEqual(int(capped_source.Get("selected_entries").GetVal()), 1)
            self.assertEqual(capped_source.Get("sampling_policy").GetTitle(),
                             "deterministic_chain_prefix_v1")
            capped_source.Close()

            make_histogram(
                Namespace(
                    config=str(config), manifest=str(manifest), group="NC_nue",
                    feature="density_sndsw_scifi", output=str(histogram_nc)
                )
            )

            make_plot(
                Namespace(
                    config=str(config), feature="density_sndsw_scifi",
                    normalization="unit_area",
                    hist=[f"CC_nue={histogram}", f"NC_nue={histogram_nc}"],
                    output=str(plot), summary=str(summary)
                )
            )
            self.assertTrue(plot.is_file())
            plotted = pd.read_csv(summary)
            self.assertEqual(len(plotted), 2)
            self.assertEqual(plotted.loc[0, "normalization"], "unit_area")
            self.assertTrue(
                math.isclose(
                    plotted.loc[0, "scaled_in_range_integral"] * 500.0,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )

    def test_shard_chain_matches_canonical_histogram_and_exposure(self):
        config = REPO_ROOT / "snakemake/metadata/configs/unified_shower_distributions_v1.yaml"
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            directory = Path(directory)
            uid = "unified_src_0123456789abcdef0123"

            def write_feature(path, densities):
                root_file = ROOT.TFile(str(path), "RECREATE")
                tree = ROOT.TTree("sndData", "")
                count_scifi = array("i", [50])
                pdg = array("i", [12])
                density = array("d", [0.0])
                tree.Branch("count_scifi", count_scifi, "count_scifi/I")
                tree.Branch("pdgCode", pdg, "pdgCode/I")
                tree.Branch("density_sndsw_scifi", density, "density_sndsw_scifi/D")
                for value in densities:
                    density[0] = value
                    tree.Fill()
                tree.Write()
                root_file.Close()

            def row_for(feature, marker, unit=""):
                marker.write_text(
                    json.dumps(
                        {
                            "validation_passed": True,
                            "source_uid": uid,
                            "feature_path": str(feature),
                            "feature_sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
                            "feature_size_bytes": feature.stat().st_size,
                        }
                    ),
                    encoding="utf-8",
                )
                row = {
                    "family": "mc_neutrino",
                    "subfolder": "2024_ve",
                    "source_uid": uid,
                    "feature_scifi_cluster_v1_path": str(feature),
                    "validation_record": str(marker),
                    "lumi_per_file": 7.0,
                    "exposure_kind": "simulated_luminosity_fb",
                }
                if unit:
                    row["feature_unit_uid"] = unit
                    row["feature_sha256"] = hashlib.sha256(feature.read_bytes()).hexdigest()
                    row["feature_size_bytes"] = feature.stat().st_size
                    row["validation_record_sha256"] = hashlib.sha256(marker.read_bytes()).hexdigest()
                return row

            canonical = directory / "canonical.root"
            write_feature(canonical, [750.0, 1250.0, 1750.0, 2250.0])
            canonical_rows = [
                row_for(canonical, directory / "canonical.validated.json")
            ]
            shard_rows = []
            for index, values in enumerate(([750.0, 1250.0], [1750.0, 2250.0])):
                feature = directory / f"part_{index}.root"
                write_feature(feature, values)
                shard_rows.append(
                    row_for(
                        feature,
                        directory / f"part_{index}.validated.json",
                        f"{uid}:part_{index:04d}",
                    )
                )
                shard_rows[-1]["selected_entry_start"] = index * 2

            canonical_manifest = directory / "canonical.csv"
            shard_manifest = directory / "shards.csv"
            pd.DataFrame(canonical_rows).to_csv(canonical_manifest, index=False)
            pd.DataFrame(shard_rows).to_csv(shard_manifest, index=False)
            canonical_histogram = directory / "canonical_hist.root"
            shard_histogram = directory / "shard_hist.root"
            for manifest, output in (
                (canonical_manifest, canonical_histogram),
                (shard_manifest, shard_histogram),
            ):
                make_histogram(
                    Namespace(
                        config=str(config),
                        manifest=str(manifest),
                        group="CC_nue",
                        feature="density_sndsw_scifi",
                        output=str(output),
                    )
                )

            canonical_file = ROOT.TFile.Open(str(canonical_histogram))
            shard_file = ROOT.TFile.Open(str(shard_histogram))
            canonical_hist = canonical_file.Get("hist")
            shard_hist = shard_file.Get("hist")
            for index in range(canonical_hist.GetNbinsX() + 2):
                self.assertEqual(
                    canonical_hist.GetBinContent(index), shard_hist.GetBinContent(index)
                )
            self.assertEqual(int(shard_file.Get("source_count").GetVal()), 1)
            self.assertEqual(float(shard_file.Get("sample_exposure").GetVal()), 7.0)
            canonical_file.Close()
            shard_file.Close()


if __name__ == "__main__":
    unittest.main()
