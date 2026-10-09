#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "snakemake/metadata"))

from build_unified_partial_feature_snapshots import (  # noqa: E402
    build_snapshots,
    write_coverage,
)


class UnifiedPartialFeatureSnapshotTests(unittest.TestCase):
    def test_snapshot_keeps_only_matching_successful_validation(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            root = Path(directory)
            rows = []
            for index, status in enumerate(("valid", "missing", "failed", "mismatch")):
                uid = f"unified_src_{index:020d}"
                feature = root / f"feature_{index}.root"
                marker = root / f"feature_{index}.validated.json"
                feature.touch()
                if status != "missing":
                    marker.write_text(
                        json.dumps(
                            {
                                "validation_passed": status != "failed",
                                "source_uid": "wrong" if status == "mismatch" else uid,
                                "feature_path": str(feature),
                            }
                        ),
                        encoding="utf-8",
                    )
                rows.append(
                    {
                        "family": "mc_neutrino",
                        "source_uid": uid,
                        "feature_scifi_cluster_v1_path": str(feature),
                        "validation_record": str(marker),
                        "lumi_per_file": 15.0,
                        "exposure_kind": "simulated_luminosity_fb",
                    }
                )
            manifest = root / "source.csv"
            pd.DataFrame(list(reversed(rows))).to_csv(manifest, index=False)
            config = root / "config.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "version": "test_snapshot",
                        "source_manifests": {"mc_neutrino": str(manifest)},
                    }
                ),
                encoding="utf-8",
            )

            output_root = root / "snapshot"
            coverage = build_snapshots(config, output_root, workers=2)
            frozen = pd.read_csv(output_root / "mc_neutrino.csv")
            self.assertEqual(frozen["source_uid"].tolist(), [rows[0]["source_uid"]])
            self.assertEqual(coverage[0]["available_source_count"], 1)
            self.assertEqual(coverage[0]["unavailable_source_count"], 3)
            self.assertEqual(
                coverage[0]["status_counts"],
                {
                    "available": 1,
                    "failed_validation": 1,
                    "missing_validation_record": 1,
                    "source_uid_mismatch": 1,
                },
            )
            self.assertEqual(coverage[0]["available_exposure"], 15.0)

            coverage_json = root / "coverage.json"
            coverage_csv = root / "coverage.csv"
            write_coverage(coverage, coverage_json, coverage_csv)
            self.assertEqual(
                json.loads(coverage_json.read_text(encoding="utf-8"))["families"][0][
                    "available_source_count"
                ],
                1,
            )
            self.assertEqual(len(pd.read_csv(coverage_csv)), 1)

    def test_snapshot_expands_complete_shards_without_multiplying_exposure(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            root = Path(directory)
            uid = "unified_src_0123456789abcdef0123"
            canonical = root / "canonical.root"
            canonical_marker = root / "canonical.root.validated.json"
            shard_root = root / "shards"
            source_root = shard_root / uid
            source_root.mkdir(parents=True)
            shards = []
            for index, (start, stop) in enumerate(((0, 2), (2, 5))):
                feature = source_root / f"part_{index}.root"
                marker = source_root / f"part_{index}.validated.json"
                feature.touch()
                marker.write_text(
                    json.dumps(
                        {
                            "validation_passed": True,
                            "source_uid": uid,
                            "feature_path": str(feature),
                        }
                    ),
                    encoding="utf-8",
                )
                shards.append(
                    {
                        "shard_index": index,
                        "selected_entry_start": start,
                        "selected_entry_stop": stop,
                        "written_event_count": stop - start,
                        "feature_path": str(feature),
                        "validation_record": str(marker),
                    }
                )
            (source_root / "shards.complete.json").write_text(
                json.dumps(
                    {
                        "validation_passed": True,
                        "source_uid": uid,
                        "selected_event_count": 5,
                        "shards": shards,
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "source.csv"
            pd.DataFrame(
                [
                    {
                        "family": "mc_muon_up",
                        "source_uid": uid,
                        "feature_scifi_cluster_v1_path": str(canonical),
                        "validation_record": str(canonical_marker),
                        "selected_event_count": 5,
                        "lumi_per_file": 15.0,
                        "exposure_kind": "simulated_luminosity_fb",
                    }
                ]
            ).to_csv(manifest, index=False)
            config = root / "config.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "version": "test_shards",
                        "source_manifests": {"mc_muon_up": str(manifest)},
                        "shard_receipt_roots": {"mc_muon_up": str(shard_root)},
                    }
                ),
                encoding="utf-8",
            )

            output_root = root / "snapshot"
            coverage = build_snapshots(config, output_root, workers=2)
            frozen = pd.read_csv(output_root / "mc_muon_up.csv")
            self.assertEqual(len(frozen), 2)
            self.assertEqual(frozen["source_uid"].nunique(), 1)
            self.assertEqual(frozen["feature_layout"].tolist(), ["shard", "shard"])
            self.assertEqual(coverage[0]["available_source_count"], 1)
            self.assertEqual(coverage[0]["available_feature_file_count"], 2)
            self.assertEqual(coverage[0]["available_exposure"], 15.0)

    def test_partial_config_has_all_stage2_pairs(self):
        config_path = (
            REPO_ROOT
            / "snakemake/metadata/configs/"
            "unified_partial_distributions_20261007_v1.yaml"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        comparisons = config["stages"]["stage2"]["comparisons"]
        self.assertEqual(
            set(config["features"]),
            {
                "density_sndsw_scifi",
                "scifi_cluster_qdc_per_hit",
                "scifi_qdc_per_hit_direct",
                "scifi_vertical_cluster1_qdc_std",
                "scifi_horizontal_cluster1_qdc_std",
            },
        )
        self.assertEqual(
            config["features"]["scifi_vertical_cluster1_qdc_std"]["feature_cut"],
            "scifi_vertical_cluster1_exists == 1",
        )
        self.assertEqual(
            config["features"]["scifi_horizontal_cluster1_qdc_std"]["feature_cut"],
            "scifi_horizontal_cluster1_exists == 1",
        )
        self.assertEqual(
            set(comparisons),
            {
                "electron_inclusive",
                "electron_50",
                "electron_100",
                "electron_200",
                "pion_180",
            },
        )
        for groups in comparisons.values():
            self.assertEqual(len(groups), 2)
            self.assertEqual(
                {config["groups"][group]["family"] for group in groups},
                {"testbeam_mc_2024", "testbeam_data_2024"},
            )


if __name__ == "__main__":
    unittest.main()
