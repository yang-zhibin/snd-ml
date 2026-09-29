#!/usr/bin/env python3
"""Focused configuration and merge-audit tests for V7 event displays."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_v7.yaml"
)
MC_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_mc_v7.yaml"
)
VETO0_SCIFI_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_event_display_scifi_veto0_v7.yaml"
)
STUDY_CONFIG = REPO_ROOT / (
    "snakemake/metadata/configs/region_evaluation/"
    "passing_muon_binary_region_study_v7.yaml"
)
MERGER = REPO_ROOT / (
    "evaluation_region_partitions/"
    "merge_passing_muon_v7_event_display_summaries.py"
)
SCIFI_MERGER = REPO_ROOT / (
    "evaluation_region_partitions/"
    "merge_passing_muon_v7_scifi_event_display_summaries.py"
)
SELECTOR = REPO_ROOT / (
    "evaluation_region_partitions/"
    "select_passing_muon_v7_event_display_events.py"
)
RENDERER = REPO_ROOT / (
    "evaluation_region_partitions/plot_passing_muon_v6_event_displays.py"
)

sys.path.insert(0, str(REPO_ROOT / "evaluation_region_partitions"))
from plot_passing_muon_v6_event_displays import SUMMARY_FIELDS  # noqa: E402
from merge_passing_muon_v7_scifi_event_display_summaries import (  # noqa: E402
    PAGE_TYPES as SCIFI_PAGE_TYPES,
    SUMMARY_FIELDS as SCIFI_SUMMARY_FIELDS,
)
from select_passing_muon_v6_event_display_events import MANIFEST_FIELDS  # noqa: E402
from select_passing_muon_v7_event_display_events import (  # noqa: E402
    category_definitions,
    configured_samples,
    sources_for_sample,
    validate_sampling,
)


def load(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_summary(
    path, category, target, duplicate_identity=False, sample="run8329",
):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for page in range(1, target + 1):
            row = {field: "" for field in SUMMARY_FIELDS}
            row.update({
                "category": category,
                "page": page,
                "sample": sample,
                "source_feature_path": "/tmp/source.root",
                "source_event_index": 0 if duplicate_identity else (
                    1000 * sum(ord(value) for value in category) + page
                ),
            })
            writer.writerow(row)


def write_no_event_summary(path, category, sample="run8329"):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        row = {field: "" for field in SUMMARY_FIELDS}
        row.update({
            "category": category,
            "page": 1,
            "sample": sample,
            "status": "no_events",
            "message": "No selected event is displayable for this category.",
        })
        writer.writerow(row)


def write_feature_source(ROOT, path, source_index):
    # One event per base category. Failure events also populate the configured
    # SciFi > 100 subsets, intentionally overlapping their base categories.
    values = {
        "runId": [8329] * 4,
        "eventId": [source_index * 100 + value for value in range(4)],
        "eventIndex": [source_index * 1000 + value for value in range(4)],
        "count_veto1": [1, 0, 1, 2],
        "count_veto2": [1, 0, 1, 2],
        "count_veto3": [1, 0, 1, 1],
        "scifi_n_active_stations_xy": [5, 5, 4, 5],
        "density_sndsw_scifi": [50.0, 6000.0, 6000.0, 6000.0],
        "pdgCode": [13] * 4,
    }
    for station in range(1, 6):
        values[f"count_scifi{station}"] = [2, 21, 21, 21]
        values[f"count_us{station}"] = [1, 1, 1, 1]
    for station in range(1, 5):
        values[f"count_ds{station}"] = [1, 1, 1, 1]
    for detector, stations in (("veto", 3), ("scifi", 5), ("us", 5), ("ds", 4)):
        values[f"count_{detector}"] = np.sum(
            [values[f"count_{detector}{station}"] for station in range(1, stations + 1)],
            axis=0,
        )
    arrays = {
        name: np.asarray(value, dtype=np.int64)
        for name, value in values.items()
    }
    result = ROOT.RDF.FromNumpy(arrays).Snapshot("sndData", str(path))
    if hasattr(result, "GetValue"):
        result.GetValue()


class PassingMuonV7EventDisplayTest(unittest.TestCase):
    def test_veto0_scifi_categories_are_independent_of_loose_selection(self):
        config = load(VETO0_SCIFI_CONFIG)
        study = load(STUDY_CONFIG)
        definitions = category_definitions(config, study)
        expected = [
            f"run8329__inclusive__veto_eq0__scifi_density_gt{threshold}"
            for threshold in (100, 500, 1000, 5000)
        ]
        self.assertEqual(list(definitions), expected)
        self.assertEqual(config["frozen_input"]["categories"], expected)
        for threshold, definition in zip((100, 500, 1000, 5000), definitions.values()):
            expression = definition["expression"]
            self.assertEqual(definition["population"], "inclusive")
            self.assertEqual(
                expression,
                f"(true) && (count_veto == 0) && "
                f"(density_sndsw_scifi > {threshold})",
            )
            self.assertNotIn("!", expression)

    def test_data_only_categories_and_sampling(self):
        config = load(CONFIG)
        study = load(STUDY_CONFIG)
        self.assertEqual(configured_samples(config), ["run8329"])
        definitions = category_definitions(config, study)
        self.assertEqual(list(definitions), [
            "run8329__candidate",
            "run8329__failure__veto_lt3",
            "run8329__failure__veto_eq3",
            "run8329__failure__veto_gt3",
            *(
                f"run8329__failure__{veto}__{criterion}"
                for criterion in (
                    "scifi_hit_count_gt100",
                    "scifi_density_gt100",
                    "scifi_density_gt500",
                    "scifi_density_gt1000",
                    "scifi_density_gt5000",
                )
                for veto in ("veto_lt3", "veto_eq3", "veto_gt3")
            ),
        ])
        self.assertIn("!", definitions["run8329__failure__veto_eq3"]["expression"])
        self.assertIn(
            "count_veto == 3",
            definitions["run8329__failure__veto_eq3"]["expression"],
        )
        self.assertIn(
            "count_scifi > 100",
            definitions[
                "run8329__failure__veto_eq3__scifi_hit_count_gt100"
            ]["expression"],
        )
        self.assertIn(
            "density_sndsw_scifi > 5000",
            definitions[
                "run8329__failure__veto_eq3__scifi_density_gt5000"
            ]["expression"],
        )
        self.assertEqual(validate_sampling(config), (25, 50, 10))

    def test_mc_categories_sources_and_baseline(self):
        config = load(MC_CONFIG)
        study = load(STUDY_CONFIG)
        self.assertEqual(config["selection"], load(CONFIG)["selection"])
        self.assertEqual(configured_samples(config), ["mc_muon_up"])
        definitions = category_definitions(config, study)
        self.assertEqual(len(definitions), 19)
        self.assertEqual(
            list(definitions)[:4],
            [
                "mc_muon_up__candidate",
                "mc_muon_up__failure__veto_lt3",
                "mc_muon_up__failure__veto_eq3",
                "mc_muon_up__failure__veto_gt3",
            ],
        )
        for definition in definitions.values():
            self.assertIn("std::abs(pdgCode) == 13", definition["expression"])
        self.assertIn(
            "count_scifi > 100",
            definitions[
                "mc_muon_up__failure__veto_eq3__scifi_hit_count_gt100"
            ]["expression"],
        )
        sources = sources_for_sample(
            "mc_muon_up", config, study, REPO_ROOT
        )
        self.assertEqual(len(sources), 15)
        self.assertEqual(
            set(source["production_key"] for source in sources),
            set(study["mc_muon_up"]["productions"]),
        )
        self.assertTrue(all(source["source_feature_path"] for source in sources))
        self.assertTrue(all(source["event_source_path"] for source in sources))
        self.assertTrue(all(source["geo_path"] for source in sources))

    def test_selector_builds_unique_round_robin_reserves(self):
        import ROOT

        ROOT.gROOT.SetBatch(True)
        with tempfile.TemporaryDirectory(prefix="v7_event_selector_test_") as tmp:
            tmp = Path(tmp)
            sources = []
            for source_index in range(2):
                feature = tmp / f"source_{source_index}.root"
                write_feature_source(ROOT, feature, source_index)
                sources.append({
                    "source_file_index": source_index,
                    "feature_full_path": str(feature),
                    "digi_path": str(tmp / f"digi_{source_index}.root"),
                    "geo_path": str(tmp / "geo.root"),
                })
            metadata = tmp / "metadata.csv"
            pd.DataFrame(sources).to_csv(metadata, index=False)
            config = load(CONFIG)
            config["study_config"] = str(STUDY_CONFIG)
            config["input"]["real_data_metadata"] = str(metadata)
            config["sampling"].update({
                "events_per_category": 1,
                "reserve_factor": 2,
                "minimum_sources": 2,
            })
            config_path = tmp / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            manifest = tmp / "selected.csv"
            counts = tmp / "counts.csv"
            subprocess.run([
                sys.executable,
                str(SELECTOR),
                "--config", str(config_path),
                "--output-manifest", str(manifest),
                "--output-counts", str(counts),
                "--repo-root", str(REPO_ROOT),
            ], check=True, cwd=REPO_ROOT)
            selected = pd.read_csv(manifest, keep_default_na=False)
            self.assertEqual(len(selected), 38)
            self.assertEqual(set(selected["source_file_index"]), {0, 1})
            self.assertTrue(all(selected.groupby("category").size() == 2))
            self.assertFalse(selected.duplicated([
                "category", "sample", "source_feature_path", "source_event_index"
            ]).any())
            self.assertTrue(selected.duplicated([
                "sample", "source_feature_path", "source_event_index"
            ]).any())

    def test_selector_builds_veto0_density_reserves(self):
        import ROOT

        ROOT.gROOT.SetBatch(True)
        with tempfile.TemporaryDirectory(prefix="v7_veto0_selector_test_") as tmp:
            tmp = Path(tmp)
            sources = []
            for source_index in range(2):
                feature = tmp / f"source_{source_index}.root"
                write_feature_source(ROOT, feature, source_index)
                sources.append({
                    "source_file_index": source_index,
                    "feature_full_path": str(feature),
                    "digi_path": str(tmp / f"digi_{source_index}.root"),
                    "geo_path": str(tmp / "geo.root"),
                })
            metadata = tmp / "metadata.csv"
            pd.DataFrame(sources).to_csv(metadata, index=False)
            config = load(VETO0_SCIFI_CONFIG)
            config["study_config"] = str(STUDY_CONFIG)
            config["input"]["real_data_metadata"] = str(metadata)
            config["sampling"].update({
                "events_per_category": 1,
                "reserve_factor": 2,
                "minimum_sources": 2,
            })
            config_path = tmp / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            manifest = tmp / "selected.csv"
            counts = tmp / "counts.csv"
            subprocess.run([
                sys.executable,
                str(SELECTOR),
                "--config", str(config_path),
                "--output-manifest", str(manifest),
                "--output-counts", str(counts),
                "--repo-root", str(REPO_ROOT),
            ], check=True, cwd=REPO_ROOT)
            selected = pd.read_csv(manifest, keep_default_na=False)
            self.assertEqual(len(selected), 8)
            self.assertTrue(all(selected.groupby("category").size() == 2))
            self.assertEqual(set(selected["count_veto"]), {0})
            self.assertTrue(
                selected["selection_expression"].str.contains(
                    "density_sndsw_scifi >", regex=False
                ).all()
            )
            self.assertFalse(selected.duplicated([
                "category", "sample", "source_feature_path", "source_event_index"
            ]).any())

    def test_veto0_scifi_closure_accepts_sparse_selection_manifest(self):
        config = load(VETO0_SCIFI_CONFIG)
        categories = config["frozen_input"]["categories"]
        with tempfile.TemporaryDirectory(prefix="v7_veto0_closure_test_") as tmp:
            tmp = Path(tmp)
            manifest = tmp / "selected.csv"
            with open(manifest, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                for index, category in enumerate(categories):
                    row = {field: "" for field in MANIFEST_FIELDS}
                    row.update({
                        "category": category,
                        "sample": "run8329",
                        "source_feature_path": f"/tmp/source_{index}.root",
                        "source_event_index": index,
                    })
                    writer.writerow(row)

            summaries = []
            audits = []
            for index, category in enumerate(categories):
                summary = tmp / f"summary_{index}.csv"
                with open(summary, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=SCIFI_SUMMARY_FIELDS)
                    writer.writeheader()
                    for page, page_type in enumerate(SCIFI_PAGE_TYPES, start=1):
                        row = {field: "" for field in SCIFI_SUMMARY_FIELDS}
                        row.update({
                            "category": category,
                            "event_sequence": 1,
                            "document_page": page,
                            "page_type": page_type,
                            "sample": "run8329",
                            "source_feature_path": f"/tmp/source_{index}.root",
                            "source_event_index": index,
                            "coordinate_validation_passed": "true",
                            "unmapped_scifi_hits": 0,
                        })
                        writer.writerow(row)
                summaries.append(summary)
                audit = tmp / f"audit_{index}.json"
                audit.write_text(json.dumps({
                    "category": category,
                    "event_count": 1,
                    "page_count": 3,
                    "page_types_per_event": list(SCIFI_PAGE_TYPES),
                    "limited_test_run": False,
                }), encoding="utf-8")
                audits.append(audit)

            output = tmp / "rendered.csv"
            subprocess.run([
                sys.executable,
                str(SCIFI_MERGER),
                "--config", str(VETO0_SCIFI_CONFIG),
                "--manifests", str(manifest),
                "--output", str(output),
                "--summaries", *(str(path) for path in summaries),
                "--audits", *(str(path) for path in audits),
            ], check=True, cwd=REPO_ROOT)
            with open(output, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4 * len(SCIFI_PAGE_TYPES))

    def test_selector_builds_mc_reserves(self):
        import ROOT

        ROOT.gROOT.SetBatch(True)
        with tempfile.TemporaryDirectory(prefix="v7_mc_selector_test_") as tmp:
            tmp = Path(tmp)
            sources = []
            for source_index, partition in enumerate(("scoring_1.8", "scoring_2")):
                feature = tmp / f"mc_source_{source_index}.root"
                write_feature_source(ROOT, feature, source_index)
                sources.append({
                    "data_type": "MC_muon",
                    "subfolder": "up",
                    "partition": partition,
                    "output_base_path": "",
                    "feature_path": str(feature),
                    "digi_path": str(tmp / f"mc_digi_{source_index}.root"),
                    "geo_path": str(tmp / f"mc_geo_{source_index}.root"),
                })
            metadata = tmp / "mc_metadata.csv"
            pd.DataFrame(sources).to_csv(metadata, index=False)

            study = load(STUDY_CONFIG)
            study["mc_muon_up"]["expected_source_files"] = 2
            study_path = tmp / "study.yaml"
            study_path.write_text(
                yaml.safe_dump(study, sort_keys=False), encoding="utf-8"
            )
            config = load(MC_CONFIG)
            config["study_config"] = str(study_path)
            config["input"]["mc_metadata"] = str(metadata)
            config["sampling"].update({
                "events_per_category": 1,
                "reserve_factor": 2,
                "minimum_sources": 2,
            })
            config_path = tmp / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            manifest = tmp / "selected.csv"
            counts = tmp / "counts.csv"
            subprocess.run([
                sys.executable,
                str(SELECTOR),
                "--config", str(config_path),
                "--output-manifest", str(manifest),
                "--output-counts", str(counts),
                "--repo-root", str(REPO_ROOT),
            ], check=True, cwd=REPO_ROOT)
            selected = pd.read_csv(manifest, keep_default_na=False)
            self.assertEqual(len(selected), 38)
            self.assertEqual(set(selected["sample"]), {"mc_muon_up"})
            self.assertEqual(
                set(selected["production_key"]), {"scoring_1p8", "scoring_2"}
            )
            self.assertTrue(all(selected.groupby("category").size() == 2))
            self.assertFalse(selected.duplicated([
                "category", "sample", "source_feature_path", "source_event_index"
            ]).any())

    def test_merge_requires_nineteen_categories_and_exact_pages(self):
        config = load(CONFIG)
        categories = list(category_definitions(config, load(STUDY_CONFIG)))
        target = int(config["sampling"]["events_per_category"])
        with tempfile.TemporaryDirectory(prefix="v7_event_display_test_") as tmp:
            tmp = Path(tmp)
            summaries = []
            for category in categories:
                path = tmp / f"{category}.csv"
                write_summary(path, category, target)
                summaries.append(path)
            output = tmp / "rendered.csv"
            subprocess.run([
                sys.executable,
                str(MERGER),
                "--config", str(CONFIG),
                "--output", str(output),
                *(str(path) for path in summaries),
            ], check=True, cwd=REPO_ROOT)
            with open(output, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 19 * target)
            self.assertEqual(
                [row["category"] for row in rows[::target]], categories
            )

    def test_merge_accepts_mc_only_categories(self):
        config = load(MC_CONFIG)
        categories = list(category_definitions(config, load(STUDY_CONFIG)))
        target = int(config["sampling"]["events_per_category"])
        with tempfile.TemporaryDirectory(prefix="v7_mc_event_merge_test_") as tmp:
            tmp = Path(tmp)
            summaries = []
            for category in categories:
                path = tmp / f"{category}.csv"
                write_summary(
                    path, category, target, sample="mc_muon_up"
                )
                summaries.append(path)
            output = tmp / "rendered.csv"
            subprocess.run([
                sys.executable,
                str(MERGER),
                "--config", str(MC_CONFIG),
                "--output", str(output),
                *(str(path) for path in summaries),
            ], check=True, cwd=REPO_ROOT)
            with open(output, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 19 * target)
            self.assertEqual({row["sample"] for row in rows}, {"mc_muon_up"})

    def test_renderer_and_merger_accept_no_event_category(self):
        config = load(MC_CONFIG)
        categories = list(category_definitions(config, load(STUDY_CONFIG)))
        target = int(config["sampling"]["events_per_category"])
        empty_category = categories[-1]
        with tempfile.TemporaryDirectory(prefix="v7_no_event_test_") as tmp:
            tmp = Path(tmp)
            manifest = tmp / "empty_manifest.csv"
            with open(manifest, "w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS).writeheader()
            placeholder_pdf = tmp / "placeholder.pdf"
            placeholder_summary = tmp / "placeholder.csv"
            subprocess.run([
                sys.executable,
                str(RENDERER),
                "--config", str(MC_CONFIG),
                "--manifest", str(manifest),
                "--category", empty_category,
                "--output-pdf", str(placeholder_pdf),
                "--output-summary", str(placeholder_summary),
            ], check=True, cwd=REPO_ROOT)
            self.assertTrue(placeholder_pdf.is_file())
            with open(placeholder_summary, newline="", encoding="utf-8") as stream:
                placeholder_rows = list(csv.DictReader(stream))
            self.assertEqual(len(placeholder_rows), 1)
            self.assertEqual(placeholder_rows[0]["status"], "no_events")
            self.assertEqual(placeholder_rows[0]["page"], "1")

            summaries = []
            for category in categories:
                path = tmp / f"merge_{category}.csv"
                if category == empty_category:
                    write_no_event_summary(path, category, sample="mc_muon_up")
                else:
                    write_summary(path, category, target, sample="mc_muon_up")
                summaries.append(path)
            output = tmp / "rendered.csv"
            subprocess.run([
                sys.executable,
                str(MERGER),
                "--config", str(MC_CONFIG),
                "--output", str(output),
                *(str(path) for path in summaries),
            ], check=True, cwd=REPO_ROOT)
            with open(output, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 18 * target + 1)


if __name__ == "__main__":
    unittest.main()
