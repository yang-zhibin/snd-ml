import json
import math
import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "convertData"))

from build_3d_cnn_fold_views import build_fold_views
from data_driven_3d_dataset import configured_density_bins
from validate_3d_cnn_dataset import validate


CONFIG_PATH = REPO / "snakemake/metadata/configs/ml_dataset/data_driven_3d_cnn_v1.yaml"


class DatasetValidatorTest(unittest.TestCase):
    def make_inputs(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        classes = [
            name for name, value in config["classes"].items()
            if value.get("required", True)
        ]
        folded_rows = []
        source_rows = []
        density_rows = []
        bins = configured_density_bins(config)
        schema = [{
            "key": f"bin_{item['index']}",
            "name": item["name"],
            "low": item["low"],
            "high": None if math.isinf(item["high"]) else item["high"],
        } for item in bins]
        for fold in range(5):
            for class_name in classes:
                uid = f"{class_name}_{fold}"
                class_id = config["classes"][class_name]["class_id"]
                folded_rows.append({
                    "dataset_file_uid": f"dsf_{uid}", "source_uid": f"src_{uid}",
                    "fold_group_uid": f"group_{uid}", "lumi_source_uid": f"src_{uid}",
                    "class_id": class_id, "class_name": class_name,
                    "domain": "data" if class_name in {"passing_muon", "muon_DIS"} else "mc",
                    "n_events": 10, "lumi_per_file": 0.1, "fold": fold,
                    "hit3d_path": f"{uid}.root",
                })
                source_rows.append({
                    "source_uid": f"src_{uid}", "fold_group_uid": f"group_{uid}",
                    "lumi_source_uid": f"src_{uid}", "lumi_per_file": 0.1,
                    "input_available": True, "role": "prepared_physics",
                    "input_empty": False, "feature_entries": 10,
                    "hit3d_entries": 10,
                })
                if class_name == "muon_DIS":
                    counts = {f"bin_{item['index']}": 0 for item in bins}
                    counts["bin_0"] = 10
                    counts["invalid"] = 0
                    density_rows.append({
                        "dataset_file_uid": f"dsf_{uid}", "n_events": 10,
                        "index_path": f"{uid}.npz",
                        "density_counts": json.dumps(counts, sort_keys=True),
                        "density_bin_schema": json.dumps(schema, sort_keys=True),
                    })
        folded = pd.DataFrame(folded_rows)
        density = pd.DataFrame(density_rows)
        views = build_fold_views(folded, density, 5)
        return config, pd.DataFrame(source_rows), folded.drop(columns="fold"), folded, density, views

    def test_complete_metadata_passes(self):
        config, sources, catalog, folded, density, views = self.make_inputs()
        result = validate(
            config, sources, pd.DataFrame(), catalog, folded, density, views,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["fold_view_rows"], 5 * len(catalog))

    def test_modified_validation_sampling_fails(self):
        config, sources, catalog, folded, density, views = self.make_inputs()
        validation_index = views.index[views["split"].eq("validation")][0]
        views.loc[validation_index, "sampling_strategy"] = "uniform_populated_density_bins"
        with self.assertRaisesRegex(ValueError, "validation distribution"):
            validate(config, sources, pd.DataFrame(), catalog, folded, density, views)

    def test_sealed_signal_region_path_in_catalog_fails(self):
        config, sources, catalog, folded, density, views = self.make_inputs()
        source_uid = str(sources.iloc[0]["source_uid"])
        sources.loc[sources["source_uid"].eq(source_uid), "role"] = "muon_split"
        sealed_path = str(catalog.iloc[0]["hit3d_path"])
        audit = pd.DataFrame([{
            "source_uid": source_uid, "domain": "data", "input_entries": 12,
            "parent_entries": 10, "excluded_by_parent_baseline": 2,
            "pass_entries": 6, "fail_entries": 4, "identity_mismatches": 0,
            "output_identity_mismatches": 0,
            "pass_density_counts": json.dumps({"bin_0": 6}),
            "fail_density_counts": json.dumps({"bin_0": 4}),
            "blinding_applied": True, "blinded_entries": 2,
            "blinded_fail_leakage": 0, "signal_region_entries": 2,
            "signal_region_feature_path": "sealed_feature.root",
            "signal_region_hit3d_path": sealed_path,
        }])
        with self.assertRaisesRegex(ValueError, "Sealed signal-region files leaked"):
            validate(config, sources, audit, catalog, folded, density, views)

    def test_each_dataset_must_be_validation_exactly_once(self):
        config, sources, catalog, folded, density, views = self.make_inputs()
        dataset_uid = views.iloc[0]["dataset_file_uid"]
        selected = views["dataset_file_uid"].eq(dataset_uid)
        validation_index = views.index[selected & views["split"].eq("validation")][0]
        views.loc[validation_index, "split"] = "train"
        with self.assertRaisesRegex(ValueError, "validation once"):
            validate(config, sources, pd.DataFrame(), catalog, folded, density, views)


if __name__ == "__main__":
    unittest.main()
