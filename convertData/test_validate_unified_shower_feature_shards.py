#!/usr/bin/env python3

from __future__ import annotations

from argparse import Namespace
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_unified_shower_feature_shards import COMMON_RECEIPT_FIELDS, sha256  # noqa: E402
from validate_unified_shower_feature_shards import validate_complete_shards  # noqa: E402


class UnifiedShowerFeatureShardValidationTests(unittest.TestCase):
    def _fixture(self, root: Path, ranges=((0, 2), (2, 5))):
        uid = "unified_src_0123456789abcdef0123"
        records = []
        common = {field: f"test-{field}" for field in COMMON_RECEIPT_FIELDS}
        common.update(
            {
                "source_uid": uid,
                "available_selected_event_count": 5,
                "conversion_max_events": 0,
                "expected_input_event_count": 5,
            }
        )
        for index, (start, stop) in enumerate(ranges):
            feature = root / f"part_{index}.root"
            feature.write_bytes(f"shard-{index}".encode())
            record = root / f"part_{index}.validated.json"
            payload = {
                **common,
                "validation_passed": True,
                "selected_entry_start": start,
                "selected_entry_stop": stop,
                "written_event_count": stop - start,
                "feature_path": str(feature),
                "feature_sha256": sha256(feature),
            }
            record.write_text(json.dumps(payload), encoding="utf-8")
            records.append(str(record))
        return uid, records

    def test_complete_receipt_preserves_ordered_shard_units(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            root = Path(directory)
            uid, records = self._fixture(root)
            output = root / "shards.complete.json"
            report = validate_complete_shards(
                Namespace(
                    shard_record=list(reversed(records)),
                    family="mc_muon_up",
                    expected_source_uid=uid,
                    expected_selected_event_count=5,
                    expected_input_digi_path="test-input_digi_path",
                    expected_input_cutflow_path="test-input_cutflow_path",
                    output=str(output),
                )
            )
            self.assertTrue(report["validation_passed"])
            self.assertEqual(report["shard_count"], 2)
            self.assertEqual(
                [item["selected_entry_start"] for item in report["shards"]],
                [0, 2],
            )
            self.assertTrue(output.is_file())

    def test_gap_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            root = Path(directory)
            uid, records = self._fixture(root, ranges=((0, 2), (3, 5)))
            with self.assertRaisesRegex(RuntimeError, "gap or overlap"):
                validate_complete_shards(
                    Namespace(
                        shard_record=records,
                        family="mc_muon_up",
                        expected_source_uid=uid,
                        expected_selected_event_count=5,
                        expected_input_digi_path="test-input_digi_path",
                        expected_input_cutflow_path="test-input_cutflow_path",
                        output=str(root / "complete.json"),
                    )
                )

    def test_expected_source_uid_is_enforced(self):
        with tempfile.TemporaryDirectory(dir="/tmp/zhibin") as directory:
            root = Path(directory)
            _, records = self._fixture(root)
            with self.assertRaisesRegex(RuntimeError, "source UID"):
                validate_complete_shards(
                    Namespace(
                        shard_record=records,
                        family="mc_muon_up",
                        expected_source_uid="unified_src_ffffffffffffffffffff",
                        expected_selected_event_count=5,
                        expected_input_digi_path="test-input_digi_path",
                        expected_input_cutflow_path="test-input_cutflow_path",
                        output=str(root / "complete.json"),
                    )
                )


if __name__ == "__main__":
    unittest.main()
