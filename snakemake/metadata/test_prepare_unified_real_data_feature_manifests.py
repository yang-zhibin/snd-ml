import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from prepare_unified_real_data_feature_manifests import build_manifests


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_source(tmp_path, uid, no_veto_count, veto_count, corrupt=False):
    source_metadata = tmp_path / f"{uid}_source.csv"
    pd.DataFrame(
        [{"digi_path": f"/input/{uid}.root", "geo_path": "/geometry/geofile.root"}]
    ).to_csv(source_metadata, index=False)
    no_veto = tmp_path / f"{uid}_no_veto.root"
    veto = tmp_path / f"{uid}_veto.root"
    no_veto.write_bytes(f"no-veto-{uid}".encode())
    veto.write_bytes(f"veto-{uid}".encode())
    validation = tmp_path / f"{uid}_validated.json"
    summary = tmp_path / f"{uid}_summary.json"
    payload = {
        "status": "passed",
        "source_uid": uid,
        "input_event_count": 100,
        "no_veto_event_count": no_veto_count,
        "veto_10pct_event_count": veto_count,
        "no_veto_sha256": "bad" if corrupt else _sha256(no_veto),
        "veto_10pct_sha256": _sha256(veto),
    }
    validation.write_text(json.dumps(payload), encoding="utf-8")
    summary.write_text(
        json.dumps(
            {
                "source_uid": uid,
                "count_conservation_passed": True,
                "no_veto_event_count": no_veto_count,
                "veto_10pct_event_count": veto_count,
            }
        ),
        encoding="utf-8",
    )
    return {
        "source_uid": uid,
        "source_metadata_file": str(source_metadata),
        "source_metadata_row": 0,
        "digi_path": f"/input/{uid}.root",
        "input_tree": "cbmsim",
        "input_event_count": 100,
        "lumi_per_file": 0.25,
        "no_veto_index_path": str(no_veto),
        "veto_10pct_index_path": str(veto),
        "selection_summary_path": str(summary),
        "validation_record": str(validation),
    }


def _write_config(tmp_path, rows):
    selection = tmp_path / "selection.csv"
    pd.DataFrame(rows).to_csv(selection, index=False)
    config = {
        "version": "test_real_data_distributions_v1",
        "profiles": {
            "no_veto": {
                "family": "real_data_2024_no_veto",
                "index_column": "no_veto_index_path",
                "count_field": "no_veto_event_count",
                "checksum_field": "no_veto_sha256",
                "sampling_probability": 1.0,
                "event_weight": 1.0,
            },
            "veto_10pct": {
                "family": "real_data_2024_veto_10pct",
                "index_column": "veto_10pct_index_path",
                "count_field": "veto_10pct_event_count",
                "checksum_field": "veto_10pct_sha256",
                "sampling_probability": 0.1,
                "event_weight": 10.0,
            },
        },
        "inputs": {"selection_manifest": str(selection)},
        "source_manifests": {
            "real_data_2024_no_veto": str(tmp_path / "no_veto.csv"),
            "real_data_2024_veto_10pct": str(tmp_path / "veto.csv"),
        },
        "outputs": {"feature_root": str(tmp_path / "features")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_builds_two_profile_manifests_and_picks_smallest_canary(tmp_path):
    large = _write_source(tmp_path, "unified_src_aaaaaaaaaaaaaaaaaaaa", 8, 6)
    small = _write_source(tmp_path, "unified_src_bbbbbbbbbbbbbbbbbbbb", 2, 3)
    config = _write_config(tmp_path, [large, small])

    manifests, coverage = build_manifests(config, workers=2)

    no_veto = manifests["real_data_2024_no_veto"]
    veto = manifests["real_data_2024_veto_10pct"]
    assert len(no_veto) == len(veto) == 2
    assert no_veto.loc[no_veto["is_canary"].eq(1), "selected_event_count"].item() == 2
    assert veto.loc[veto["is_canary"].eq(1), "selected_event_count"].item() == 3
    assert set(no_veto["selection_mode"]) == {"indexed_cutflow"}
    assert set(no_veto["exposure_kind"]) == {"event_count_only"}
    assert set(veto["sampling_probability"]) == {0.1}
    assert set(veto["event_weight"]) == {10.0}
    assert coverage["validated_source_count"] == 2


def test_excludes_a_source_when_an_index_checksum_is_wrong(tmp_path):
    good = _write_source(tmp_path, "unified_src_cccccccccccccccccccc", 4, 5)
    bad = _write_source(
        tmp_path, "unified_src_dddddddddddddddddddd", 7, 8, corrupt=True
    )
    config = _write_config(tmp_path, [good, bad])

    manifests, coverage = build_manifests(config, workers=2)

    for frame in manifests.values():
        assert list(frame["source_uid"]) == ["unified_src_cccccccccccccccccccc"]
    assert coverage["validated_source_count"] == 1
    assert coverage["status_counts"] == {
        "available": 1,
        "no_veto_index_checksum_mismatch": 1,
    }
