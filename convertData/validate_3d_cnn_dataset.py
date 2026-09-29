#!/usr/bin/env python3
"""Validate class, lineage, fold, density, closure, and luminosity invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_driven_3d_dataset import configured_density_bins, load_yaml, truthy


def require_columns(frame: pd.DataFrame, columns: set[str], label: str):
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks columns: {sorted(missing)}")


def validate(config: dict, sources: pd.DataFrame, audits: pd.DataFrame,
             catalog: pd.DataFrame, folded: pd.DataFrame,
             density_manifest: pd.DataFrame,
             fold_views: pd.DataFrame | None = None,
             entry_manifest: pd.DataFrame | None = None,
             neutrino_audits: pd.DataFrame | None = None) -> dict:
    require_columns(sources, {
        "source_uid", "fold_group_uid", "lumi_source_uid", "lumi_per_file",
        "input_available", "input_empty", "feature_entries", "hit3d_entries",
        "role",
    }, "source catalog")
    require_columns(catalog, {
        "dataset_file_uid", "source_uid", "fold_group_uid", "lumi_source_uid",
        "class_id", "class_name", "domain", "n_events", "lumi_per_file",
    }, "dataset catalog")
    require_columns(folded, set(catalog.columns) | {"fold"}, "folded catalog")
    if sources["source_uid"].astype(str).duplicated().any():
        raise ValueError("Source catalog has duplicate source_uid values")
    if catalog["dataset_file_uid"].astype(str).duplicated().any():
        raise ValueError("Dataset catalog has duplicate dataset_file_uid values")
    unknown_sources = set(catalog["source_uid"].astype(str)) - set(sources["source_uid"].astype(str))
    if unknown_sources:
        raise ValueError(f"Dataset catalog references unknown sources: {sorted(unknown_sources)[:5]}")
    if len(catalog) != len(folded):
        raise ValueError("Folded catalog row count differs from dataset catalog")
    if set(catalog["dataset_file_uid"].astype(str)) != set(folded["dataset_file_uid"].astype(str)):
        raise ValueError("Folded catalog does not contain the same dataset_file_uid values")
    if folded.groupby("fold_group_uid")["fold"].nunique().max() != 1:
        raise ValueError("One fold group appears in multiple folds")
    n_folds = int(config["cross_validation"]["n_folds"])
    actual_folds = set(pd.to_numeric(folded["fold"]).astype(int))
    if actual_folds != set(range(n_folds)):
        raise ValueError(f"Expected folds 0..{n_folds - 1}, found {sorted(actual_folds)}")

    class_ids = {name: int(value["class_id"]) for name, value in config["classes"].items()}
    observed_pairs = set(zip(folded["class_name"].astype(str), pd.to_numeric(folded["class_id"]).astype(int)))
    invalid_pairs = observed_pairs - set(class_ids.items())
    if invalid_pairs:
        raise ValueError(f"Invalid class name/id pairs: {sorted(invalid_pairs)}")
    missing_required = [
        name for name, value in config["classes"].items()
        if bool(value.get("required", True)) and name not in set(folded["class_name"])
    ]
    if missing_required:
        raise ValueError(f"Required classes are absent: {missing_required}")
    for name in set(folded["class_name"]):
        folds = set(pd.to_numeric(folded.loc[folded["class_name"] == name, "fold"]).astype(int))
        if len(folds) != n_folds:
            missing_folds = sorted(set(range(n_folds)) - folds)
            raise ValueError(f"Class {name} is absent from folds: {missing_folds}")

    source_lumi = sources[["lumi_source_uid", "lumi_per_file"]].drop_duplicates()
    if source_lumi["lumi_source_uid"].duplicated().any():
        raise ValueError("One lumi_source_uid has inconsistent source luminosities")
    catalog_lumi = catalog[["lumi_source_uid", "lumi_per_file"]].drop_duplicates()
    if catalog_lumi["lumi_source_uid"].duplicated().any():
        raise ValueError("One lumi_source_uid has inconsistent dataset luminosities")
    source_lumi_values = source_lumi.set_index("lumi_source_uid")["lumi_per_file"].astype(float)
    for _, row in catalog_lumi.iterrows():
        lumi_uid = str(row["lumi_source_uid"])
        if lumi_uid not in source_lumi_values.index:
            raise ValueError(f"Dataset catalog references unknown lumi source {lumi_uid}")
        if abs(float(row["lumi_per_file"]) - float(source_lumi_values.loc[lumi_uid])) > 1e-12:
            raise ValueError(f"Luminosity changed for source {lumi_uid}")

    empty_sources = sources["input_empty"].map(truthy)
    if (empty_sources & ~sources["input_available"].map(truthy)).any():
        raise ValueError("A source is marked empty without both input products")
    if not (
        pd.to_numeric(sources.loc[empty_sources, "feature_entries"]).eq(0).all()
        and pd.to_numeric(sources.loc[empty_sources, "hit3d_entries"]).eq(0).all()
    ):
        raise ValueError("An empty source does not have two zero-entry trees")

    expected_audits = set(sources.loc[
        sources["input_available"].map(truthy)
        & ~empty_sources
        & sources["role"].eq("muon_split"),
        "source_uid",
    ].astype(str))
    if audits.empty and expected_audits:
        raise ValueError(f"Muon audits are missing for {len(expected_audits)} available sources")
    if not audits.empty:
        require_columns(audits, {
            "source_uid", "input_entries", "parent_entries", "excluded_by_parent_baseline",
            "pass_entries", "fail_entries", "identity_mismatches", "pass_density_counts",
            "output_identity_mismatches", "fail_density_counts",
            "blinding_applied", "blinded_entries", "blinded_fail_leakage",
            "signal_region_entries", "signal_region_feature_path",
            "signal_region_hit3d_path",
        }, "muon audit")
        if audits["source_uid"].astype(str).duplicated().any():
            raise ValueError("Muon audit has duplicate sources")
        actual_audits = set(audits["source_uid"].astype(str))
        if expected_audits != actual_audits:
            raise ValueError(
                "Muon audit source mismatch: "
                f"missing={sorted(expected_audits - actual_audits)[:5]}, "
                f"extra={sorted(actual_audits - expected_audits)[:5]}"
            )
        for _, row in audits.iterrows():
            parent = int(row["parent_entries"])
            if int(row["input_entries"]) != parent + int(row["excluded_by_parent_baseline"]):
                raise ValueError(f"Parent baseline closure failed for {row['source_uid']}")
            if parent != int(row["pass_entries"]) + int(row["fail_entries"]):
                raise ValueError(f"Pass/fail closure failed for {row['source_uid']}")
            if int(row["identity_mismatches"]) != 0:
                raise ValueError(f"Identity mismatch recorded for {row['source_uid']}")
            if int(row["output_identity_mismatches"]) != 0:
                raise ValueError(f"Output identity mismatch recorded for {row['source_uid']}")
            if int(row["blinded_fail_leakage"]) != 0:
                raise ValueError(f"Blinded data leaked into fail output for {row['source_uid']}")
            if str(row.get("domain", "")) == "data":
                if not truthy(row["blinding_applied"]):
                    raise ValueError(f"Data blinding was not applied for {row['source_uid']}")
                if int(row["input_entries"]) != parent + int(row["blinded_entries"]):
                    raise ValueError(f"Data blinding closure failed for {row['source_uid']}")
                if int(row["signal_region_entries"]) != int(row["blinded_entries"]):
                    raise ValueError(f"Sealed signal-region count mismatch for {row['source_uid']}")
            for prefix in ("pass", "fail"):
                counts = json.loads(str(row[f"{prefix}_density_counts"]))
                if sum(int(value) for value in counts.values()) != int(row[f"{prefix}_entries"]):
                    raise ValueError(f"Density closure failed for {row['source_uid']} {prefix}")

        sealed_feature_paths = {
            str(value).strip() for value in audits["signal_region_feature_path"]
            if str(value).strip()
        }
        sealed_hit3d_paths = {
            str(value).strip() for value in audits["signal_region_hit3d_path"]
            if str(value).strip()
        }
        catalog_feature_paths = set(
            catalog.get("feature_path", pd.Series(dtype=str)).astype(str).str.strip()
        )
        catalog_hit3d_paths = set(
            catalog.get("hit3d_path", pd.Series(dtype=str)).astype(str).str.strip()
        )
        leaked_features = sealed_feature_paths & catalog_feature_paths
        leaked_hits = sealed_hit3d_paths & catalog_hit3d_paths
        if leaked_features or leaked_hits:
            raise ValueError(
                "Sealed signal-region files leaked into the dataset catalog: "
                f"features={sorted(leaked_features)[:3]}, hits={sorted(leaked_hits)[:3]}"
            )
        if fold_views is not None:
            view_hit3d_paths = set(
                fold_views.get("hit3d_path", pd.Series(dtype=str)).astype(str).str.strip()
            )
            leaked_view_hits = sealed_hit3d_paths & view_hit3d_paths
            if leaked_view_hits:
                raise ValueError(
                    "Sealed signal-region files leaked into fold views: "
                    f"{sorted(leaked_view_hits)[:3]}"
                )

    neutrino_audits = pd.DataFrame() if neutrino_audits is None else neutrino_audits
    expected_neutrino = set(sources.loc[
        sources["input_available"].map(truthy)
        & ~empty_sources
        & sources["role"].eq("neutrino_split"),
        "source_uid",
    ].astype(str))
    actual_neutrino = set(
        neutrino_audits.get("source_uid", pd.Series(dtype=str)).astype(str)
    )
    if expected_neutrino != actual_neutrino:
        raise ValueError(
            "Neutrino audit source mismatch: "
            f"missing={sorted(expected_neutrino - actual_neutrino)[:5]}, "
            f"extra={sorted(actual_neutrino - expected_neutrino)[:5]}"
        )
    if not neutrino_audits.empty:
        require_columns(neutrino_audits, {
            "source_uid", "input_entries", "partitioned_entries",
            "unclassified_entries", "identity_mismatches",
            "output_identity_mismatches", "partitions",
        }, "neutrino audit")
        if neutrino_audits["source_uid"].astype(str).duplicated().any():
            raise ValueError("Neutrino audit has duplicate sources")
        expected_names = {"CC_numu", "CC_nue", "CC_nutau", "NC"}
        for _, row in neutrino_audits.iterrows():
            partitions = json.loads(str(row["partitions"]))
            if set(partitions) != expected_names:
                raise ValueError(f"Incomplete neutrino classes for {row['source_uid']}")
            partition_sum = sum(int(value["entries"]) for value in partitions.values())
            if partition_sum != int(row["partitioned_entries"]):
                raise ValueError(f"Neutrino partition count mismatch for {row['source_uid']}")
            if int(row["input_entries"]) != partition_sum + int(row["unclassified_entries"]):
                raise ValueError(f"Neutrino partition closure failed for {row['source_uid']}")
            if int(row["unclassified_entries"]) != 0:
                raise ValueError(f"Unclassified neutrino events for {row['source_uid']}")
            if int(row["identity_mismatches"]) or int(row["output_identity_mismatches"]):
                raise ValueError(f"Neutrino identity mismatch for {row['source_uid']}")

        physical_neutrino_sources = set(neutrino_audits["source_uid"].astype(str))
        physical_rows = catalog.loc[
            catalog["source_uid"].astype(str).isin(physical_neutrino_sources)
        ]
        filters = physical_rows.get(
            "event_filter", pd.Series("", index=physical_rows.index)
        ).astype(str).str.strip().str.lower()
        if not filters.isin(["true", "1"]).all():
            raise ValueError("Physical neutrino partitions must not retain event filters")

    expected_density = set(
        folded.loc[folded["class_name"].eq("muon_DIS"), "dataset_file_uid"].astype(str)
    )
    actual_density = set(density_manifest.get("dataset_file_uid", pd.Series(dtype=str)).astype(str))
    if expected_density != actual_density:
        raise ValueError(
            "Density-index manifest mismatch: "
            f"missing={sorted(expected_density - actual_density)[:5]}, "
            f"extra={sorted(actual_density - expected_density)[:5]}"
        )
    if not density_manifest.empty:
        require_columns(density_manifest, {
            "dataset_file_uid", "n_events", "density_counts", "density_bin_schema",
        }, "density-index manifest")
        if density_manifest["dataset_file_uid"].astype(str).duplicated().any():
            raise ValueError("Density-index manifest has duplicate dataset files")
        expected_counts = folded.set_index("dataset_file_uid")["n_events"].astype(int)
        expected_schema = [
            {
                "key": f"bin_{item['index']}", "name": item["name"],
                "low": item["low"], "high": None if item["high"] == float("inf") else item["high"],
            }
            for item in configured_density_bins(config)
        ]
        for _, row in density_manifest.iterrows():
            if int(row["n_events"]) != int(expected_counts.loc[str(row["dataset_file_uid"])]):
                raise ValueError(f"Density index count mismatch for {row['dataset_file_uid']}")
            counts = json.loads(str(row["density_counts"]))
            if sum(int(value) for value in counts.values()) != int(row["n_events"]):
                raise ValueError(f"Density count closure failed for {row['dataset_file_uid']}")
            if json.loads(str(row["density_bin_schema"])) != expected_schema:
                raise ValueError(f"Density bin schema mismatch for {row['dataset_file_uid']}")

    if fold_views is not None:
        require_columns(fold_views, {
            "cv_fold", "split", "sampling_strategy", "density_index_path",
            "dataset_file_uid", "fold", "class_name",
        }, "fold views")
        expected_rows = len(folded) * n_folds
        if len(fold_views) != expected_rows:
            raise ValueError(f"Expected {expected_rows} fold-view rows, found {len(fold_views)}")
        if fold_views.duplicated(["cv_fold", "dataset_file_uid"]).any():
            raise ValueError("Fold views contain duplicate CV-fold/dataset-file pairs")
        split_counts = fold_views.groupby(["dataset_file_uid", "split"]).size().unstack(
            fill_value=0
        )
        validation_counts = split_counts.get(
            "validation", pd.Series(0, index=split_counts.index)
        )
        training_counts = split_counts.get(
            "train", pd.Series(0, index=split_counts.index)
        )
        if not validation_counts.eq(1).all() or not training_counts.eq(n_folds - 1).all():
            raise ValueError(
                "Every dataset file must be validation once and training in all other folds"
            )
        for held_out_fold, view in fold_views.groupby("cv_fold"):
            held_out_fold = int(held_out_fold)
            is_validation = view["split"].eq("validation")
            expected_validation = pd.to_numeric(view["fold"]).astype(int).eq(held_out_fold)
            if not is_validation.equals(expected_validation):
                raise ValueError(f"CV view {held_out_fold} does not hold out exactly its own fold")
            if not view.loc[is_validation, "sampling_strategy"].eq("natural").all():
                raise ValueError(f"CV view {held_out_fold} modifies its validation distribution")
            expected_uniform = (~is_validation) & view["class_name"].eq("muon_DIS")
            actual_uniform = view["sampling_strategy"].eq("uniform_populated_density_bins")
            if not actual_uniform.equals(expected_uniform):
                raise ValueError(f"CV view {held_out_fold} has an invalid sampling policy")

    entry_manifest = pd.DataFrame() if entry_manifest is None else entry_manifest
    filters = folded.get("event_filter", pd.Series("true", index=folded.index)).astype(str)
    expected_entries = set(folded.loc[
        ~filters.str.strip().str.lower().isin(["", "true", "1"]),
        "dataset_file_uid",
    ].astype(str))
    actual_entries = set(entry_manifest.get("dataset_file_uid", pd.Series(dtype=str)).astype(str))
    if expected_entries != actual_entries:
        raise ValueError(
            "Entry-index manifest mismatch: "
            f"missing={sorted(expected_entries - actual_entries)[:5]}, "
            f"extra={sorted(actual_entries - expected_entries)[:5]}"
        )
    if not entry_manifest.empty:
        require_columns(entry_manifest, {
            "dataset_file_uid", "entry_index_path", "event_filter", "n_events",
        }, "entry-index manifest")
        expected_counts = folded.set_index("dataset_file_uid")["n_events"].astype(int)
        for _, row in entry_manifest.iterrows():
            if int(row["n_events"]) != int(expected_counts.loc[str(row["dataset_file_uid"])]):
                raise ValueError(f"Entry-index count mismatch for {row['dataset_file_uid']}")

    return {
        "dataset_version": config["dataset_version"],
        "valid": True,
        "source_rows": int(len(sources)),
        "available_source_rows": int(sources["input_available"].map(truthy).sum()),
        "empty_source_rows": int(empty_sources.sum()),
        "dataset_rows": int(len(catalog)),
        "fold_groups": int(folded["fold_group_uid"].nunique()),
        "folds": n_folds,
        "classes_present": sorted(set(folded["class_name"].astype(str))),
        "muon_audit_rows": int(len(audits)),
        "neutrino_audit_rows": int(len(neutrino_audits)),
        "density_index_rows": int(len(density_manifest)),
        "entry_index_rows": int(len(entry_manifest)),
        "fold_view_rows": int(len(fold_views)) if fold_views is not None else 0,
        "unique_lumi_sources_used": int(catalog["lumi_source_uid"].nunique()),
    }


def read_optional(path: str | None) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False) if path else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-catalog", required=True)
    parser.add_argument("--muon-audit")
    parser.add_argument("--neutrino-audit")
    parser.add_argument("--dataset-catalog", required=True)
    parser.add_argument("--folded-catalog", required=True)
    parser.add_argument("--density-index-manifest", required=True)
    parser.add_argument("--entry-index-manifest")
    parser.add_argument("--fold-views", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(
        load_yaml(args.config),
        pd.read_csv(args.source_catalog, keep_default_na=False),
        read_optional(args.muon_audit),
        pd.read_csv(args.dataset_catalog, keep_default_na=False),
        pd.read_csv(args.folded_catalog, keep_default_na=False),
        pd.read_csv(args.density_index_manifest, keep_default_na=False),
        pd.read_csv(args.fold_views, keep_default_na=False),
        read_optional(args.entry_index_manifest),
        read_optional(args.neutrino_audit),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validated {result['dataset_rows']} dataset rows across {result['folds']} folds")


if __name__ == "__main__":
    main()
