#!/usr/bin/env python3
"""Prepare exact fold/epoch Muon-DIS source and sampled-draw manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train_3d/src"))

from train_3d.data.classification_labels import validate_scifi_classification_config
from train_3d.data.data_driven_fold_dataset import HierarchicalTrainingSampler


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def density_counts(index_path: str, n_events: int) -> tuple[int, int]:
    with np.load(index_path, allow_pickle=False) as payload:
        valid = 0
        invalid = 0
        for name in payload.files:
            entries = np.asarray(payload[name])
            if entries.dtype.kind not in "iu":
                raise TypeError(f"Non-integral density index {index_path}:{name}")
            if len(entries) and (
                int(entries.min()) < 0 or int(entries.max()) >= int(n_events)
            ):
                raise ValueError(f"Out-of-range density index {index_path}:{name}")
            if name.startswith("bin_"):
                valid += len(entries)
            elif name == "invalid":
                invalid += len(entries)
            else:
                raise ValueError(f"Unknown density index array {index_path}:{name}")
    if valid + invalid != int(n_events):
        raise ValueError(
            f"Density index closure failed for {index_path}: "
            f"{valid} + {invalid} != {n_events}"
        )
    return int(valid), int(invalid)


def prepare_manifests(training_config: dict, fold_views: pd.DataFrame,
                      cv_fold: int, epoch: int):
    validate_scifi_classification_config(training_config)
    cv_fold = int(cv_fold)
    epoch = int(epoch)
    if cv_fold < 0 or cv_fold >= int(training_config["data"]["n_folds"]):
        raise ValueError("cv_fold is outside the configured fold range")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")

    data = training_config["data"]
    sampler = HierarchicalTrainingSampler(
        fold_views, cv_fold=cv_fold,
        seed=int(training_config["runtime"]["seed"]),
        class_probabilities=data["class_probabilities"],
        domain_probabilities=data.get("domain_probabilities"),
        muondis_density_probabilities=data["muondis_density_probabilities"],
    )
    epoch_size = int(data["epoch_size"])
    references = sampler.sample_epoch(epoch_size, epoch)
    class_counts = pd.Series(
        [reference.class_name for reference in references], dtype="object"
    ).value_counts().sort_index()
    muondis = [reference for reference in references
               if reference.class_name == "muon_DIS"]
    if not muondis:
        raise RuntimeError("The sampled epoch contains no Muon-DIS draws")

    sampled = pd.DataFrame([{
        "dataset_file_uid": reference.dataset_file_uid,
        "source_entry": int(reference.source_entry),
        "domain": reference.domain,
        "density_bin": reference.density_bin,
    } for reference in muondis])
    sampled = (
        sampled.groupby(
            ["dataset_file_uid", "source_entry", "domain", "density_bin"],
            sort=True, as_index=False, dropna=False,
        ).size().rename(columns={"size": "multiplicity"})
    )
    sampled["source_entry"] = sampled["source_entry"].astype(np.int64)
    sampled["multiplicity"] = sampled["multiplicity"].astype(np.int64)

    sources = fold_views.loc[
        pd.to_numeric(fold_views["cv_fold"]).astype(int).eq(cv_fold)
        & fold_views["split"].astype(str).eq("train")
        & fold_views["class_name"].astype(str).eq("muon_DIS")
    ].copy()
    if sources.empty or sources["dataset_file_uid"].astype(str).duplicated().any():
        raise ValueError("Expected unique non-empty Muon-DIS training source rows")
    expected_strategy = sources["sampling_strategy"].astype(str)
    if not expected_strategy.eq("uniform_populated_density_bins").all():
        raise ValueError("Muon-DIS training rows must use uniform populated density bins")
    keep = [
        "dataset_file_uid", "source_uid", "domain", "fold", "feature_path",
        "hit3d_path", "density_index_path", "n_events", "event_filter",
    ]
    missing = set(keep) - set(sources.columns)
    if missing:
        raise ValueError(f"Fold views lack source fields: {sorted(missing)}")
    sources = sources[keep].sort_values("dataset_file_uid").reset_index(drop=True)
    source_uids = set(sources["dataset_file_uid"].astype(str))
    sampled_uids = set(sampled["dataset_file_uid"].astype(str))
    if not sampled_uids <= source_uids:
        raise ValueError(f"Sampled unknown sources: {sorted(sampled_uids - source_uids)}")

    valid_counts, invalid_counts = [], []
    for row in sources.itertuples(index=False):
        valid, invalid = density_counts(row.density_index_path, int(row.n_events))
        valid_counts.append(valid)
        invalid_counts.append(invalid)
    sources["valid_density_events"] = valid_counts
    sources["invalid_density_events"] = invalid_counts
    source_lookup = sources.set_index("dataset_file_uid")
    for row in sampled.itertuples(index=False):
        source = source_lookup.loc[str(row.dataset_file_uid)]
        if int(row.source_entry) < 0 or int(row.source_entry) >= int(source.n_events):
            raise ValueError(
                f"Sampled entry outside {row.dataset_file_uid}: {row.source_entry}"
            )
        if str(row.domain) != str(source.domain):
            raise ValueError(f"Sampled domain mismatch for {row.dataset_file_uid}")

    domain_draws = {
        str(key): int(value)
        for key, value in sampled.groupby("domain")["multiplicity"].sum().items()
    }
    density_draws = {
        f"{domain}:{density_bin}": int(value)
        for (domain, density_bin), value in sampled.groupby(
            ["domain", "density_bin"]
        )["multiplicity"].sum().items()
    }
    summary = {
        "version": "muondis_feature_sampling_v1",
        "sampling_stage": "hierarchical_pre_empty_scifi_filter",
        "cv_fold": cv_fold,
        "epoch": epoch,
        "seed": int(training_config["runtime"]["seed"]),
        "epoch_size": epoch_size,
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "muondis_draws": int(sum(reference.class_name == "muon_DIS"
                                  for reference in references)),
        "muondis_unique_events": int(len(sampled)),
        "muondis_repeated_draws": int(sampled["multiplicity"].sum() - len(sampled)),
        "domain_draws": domain_draws,
        "density_draws": density_draws,
        "source_files": int(len(sources)),
        "natural_valid_events": int(sources["valid_density_events"].sum()),
        "natural_invalid_events": int(sources["invalid_density_events"].sum()),
        "natural_domain_events": {
            str(key): int(value) for key, value in sources.groupby("domain")[
                "valid_density_events"
            ].sum().items()
        },
    }
    if int(sampled["multiplicity"].sum()) != summary["muondis_draws"]:
        raise RuntimeError("Sample multiplicity closure failed")
    return sources, sampled, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--fold-views", required=True)
    parser.add_argument("--cv-fold", required=True, type=int)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--output-sources", required=True)
    parser.add_argument("--output-samples", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    sources, sampled, summary = prepare_manifests(
        load_yaml(args.training_config),
        pd.read_csv(args.fold_views, keep_default_na=False),
        args.cv_fold, args.epoch,
    )
    atomic_csv(sources, args.output_sources)
    atomic_csv(sampled, args.output_samples)
    atomic_json(summary, args.output_summary)
    print(
        f"Prepared fold {args.cv_fold} epoch {args.epoch}: "
        f"{summary['natural_valid_events']} natural events, "
        f"{summary['muondis_draws']} sampled draws"
    )


if __name__ == "__main__":
    main()
