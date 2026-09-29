#!/usr/bin/env python3
"""Assign complete source groups to deterministic, approximately balanced folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from data_driven_3d_dataset import load_yaml


def feature_vector(row: pd.Series) -> dict[str, float]:
    count = max(0, int(row["n_events"]))
    class_name = str(row["class_name"])
    domain = str(row["domain"])
    result = {
        f"class::{class_name}": count,
        f"domain_class::{domain}::{class_name}": count,
    }
    energy = str(row.get("energy_bin", "")).strip()
    if energy and class_name in {"neutral_kaon", "neutron"}:
        result[f"energy::{class_name}::{energy}"] = count
    if class_name == "muon_DIS" and str(row.get("density_counts", "")).strip():
        for name, value in json.loads(str(row["density_counts"])).items():
            result[f"density::{domain}::{name}"] = float(value)
    production = str(row.get("production", "")).strip()
    if production:
        result[f"production::{domain}::{production}"] = count
    return result


def group_vectors(catalog: pd.DataFrame):
    vectors = defaultdict(lambda: defaultdict(float))
    seen_luminosity_sources = set()
    for _, row in catalog.iterrows():
        group = str(row["fold_group_uid"])
        for name, value in feature_vector(row).items():
            vectors[group][name] += float(value)
        vectors[group]["total_events"] += max(0, int(row["n_events"]))
        lumi_uid = str(row.get("lumi_source_uid", "")).strip()
        lumi_key = (group, lumi_uid)
        if lumi_uid and lumi_key not in seen_luminosity_sources:
            lumi = float(row.get("lumi_per_file", 0.0))
            if pd.notna(lumi) and lumi >= 0:
                vectors[group][f"luminosity::{row['domain']}"] += lumi
            seen_luminosity_sources.add(lumi_key)
    return {group: dict(values) for group, values in vectors.items()}


def tie_value(seed: int, group: str, fold: int) -> int:
    text = f"{seed}:{group}:{fold}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:12], 16)


def balance_delta(vector: dict[str, float], fold_values: dict[str, float],
                  targets: dict[str, float]) -> float:
    """Return the change in normalized squared imbalance for one assignment.

    Comparing the complete post-assignment score made nearly empty folds look
    permanently unattractive: every feature absent from such a fold contributed
    a constant penalty.  The marginal score cancels those unchanged terms and
    compares only the imbalance introduced (or removed) by this group.
    """
    score = 0.0
    for name, value in vector.items():
        target = max(targets[name], 1.0)
        before = fold_values.get(name, 0.0)
        after = before + value
        score += ((after - targets[name]) / target) ** 2
        score -= ((before - targets[name]) / target) ** 2
    return score


def assign_groups(catalog: pd.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    vectors = group_vectors(catalog)
    if len(vectors) < n_folds:
        raise ValueError(f"Only {len(vectors)} fold groups are available for {n_folds} folds")
    features = sorted({name for vector in vectors.values() for name in vector})
    totals = {name: sum(vector.get(name, 0.0) for vector in vectors.values()) for name in features}
    targets = {name: totals[name] / n_folds for name in features}

    group_classes = {
        group: {
            name.removeprefix("class::")
            for name, value in vector.items()
            if name.startswith("class::") and value > 0
        }
        for group, vector in vectors.items()
    }
    class_groups = {
        class_name: sorted(
            group for group, classes in group_classes.items()
            if class_name in classes
        )
        for class_name in sorted(set(catalog["class_name"].astype(str)))
    }
    for class_name, groups in class_groups.items():
        if len(groups) < n_folds:
            raise ValueError(
                f"Class {class_name} has only {len(groups)} independent fold groups "
                f"for {n_folds} folds"
            )

    def priority(item):
        group, vector = item
        rarity = sum(
            value / max(totals[name], 1.0)
            for name, value in vector.items() if name != "total_events"
        )
        return (-rarity, -vector.get("total_events", 0.0), group)

    ordered = sorted(vectors.items(), key=priority)
    fold_values = [defaultdict(float) for _ in range(n_folds)]
    fold_classes = [set() for _ in range(n_folds)]
    assignments = {}

    def place(group: str, fold: int) -> None:
        assignments[group] = fold
        fold_classes[fold].update(group_classes[group])
        for name, value in vectors[group].items():
            fold_values[fold][name] += value

    # First satisfy the invariant checked by the validator.  Process classes
    # with the fewest independent groups first, and prefer placements that add
    # coverage for more still-missing classes.
    for class_name in sorted(class_groups, key=lambda name: (len(class_groups[name]), name)):
        while True:
            missing_folds = [
                fold for fold in range(n_folds)
                if class_name not in fold_classes[fold]
            ]
            if not missing_folds:
                break
            candidates = [
                group for group in class_groups[class_name]
                if group not in assignments
            ]
            if len(candidates) < len(missing_folds):
                raise ValueError(
                    f"Cannot give class {class_name} independent coverage in every fold"
                )
            scored = []
            for group in candidates:
                for fold in missing_folds:
                    new_coverage = sum(
                        1.0 / len(class_groups[covered_class])
                        for covered_class in group_classes[group]
                        if covered_class not in fold_classes[fold]
                    )
                    scored.append((
                        -new_coverage,
                        balance_delta(vectors[group], fold_values[fold], targets),
                        tie_value(seed, group, fold),
                        group,
                        fold,
                    ))
            _, _, _, group, fold = min(scored)
            place(group, fold)

    # Assign the remaining groups by marginal balance cost.  Unlike the old
    # full-score comparison, this does not reward already-populated folds just
    # because unrelated features happen to be present there.
    for group, vector in ordered:
        if group in assignments:
            continue
        scored = [
            (balance_delta(vector, fold_values[fold], targets),
             tie_value(seed, group, fold), fold)
            for fold in range(n_folds)
        ]
        chosen = min(scored)[2]
        place(group, chosen)
    return assignments


def build_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, class_name, domain), group in catalog.groupby(
        ["fold", "class_name", "domain"], sort=True, dropna=False
    ):
        rows.append({
            "fold": int(fold),
            "class_name": class_name,
            "domain": domain,
            "dataset_rows": int(len(group)),
            "source_groups": int(group["fold_group_uid"].nunique()),
            "source_files": int(group["source_uid"].nunique()),
            "n_events": int(pd.to_numeric(group["n_events"]).sum()),
            "unique_lumi_sources": int(group["lumi_source_uid"].nunique()),
            "lumi_per_file_sum": float(
                group.drop_duplicates("lumi_source_uid")["lumi_per_file"].astype(float).sum()
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    catalog = pd.read_csv(args.input, keep_default_na=False)
    cv = config["cross_validation"]
    assignments = assign_groups(catalog, int(cv["n_folds"]), int(cv["seed"]))
    catalog["fold"] = catalog["fold_group_uid"].astype(str).map(assignments).astype(int)
    if catalog.groupby("fold_group_uid")["fold"].nunique().max() != 1:
        raise RuntimeError("A source group was assigned to multiple folds")
    output = Path(args.output)
    summary = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output, index=False)
    build_summary(catalog).to_csv(summary, index=False)
    print(f"Assigned {len(assignments)} source groups across {cv['n_folds']} folds")


if __name__ == "__main__":
    main()
