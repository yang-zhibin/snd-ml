import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import yaml


ALLOWED_SPLITS = {"train", "val", "test", "none"}
ALLOWED_USAGES = {"train", "val", "test", "inference", "excluded"}

REQUIRED_METADATA_COLUMNS = {
    "partition_id",
    "region",
    "particle_group",
    "particle_family",
    "particle_id",
    "feature_partition_path",
    "hit3d_partition_path",
    "n_events",
    "lumi_per_partition",
}

ADDED_COLUMNS = [
    "split_version",
    "cv_enabled",
    "cv_strategy",
    "fold",
    "fold_index",
    "n_folds",
    "split",
    "usage",
    "is_train_eligible",
    "is_real_data",
    "class_id",
    "class_name",
    "class_source",
    "split_seed",
    "split_group",
    "split_strategy",
    "input_view",
    "input_view_version",
    "allowed_detType",
    "selected_hit_features",
]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_list(config, *keys):
    value = config
    for key in keys:
        value = value.get(key, {})
    if value in ({}, None):
        return []
    return list(value)


def require_keys(mapping, required, label):
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")


def validate_metadata(df, config):
    missing = sorted(REQUIRED_METADATA_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input metadata is missing required columns: {missing}")

    if config.get("validation", {}).get("require_unique_partition_id", False):
        duplicated = df.loc[df["partition_id"].duplicated(), "partition_id"].tolist()
        if duplicated:
            examples = duplicated[:10]
            raise ValueError(f"Duplicate partition_id values found, examples: {examples}")

    if df.empty:
        raise ValueError("Input metadata has no rows")


def validate_config(config, metadata_columns):
    require_keys(
        config,
        [
            "split_version",
            "seed",
            "split_strategy",
            "fractions",
            "group_by",
            "training_selection",
            "classification",
            "input_view",
            "default_unassigned",
            "real_data_default",
        ],
        "Split config",
    )

    fractions = config["fractions"]
    require_keys(fractions, ["train", "val", "test"], "fractions")
    fraction_sum = sum(float(fractions[name]) for name in ("train", "val", "test"))
    if abs(fraction_sum - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {fraction_sum}")

    cv = config.get("cross_validation", {})
    require_keys(cv, ["enabled", "strategy", "n_folds"], "cross_validation")
    if not cv.get("enabled", False):
        raise ValueError("This workflow expects cross_validation.enabled: true")
    if cv.get("strategy") != "two_fold_cross_fit":
        raise ValueError("Only cross_validation.strategy: two_fold_cross_fit is supported")
    if int(cv.get("n_folds")) != 2:
        raise ValueError("Only n_folds: 2 is supported")
    if config["split_strategy"] != "two_fold_cross_fit_by_partition_rows":
        raise ValueError("split_strategy must be two_fold_cross_fit_by_partition_rows")

    group_by = list(config["group_by"])
    missing_group_columns = [column for column in group_by if column not in metadata_columns]
    if missing_group_columns:
        raise ValueError(f"group_by columns are missing from input metadata: {missing_group_columns}")

    for label in ("default_unassigned", "real_data_default"):
        require_keys(config[label], ["split", "usage"], label)
        validate_split_usage(config[label]["split"], config[label]["usage"], label)

    input_view = config["input_view"]
    require_keys(input_view, ["name", "version", "allowed_detType", "selected_hit_features"], "input_view")
    if not input_view["allowed_detType"]:
        raise ValueError("input_view.allowed_detType must not be empty")
    if not input_view["selected_hit_features"]:
        raise ValueError("input_view.selected_hit_features must not be empty")

    classification = config["classification"]
    require_keys(classification, ["task", "n_classes", "classes"], "classification")
    classes = classification["classes"]
    if int(classification["n_classes"]) != len(classes):
        raise ValueError(
            "classification.n_classes does not match the number of class definitions: "
            f"{classification['n_classes']} != {len(classes)}"
        )
    class_ids = [item["class_id"] for item in classes]
    if len(class_ids) != len(set(class_ids)):
        raise ValueError(f"class_id values must be unique, got {class_ids}")
    for index, class_def in enumerate(classes):
        require_keys(
            class_def,
            ["class_id", "class_name", "class_source", "particle_groups", "regions"],
            f"classification.classes[{index}]",
        )


def validate_split_usage(split, usage, label):
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"{label} has invalid split '{split}'")
    if usage not in ALLOWED_USAGES:
        raise ValueError(f"{label} has invalid usage '{usage}'")
    expected_usage = {"train": "train", "val": "val", "test": None, "none": "excluded"}
    if split in {"train", "val"} and usage != expected_usage[split]:
        raise ValueError(f"{label}: split '{split}' must use usage '{expected_usage[split]}'")
    if split == "none" and usage != "excluded":
        raise ValueError(f"{label}: split 'none' must use usage 'excluded'")


def stable_group_seed(seed, group_key):
    text = f"{seed}::{group_key}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def split_group_value(row, group_by):
    return "|".join(f"{column}={row[column]}" for column in group_by)


def is_real_data_row(row):
    return str(row.get("particle_group", "")) == "real_data" or str(row.get("particle_family", "")) == "real_data"


def is_excluded(row, config):
    excluded = config.get("exclude", {})
    excluded_regions = set(excluded.get("regions", []) or [])
    excluded_particle_groups = set(excluded.get("particle_groups", []) or [])
    return row["region"] in excluded_regions or row["particle_group"] in excluded_particle_groups


def is_train_eligible(row, config):
    selection = config["training_selection"]
    include_regions = set(selection.get("include_regions", []) or [])
    include_particle_groups = set(selection.get("include_particle_groups", []) or [])
    if row["region"] not in include_regions:
        return False
    if row["particle_group"] not in include_particle_groups:
        return False
    if is_real_data_row(row) and not selection.get("allow_real_data_training", False):
        return False
    matches = matching_classes(row, config)
    if len(matches) > 1:
        raise ValueError(
            "Training-selected row matches multiple classes, "
            f"partition_id={row['partition_id']} matches {[item['class_name'] for item in matches]}"
        )
    return len(matches) == 1


def matching_classes(row, config):
    matches = []
    for class_def in config["classification"]["classes"]:
        if row["particle_group"] not in set(class_def.get("particle_groups", []) or []):
            continue
        if row["region"] not in set(class_def.get("regions", []) or []):
            continue
        matches.append(class_def)
    return matches


def class_for_row(row, config, required):
    matches = matching_classes(row, config)
    if required and len(matches) != 1:
        raise ValueError(
            "Train-eligible row must match exactly one class, "
            f"got {len(matches)} for partition_id={row['partition_id']} "
            f"region={row['region']} particle_group={row['particle_group']}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Row matches multiple classes, "
            f"partition_id={row['partition_id']} matches {[item['class_name'] for item in matches]}"
        )
    if not matches:
        return {"class_id": "", "class_name": "", "class_source": ""}
    match = matches[0]
    return {
        "class_id": int(match["class_id"]),
        "class_name": match["class_name"],
        "class_source": match["class_source"],
    }


def has_required_hit3d(row, metadata_columns):
    if "hit3d_available" in metadata_columns:
        return truthy(row.get("hit3d_available", False))
    return str(row.get("hit3d_partition_path", "")).strip() != ""


def make_base_added_columns(row, config, split_group, train_eligible, class_info):
    cv = config["cross_validation"]
    input_view = config["input_view"]
    return {
        "split_version": config["split_version"],
        "cv_enabled": bool(cv.get("enabled", False)),
        "cv_strategy": cv.get("strategy", ""),
        "n_folds": int(cv["n_folds"]),
        "is_train_eligible": bool(train_eligible),
        "is_real_data": bool(is_real_data_row(row)),
        "class_id": class_info["class_id"],
        "class_name": class_info["class_name"],
        "class_source": class_info["class_source"],
        "split_seed": int(config["seed"]),
        "split_group": split_group,
        "split_strategy": config["split_strategy"],
        "input_view": input_view["name"],
        "input_view_version": input_view["version"],
        "allowed_detType": json.dumps(input_view["allowed_detType"]),
        "selected_hit_features": json.dumps(input_view["selected_hit_features"]),
    }


def make_output_row(row, added, fold_index, split, usage):
    validate_split_usage(split, usage, f"partition_id={row['partition_id']} fold_{fold_index}")
    output = row.to_dict()
    output.update(added)
    output.update(
        {
            "fold": f"fold_{fold_index}",
            "fold_index": int(fold_index),
            "split": split,
            "usage": usage,
        }
    )
    return output


def train_val_assignment(pool_ids, val_fraction):
    n_pool = len(pool_ids)
    if n_pool == 0:
        return {}, []
    if n_pool == 1:
        return {pool_ids[0]: "train"}, []

    n_val = int(round(float(val_fraction) * n_pool))
    n_val = max(1, n_val)
    n_val = min(n_val, n_pool - 1)
    val_ids = set(pool_ids[:n_val])
    assignment = {partition_id: ("val" if partition_id in val_ids else "train") for partition_id in pool_ids}
    return assignment, sorted(val_ids)


def assign_train_eligible_group(group_df, config, group_by, metadata_columns):
    allow_empty_groups = bool(config.get("validation", {}).get("allow_empty_groups", True))
    val_fraction = float(config["cross_validation"].get("val_fraction_within_train_pool", 0.20))
    group_key = split_group_value(group_df.iloc[0], group_by)
    shuffled = group_df.sample(frac=1, random_state=stable_group_seed(config["seed"], group_key))
    partition_ids = shuffled["partition_id"].tolist()

    if len(partition_ids) < 2 and not allow_empty_groups:
        raise ValueError(f"Split group '{group_key}' needs at least 2 partitions for two-fold cross-fit")

    half_size = int(math.ceil(len(partition_ids) / 2.0))
    halves = [partition_ids[:half_size], partition_ids[half_size:]]
    if any(len(half) == 0 for half in halves) and not allow_empty_groups:
        raise ValueError(f"Split group '{group_key}' produced an empty fold half")

    fold_assignments = {}
    for fold_index in (0, 1):
        test_ids = set(halves[fold_index])
        pool_ids = halves[1 - fold_index]
        if len(pool_ids) < 2 and not allow_empty_groups:
            raise ValueError(f"Split group '{group_key}' fold_{fold_index} has fewer than 2 train/val partitions")
        train_val, _ = train_val_assignment(pool_ids, val_fraction)
        for partition_id in partition_ids:
            if partition_id in test_ids:
                fold_assignments[(partition_id, fold_index)] = ("test", "test")
            else:
                split = train_val.get(partition_id, "train")
                fold_assignments[(partition_id, fold_index)] = (split, split)

    rows = []
    for _, row in group_df.iterrows():
        if config.get("validation", {}).get("require_hit3d_for_train_eligible", False):
            if not has_required_hit3d(row, metadata_columns):
                raise ValueError(f"Train-eligible row has no required hit3D partition: {row['partition_id']}")
        class_info = class_for_row(
            row,
            config,
            required=config.get("validation", {}).get("require_class_for_train_eligible", True),
        )
        added = make_base_added_columns(row, config, group_key, True, class_info)
        for fold_index in (0, 1):
            split, usage = fold_assignments[(row["partition_id"], fold_index)]
            rows.append(make_output_row(row, added, fold_index, split, usage))
    return rows


def assign_non_train_row(row, config, group_by, excluded):
    group_key = split_group_value(row, group_by)
    if excluded:
        class_info = {"class_id": "", "class_name": "", "class_source": ""}
        split = "none"
        usage = "excluded"
    else:
        class_info = class_for_row(row, config, required=False)
        defaults = config["real_data_default"] if is_real_data_row(row) else config["default_unassigned"]
        split = defaults["split"]
        usage = defaults["usage"]

    added = make_base_added_columns(row, config, group_key, False, class_info)
    return [make_output_row(row, added, fold_index, split, usage) for fold_index in (0, 1)]


def build_split_dataframe(metadata_df, config):
    validate_metadata(metadata_df, config)
    validate_config(config, metadata_df.columns)

    group_by = list(config["group_by"])
    work = metadata_df.copy()
    work["__excluded"] = work.apply(lambda row: is_excluded(row, config), axis=1)
    work["__train_eligible"] = work.apply(
        lambda row: False if row["__excluded"] else is_train_eligible(row, config),
        axis=1,
    )

    output_rows = []
    train_eligible = work[work["__train_eligible"]]
    non_train = work[~work["__train_eligible"]]

    for _, group_df in train_eligible.groupby(group_by, sort=False):
        output_rows.extend(assign_train_eligible_group(group_df, config, group_by, metadata_df.columns))

    for _, row in non_train.iterrows():
        output_rows.extend(assign_non_train_row(row, config, group_by, bool(row["__excluded"])))

    output_df = pd.DataFrame(output_rows)
    output_df = output_df[[column for column in metadata_df.columns] + ADDED_COLUMNS]
    validate_output(output_df, config)
    return output_df


def validate_output(df, config):
    if df.empty:
        raise ValueError("Split output has no rows")

    if df[["partition_id", "fold"]].duplicated().any():
        duplicated = df.loc[df[["partition_id", "fold"]].duplicated(), ["partition_id", "fold"]]
        raise ValueError(f"Output has duplicate partition_id + fold rows, examples: {duplicated.head().to_dict('records')}")

    if df["split"].isna().any() or (df["split"].astype(str).str.strip() == "").any():
        raise ValueError("Output contains blank split values")
    if df["usage"].isna().any() or (df["usage"].astype(str).str.strip() == "").any():
        raise ValueError("Output contains blank usage values")
    if not set(df["split"]).issubset(ALLOWED_SPLITS):
        raise ValueError(f"Output contains invalid split values: {sorted(set(df['split']) - ALLOWED_SPLITS)}")
    if not set(df["usage"]).issubset(ALLOWED_USAGES):
        raise ValueError(f"Output contains invalid usage values: {sorted(set(df['usage']) - ALLOWED_USAGES)}")

    if config.get("validation", {}).get("require_input_view_for_usable", False):
        usable = df["usage"] != "excluded"
        for column in ("input_view", "input_view_version", "allowed_detType", "selected_hit_features"):
            if df.loc[usable, column].isna().any() or (df.loc[usable, column].astype(str).str.strip() == "").any():
                raise ValueError(f"Usable output rows have blank {column}")

    if config.get("validation", {}).get("require_all_rows_assigned", False):
        bad = df.loc[df["usage"] != "excluded"]
        if bad["split"].eq("none").any():
            raise ValueError("Usable output rows must not have split == none")

    train_eligible = df[df["is_train_eligible"].astype(bool)]
    if not train_eligible.empty:
        counts = train_eligible.groupby("partition_id")["split"].agg(
            test_count=lambda values: int((values == "test").sum()),
            train_val_count=lambda values: int(values.isin(["train", "val"]).sum()),
        )
        bad = counts[(counts["test_count"] != 1) | (counts["train_val_count"] != 1)]
        if not bad.empty:
            raise ValueError(
                "Train-eligible partitions must be test in exactly one fold and train/val in exactly one fold, "
                f"examples: {bad.head().to_dict('index')}"
            )
        required_class = config.get("validation", {}).get("require_class_for_train_eligible", True)
        if required_class:
            if train_eligible["class_name"].astype(str).str.strip().eq("").any():
                raise ValueError("Train-eligible rows have blank class_name")


def add_summary(summary_rows, df, summary_type, group_columns):
    numeric_df = df.copy()
    numeric_df["n_events"] = pd.to_numeric(numeric_df["n_events"], errors="coerce").fillna(0)
    numeric_df["lumi_per_partition"] = pd.to_numeric(
        numeric_df["lumi_per_partition"],
        errors="coerce",
    ).fillna(0.0)
    grouped = numeric_df.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), numeric_df)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {
            "summary_type": summary_type,
            "fold": "ALL",
            "split": "ALL",
            "usage": "ALL",
            "class_name": "ALL",
            "region": "ALL",
            "particle_group": "ALL",
            "is_train_eligible": "ALL",
            "is_real_data": "ALL",
            "n_rows": int(len(group)),
            "n_events": int(group["n_events"].sum()),
            "lumi_per_partition": float(group["lumi_per_partition"].sum()),
        }
        for column, value in zip(group_columns, key):
            row[column] = value
        summary_rows.append(row)


def build_summary_dataframe(split_df):
    summary_rows = []
    add_summary(summary_rows, split_df, "overall", [])
    add_summary(summary_rows, split_df, "by_fold_split_usage", ["fold", "split", "usage"])
    add_summary(summary_rows, split_df, "by_fold_class_split", ["fold", "class_name", "split"])
    add_summary(
        summary_rows,
        split_df,
        "by_fold_region_particle_split",
        ["fold", "region", "particle_group", "split"],
    )
    add_summary(summary_rows, split_df, "by_usage", ["usage"])
    add_summary(summary_rows, split_df, "by_class", ["class_name"])
    add_summary(summary_rows, split_df, "by_train_eligible", ["is_train_eligible", "split", "usage"])

    columns = [
        "summary_type",
        "fold",
        "split",
        "usage",
        "class_name",
        "region",
        "particle_group",
        "is_train_eligible",
        "is_real_data",
        "n_rows",
        "n_events",
        "lumi_per_partition",
    ]
    return pd.DataFrame(summary_rows, columns=columns)


def resolve_output_paths(args, config):
    output_config = config.get("output", {})
    output = args.output or output_config.get("metadata_csv")
    summary = args.summary or output_config.get("summary_csv")
    if not output:
        raise ValueError("No output CSV was provided and config.output.metadata_csv is missing")
    if not summary:
        raise ValueError("No summary CSV was provided and config.output.summary_csv is missing")
    return Path(output), Path(summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create fold-aware split metadata from region/particle partition metadata.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input combined region partition metadata CSV",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Split YAML config",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output split metadata CSV. Defaults to config.output.metadata_csv",
    )
    parser.add_argument(
        "-s",
        "--summary",
        default=None,
        help="Output summary CSV. Defaults to config.output.summary_csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    metadata_df = pd.read_csv(args.input)
    output_path, summary_path = resolve_output_paths(args, config)

    split_df = build_split_dataframe(metadata_df, config)
    summary_df = build_summary_dataframe(split_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote split metadata: {output_path} ({len(split_df)} rows)")
    print(f"Wrote split summary: {summary_path} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
