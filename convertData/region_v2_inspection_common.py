"""Side-effect-free configuration helpers for region-v2 inspection inputs."""
import math
import os
from pathlib import Path
import pandas as pd
import yaml

MANIFEST_FIELDS = ["inspection_version", "dataset_name", "data_type", "output_path", "tree_name", "source_metadata_csv", "source_row_definition", "selected_file_count", "missing_file_count", "selected_runs", "event_selection", "particle_filter", "source_luminosity_fb", "target_luminosity_fb", "normalization_weight", "input_entry_count", "output_entry_count", "mc_fraction", "production_timestamp_utc"]
AUDIT_FIELDS = ["dataset_name", "source_metadata_row", "resolved_feature_path", "partition", "run", "lumi_per_file", "file_exists", "file_opened", "sndData_entries", "cutFlowSummary_entries"]

def repo_root():
    return Path(__file__).resolve().parents[1]

def absolute_from_repo(path):
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path

def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not config.get("version") or not config.get("datasets"):
        raise ValueError(f"Invalid inspection configuration: {path}")
    return config

def resolve_feature_path(row):
    value = str(row["feature_path"])
    if value.startswith("/") or value.startswith("root://"):
        return value
    return f"{str(row['output_base_path']).rstrip('/')}/{value.lstrip('/')}"

def load_selected_rows(config, dataset_name):
    if dataset_name not in config["datasets"]:
        raise KeyError(f"Unknown dataset {dataset_name!r}")
    spec = config["datasets"][dataset_name]
    metadata_path = absolute_from_repo(spec["metadata_csv"])
    frame = pd.read_csv(metadata_path, keep_default_na=False)
    required = {"feature_path", "output_base_path", "lumi_per_file"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{metadata_path} lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["source_metadata_row"] = frame.index.astype(int)
    if "metadata_filter" in spec:
        selected = frame
        for column, expected in spec["metadata_filter"].items():
            if column not in selected.columns:
                raise ValueError(f"{metadata_path} lacks filter column {column!r}")
            selected = selected[selected[column].astype(str) == str(expected)]
    else:
        choice = spec.get("row_selection", {})
        if choice.get("method") != "first_fraction" or choice.get("rounding") != "ceil":
            raise ValueError(f"Unsupported row selection for {dataset_name}: {choice}")
        selected = frame.iloc[:int(math.ceil(float(choice["fraction"]) * len(frame)))]
    if selected.empty:
        raise ValueError(f"Metadata selection for {dataset_name} is empty")
    selected = selected.copy()
    selected["resolved_feature_path"] = selected.apply(resolve_feature_path, axis=1)
    return spec, metadata_path, selected

def source_luminosity(selected):
    values = pd.to_numeric(selected["lumi_per_file"], errors="raise")
    if (values < 0).any():
        raise ValueError("Negative lumi_per_file encountered")
    return float(values.sum())

def target_luminosity(config):
    _, _, selected = load_selected_rows(config, config["target_luminosity"]["source_dataset"])
    value = source_luminosity(selected)
    expected = float(config["target_luminosity"]["expected_fb"])
    tolerance = float(config["target_luminosity"].get("tolerance_fb", 1e-9))
    if abs(value - expected) > tolerance:
        raise ValueError(f"Target luminosity {value:.12g} differs from expected {expected:.12g}")
    return value

def output_path(config, dataset_name):
    spec = config["datasets"][dataset_name]
    return os.path.join(config["output"]["root_dir"], config["version"], spec["output_file"])
