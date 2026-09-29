#!/usr/bin/env python3
"""Combine prepared physics sources and V7 muon splits into a logical catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from data_driven_3d_dataset import load_yaml, stable_uid, truthy


OUTPUT_COLUMNS = [
    "dataset_version", "dataset_file_uid", "source_uid", "fold_group_uid",
    "lumi_source_uid", "dataset_key", "class_id", "class_name", "class_source",
    "subclass", "domain", "run", "production", "energy_bin", "feature_path",
    "hit3d_path", "tree_name", "event_filter", "n_events", "lumi_per_file",
    "selection_version", "density_counts",
]


def root_count(path: str, tree_name: str, expression: str = "") -> int:
    import ROOT
    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open ROOT file {path}")
    try:
        tree = root_file.Get(tree_name)
        if not tree:
            raise KeyError(f"Tree {tree_name!r} is missing from {path}")
        return int(tree.GetEntries(str(expression))) if expression else int(tree.GetEntries())
    finally:
        root_file.Close()


def class_info(config: dict, name: str) -> dict:
    if name not in config["classes"]:
        raise ValueError(f"Unknown class {name!r}")
    return config["classes"][name]


def dataset_row(source, config, class_name, event_filter, n_events, subclass="",
                feature_path=None, hit3d_path=None, selection_version="",
                density_counts=""):
    info = class_info(config, class_name)
    actual_feature = str(feature_path or source["feature_path"])
    actual_hit = str(hit3d_path or source["hit3d_path"])
    dataset_file_uid = stable_uid(
        source["source_uid"], class_name, actual_hit, event_filter, prefix="dsf"
    )
    return {
        "dataset_version": config["dataset_version"],
        "dataset_file_uid": dataset_file_uid,
        "source_uid": source["source_uid"],
        "fold_group_uid": source["fold_group_uid"],
        "lumi_source_uid": source["lumi_source_uid"],
        "dataset_key": source["dataset_key"],
        "class_id": int(info["class_id"]),
        "class_name": class_name,
        "class_source": str(info["class_source"]),
        "subclass": str(subclass),
        "domain": source["domain"],
        "run": source["run"],
        "production": source["production"],
        "energy_bin": source["energy_bin"],
        "feature_path": actual_feature,
        "hit3d_path": actual_hit,
        "tree_name": config["trees"]["hit3d"],
        "event_filter": str(event_filter),
        "n_events": int(n_events),
        "lumi_per_file": float(source["lumi_per_file"]),
        "selection_version": str(selection_version),
        "density_counts": str(density_counts),
    }


def prepared_rows(source, config: dict, inspect_root: bool) -> list[dict]:
    mode = str(source["class_mode"])
    tree_name = str(config["trees"]["hit3d"])
    if mode == "direct":
        class_name = str(source["direct_class"])
        count = (
            root_count(source["hit3d_path"], tree_name)
            if inspect_root else int(source["n_event_metadata"])
        )
        return [dataset_row(
            source, config, class_name, "true", count,
            subclass=source["energy_bin"],
        )] if count else []
    if mode != "pdg_map":
        raise ValueError(f"Prepared source has unsupported class_mode={mode!r}")
    mapping = json.loads(str(source["pdg_class_map"]))
    class_codes = {}
    for code, class_name in mapping.items():
        class_codes.setdefault(str(class_name), []).append(int(code))
    result = []
    for class_name, codes in sorted(class_codes.items()):
        expression = " || ".join(f"pdgCode == {code}" for code in sorted(codes))
        count = root_count(source["hit3d_path"], tree_name, expression) if inspect_root else -1
        if count == 0:
            continue
        subclass = source["dataset_key"] if class_name == "NC" else ""
        result.append(dataset_row(
            source, config, class_name, expression, count, subclass=subclass,
        ))
    return result


def load_audits(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, keep_default_na=False)
        if frame.empty:
            raise ValueError(f"Muon audit {path} contains no rows")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_catalog(source_catalog: pd.DataFrame, audits: pd.DataFrame, config: dict,
                  neutrino_audits: pd.DataFrame | None = None,
                  inspect_root: bool = True) -> pd.DataFrame:
    source_catalog = source_catalog.copy()
    source_catalog["source_uid"] = source_catalog["source_uid"].astype(str)
    source_lookup = source_catalog.set_index("source_uid", drop=False)
    rows = []
    available = source_catalog["input_available"].map(truthy)
    prepared = source_catalog.loc[
        available & source_catalog["role"].eq("prepared_physics")
    ]
    prepared_sources = prepared.iterrows()
    if inspect_root:
        prepared_sources = tqdm(
            prepared_sources,
            total=len(prepared),
            desc="Counting prepared Hit3D events",
            unit="file",
            mininterval=1.0,
            dynamic_ncols=True,
        )
    for _, source in prepared_sources:
        rows.extend(prepared_rows(source, config, inspect_root))

    if neutrino_audits is not None and not neutrino_audits.empty:
        if neutrino_audits["source_uid"].astype(str).duplicated().any():
            raise ValueError("Neutrino audits contain duplicate source_uid rows")
        for _, audit in neutrino_audits.iterrows():
            source_uid = str(audit["source_uid"])
            if source_uid not in source_lookup.index:
                raise ValueError(f"Neutrino audit has unknown source_uid {source_uid}")
            source = source_lookup.loc[source_uid]
            partitions = json.loads(str(audit["partitions"]))
            for class_name, partition in partitions.items():
                count = int(partition["entries"])
                if count == 0:
                    continue
                rows.append(dataset_row(
                    source, config, class_name, "true", count,
                    subclass=source["dataset_key"] if class_name == "NC" else "",
                    feature_path=partition["feature_path"],
                    hit3d_path=partition["hit3d_path"],
                    selection_version=audit["partition_version"],
                ))

    if not audits.empty:
        if audits["source_uid"].duplicated().any():
            raise ValueError("Muon audits contain duplicate source_uid rows")
        for _, audit in audits.iterrows():
            source_uid = str(audit["source_uid"])
            if source_uid not in source_lookup.index:
                raise ValueError(f"Muon audit has unknown source_uid {source_uid}")
            source = source_lookup.loc[source_uid]
            for prefix in ("pass", "fail"):
                count = int(audit[f"{prefix}_entries"])
                if count == 0:
                    continue
                rows.append(dataset_row(
                    source,
                    config,
                    str(audit[f"{prefix}_class"]),
                    "true",
                    count,
                    subclass=f"{source['domain']}_{prefix}",
                    feature_path=audit[f"{prefix}_feature_path"],
                    hit3d_path=audit[f"{prefix}_hit3d_path"],
                    selection_version=audit["selection_fingerprint"],
                    density_counts=audit[f"{prefix}_density_counts"],
                ))
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if result.empty:
        raise ValueError("Dataset catalog contains no usable rows")
    if result["dataset_file_uid"].duplicated().any():
        raise ValueError("Dataset catalog contains duplicate dataset_file_uid values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-catalog", required=True)
    parser.add_argument("--muon-audits", nargs="*", default=[])
    parser.add_argument("--neutrino-audits", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-root-counts", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    sources = pd.read_csv(args.source_catalog, keep_default_na=False)
    audits = load_audits(args.muon_audits)
    neutrino_audits = load_audits(args.neutrino_audits)
    result = build_catalog(
        sources, audits, config, neutrino_audits, not args.skip_root_counts
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} logical dataset files to {output}")


if __name__ == "__main__":
    main()
