#!/usr/bin/env python3
"""Build and validate reusable testbeam ROOT histogram bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import yaml

from unified_feature_distributions import (
    _numeric_exposure,
    combine_selection,
    expand_group_series,
    histogram_model,
    load_validated_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str, relative_to: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    return repo_candidate if repo_candidate.exists() else relative_to / candidate


def load_bundle_config(path: str | Path) -> tuple[dict, dict]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path = _resolve(str(config["base_distribution_config"]), config_path.parent)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    expand_group_series(base)
    config["_config_path"] = str(config_path)
    config["_base_config_path"] = str(base_path)
    return config, base


def config_hash(config: dict, base: dict) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = {"histogram_config": clean, "groups": base.get("groups", {})}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _group_rows(args: argparse.Namespace, config: dict, base: dict):
    if args.group not in base["groups"]:
        raise KeyError(f"unknown group {args.group}")
    group = base["groups"][args.group]
    rows = load_validated_rows(
        args.manifest,
        str(group["family"]),
        group.get("source_filters", {}),
        allow_missing_validated_sources=args.allow_missing_validated_sources,
    )
    subfolders = [str(value) for value in group.get("source_subfolders", [])]
    if subfolders:
        if "subfolder" not in rows:
            raise KeyError(f"manifest has no subfolder column required by {args.group}")
        rows = rows.loc[rows["subfolder"].astype(str).isin(subfolders)].copy()
        if rows.empty:
            raise ValueError(f"no sources match subfolders for {args.group}")
    return group, rows.reset_index(drop=True)


def _geometry(rows, group_name: str, expected: str) -> tuple[str, str]:
    ids = (
        sorted(set(rows["source_geometry_id"].astype(str)))
        if "source_geometry_id" in rows else [""]
    )
    checksums = (
        sorted(set(rows["source_geometry_checksum"].astype(str)))
        if "source_geometry_checksum" in rows else [""]
    )
    if len(ids) != 1 or len(checksums) != 1:
        raise ValueError(f"group {group_name} mixes geometry metadata")
    if expected and ids != [expected]:
        raise ValueError(f"group {group_name} geometry {ids[0]} != {expected}")
    return ids[0], checksums[0]


def _expressions(config: dict, base_selection: str, group_expression: str) -> list[str]:
    values = [base_selection, group_expression]
    for name, settings in config["features"].items():
        values.extend([settings.get("expression", name), settings.get("feature_cut", "")])
    for settings in config["features2d"].values():
        values.extend([
            settings["x_expression"], settings["y_expression"],
            settings.get("feature_cut", ""),
        ])
    return [str(value) for value in values if str(value).strip()]


def required_branches(config: dict, group: dict) -> set[str]:
    functions = {"abs", "sqrt", "log", "exp", "pow", "min", "max", "true", "false"}
    words = set()
    for expression in _expressions(
        config, str(config["selection"]["expression"]), str(group.get("expression", ""))
    ):
        words.update(re.findall(r"[A-Za-z_]\w*", expression))
    return {word for word in words if word not in functions}


def audit(args: argparse.Namespace) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    config, base = load_bundle_config(args.config)
    group, rows = _group_rows(args, config, base)
    tree_name = str(config.get("tree_name", "sndData"))
    first = str(rows.iloc[0]["feature_scifi_cluster_v1_path"])
    source = ROOT.TFile.Open(first)
    if not source or source.IsZombie():
        raise RuntimeError(f"cannot open representative feature file {first}")
    tree = source.Get(tree_name)
    if not tree:
        raise RuntimeError(f"missing tree {tree_name} in {first}")
    present = {branch.GetName() for branch in tree.GetListOfBranches()}
    required = required_branches(config, group)
    missing = sorted(required - present)
    source.Close()
    if missing:
        raise RuntimeError(f"missing required feature branches: {missing}")
    record = {
        "audit_passed": True,
        "group": args.group,
        "representative_file": first,
        "required_branches": sorted(required),
        "config_sha256": config_hash(config, base),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flow_1d(histogram) -> dict:
    under = float(histogram.GetBinContent(0))
    inside = float(histogram.Integral(1, histogram.GetNbinsX()))
    over = float(histogram.GetBinContent(histogram.GetNbinsX() + 1))
    return {"underflow": under, "in_range": inside, "overflow": over, "all": under + inside + over}


def _write_json(root_file, name: str, payload: dict) -> None:
    import ROOT

    root_file.cd()
    ROOT.TObjString(json.dumps(payload, sort_keys=True)).Write(name)


def build(args: argparse.Namespace) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.SetDefaultSumw2(True)
    ROOT.TH1.AddDirectory(False)
    config, base = load_bundle_config(args.config)
    group, rows = _group_rows(args, config, base)
    geometry_id, geometry_checksum = _geometry(
        rows, args.group, str(group.get("geometry_setup_id", ""))
    )
    chain = ROOT.TChain(str(config.get("tree_name", "sndData")))
    for path in rows["feature_scifi_cluster_v1_path"].astype(str):
        if chain.Add(path) != 1:
            raise RuntimeError(f"could not add validated feature file {path}")
    dataframe = ROOT.RDataFrame(chain)
    base_selection = combine_selection(
        str(config["selection"]["expression"]), str(group.get("expression", ""))
    )
    if base_selection:
        dataframe = dataframe.Filter(base_selection, "testbeam_bundle_base_selection")
    available_action = dataframe.Count()
    configured_cap = int(group.get("max_events", 0) or 0)
    requested_cap = int(args.max_events or configured_cap)
    if requested_cap < 0:
        raise ValueError("max_events cannot be negative")
    if requested_cap:
        dataframe = dataframe.Range(requested_cap)
    sampled_action = dataframe.Count()

    actions = {}
    counts = {}
    for name, settings in config["features"].items():
        view = dataframe
        cut = str(settings.get("feature_cut", "")).strip()
        if cut:
            view = view.Filter(cut, f"cut_1d_{name}")
        bins, low, high = histogram_model(name, settings)
        counts[("features", name)] = view.Count()
        actions[("features", name)] = view.Histo1D(
            (name, str(settings["axis_title"]), bins, low, high),
            str(settings.get("expression", name)),
        )
    for name, settings in config["features2d"].items():
        view = dataframe
        cut = str(settings.get("feature_cut", "")).strip()
        if cut:
            view = view.Filter(cut, f"cut_2d_{name}")
        counts[("features2d", name)] = view.Count()
        actions[("features2d", name)] = view.Histo2D(
            (
                name, f";{settings['x_title']};{settings['y_title']}",
                int(settings["x_bins"]), float(settings["x_min"]), float(settings["x_max"]),
                int(settings["y_bins"]), float(settings["y_min"]), float(settings["y_max"]),
            ),
            str(settings["x_expression"]), str(settings["y_expression"]),
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    target = ROOT.TFile(str(output), "RECREATE")
    if not target or target.IsZombie():
        raise RuntimeError(f"could not create {output}")
    metadata_hists = {}
    for directory in ("features", "features2d"):
        target.mkdir(directory)
    for (directory, name), action in actions.items():
        histogram = action.GetValue().Clone(name)
        histogram.SetDirectory(0)
        target.cd(directory)
        histogram.Write(name)
        selected = int(counts[(directory, name)].GetValue())
        if directory == "features":
            flow = _flow_1d(histogram)
            if not math.isclose(flow["all"], selected, abs_tol=1e-7):
                raise RuntimeError(f"flow closure failed for {name}: {flow['all']} != {selected}")
            metadata_hists[name] = {"dimension": 1, "selected_entries": selected, **flow}
        else:
            metadata_hists[name] = {
                "dimension": 2,
                "selected_entries": selected,
                "in_range": float(histogram.Integral()),
            }
    available = int(available_action.GetValue())
    sampled_entries = int(sampled_action.GetValue())
    exposure_kinds = sorted(set(rows["exposure_kind"].astype(str)))
    if len(exposure_kinds) != 1:
        raise ValueError(f"mixed exposure kinds: {exposure_kinds}")
    metadata = {
        "bundle_version": str(config["version"]),
        "config_sha256": config_hash(config, base),
        "group": args.group,
        "family": str(group["family"]),
        "particle_charge": str(group.get("particle_charge", "")),
        "beam_energy_gev": group.get("beam_energy_gev", ""),
        "geometry_setup_id": geometry_id,
        "geometry_checksum": geometry_checksum,
        "base_selection": base_selection,
        "sampling_policy": str(group.get("sampling_policy", "all_selected_events")),
        "configured_event_cap": configured_cap,
        "effective_event_cap": requested_cap,
        "available_selected_entries": available,
        "sampled_entries": sampled_entries,
        "source_count": int(rows["source_uid"].nunique()),
        "feature_file_count": int(len(rows)),
        "omitted_missing_source_count": int(rows.attrs.get("omitted_missing_source_count", 0)),
        "sample_exposure": _numeric_exposure(rows),
        "exposure_kind": exposure_kinds[0],
        "histograms": metadata_hists,
        "histogram_config": {key: value for key, value in config.items() if not key.startswith("_")},
    }
    target.mkdir("metadata")
    target.cd("metadata")
    _write_json(target.GetDirectory("metadata"), "bundle_json", metadata)
    target.Close()
    print(f"wrote {output}: group={args.group}, sampled={sampled_entries}, 1D={len(config['features'])}, 2D={len(config['features2d'])}")


def _read_metadata(source) -> dict:
    obj = source.Get("metadata/bundle_json")
    if not obj:
        raise RuntimeError("missing metadata/bundle_json")
    return json.loads(str(obj.GetString().Data()))


def validate(args: argparse.Namespace) -> None:
    import ROOT

    config, base = load_bundle_config(args.config)
    source = ROOT.TFile.Open(args.bundle)
    if not source or source.IsZombie():
        raise RuntimeError(f"cannot open bundle {args.bundle}")
    metadata = _read_metadata(source)
    if metadata.get("group") != args.group:
        raise RuntimeError("bundle group mismatch")
    if metadata.get("config_sha256") != config_hash(config, base):
        raise RuntimeError("bundle histogram configuration hash mismatch")
    checked = []
    for name, settings in config["features"].items():
        histogram = source.Get(f"features/{name}")
        if not histogram or histogram.GetDimension() != 1:
            raise RuntimeError(f"missing or invalid TH1 features/{name}")
        bins, low, high = histogram_model(name, settings)
        if histogram.GetNbinsX() != bins or not math.isclose(histogram.GetXaxis().GetXmin(), low) or not math.isclose(histogram.GetXaxis().GetXmax(), high):
            raise RuntimeError(f"binning mismatch for {name}")
        if any(
            not math.isfinite(float(histogram.GetBinContent(index)))
            for index in range(0, histogram.GetNbinsX() + 2)
        ):
            raise RuntimeError(f"non-finite bin content for {name}")
        checked.append(name)
    for name, settings in config["features2d"].items():
        histogram = source.Get(f"features2d/{name}")
        if not histogram or histogram.GetDimension() != 2:
            raise RuntimeError(f"missing or invalid TH2 features2d/{name}")
        if histogram.GetNbinsX() != int(settings["x_bins"]) or histogram.GetNbinsY() != int(settings["y_bins"]):
            raise RuntimeError(f"binning mismatch for {name}")
        if not all((
            math.isclose(histogram.GetXaxis().GetXmin(), float(settings["x_min"])),
            math.isclose(histogram.GetXaxis().GetXmax(), float(settings["x_max"])),
            math.isclose(histogram.GetYaxis().GetXmin(), float(settings["y_min"])),
            math.isclose(histogram.GetYaxis().GetXmax(), float(settings["y_max"])),
        )):
            raise RuntimeError(f"axis-range mismatch for {name}")
        if any(
            not math.isfinite(float(histogram.GetBinContent(ix, iy)))
            for ix in range(0, histogram.GetNbinsX() + 2)
            for iy in range(0, histogram.GetNbinsY() + 2)
        ):
            raise RuntimeError(f"non-finite bin content for {name}")
        checked.append(name)
    source.Close()
    record = {
        "validation_passed": True,
        "bundle": str(args.bundle),
        "group": args.group,
        "config_sha256": config_hash(config, base),
        "histogram_count": len(checked),
        "histograms": checked,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"validated {args.bundle}: {len(checked)} histograms")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    for name, function in (("audit", audit), ("build", build)):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--group", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--allow-missing-validated-sources", action="store_true")
        if name == "build":
            command.add_argument("--max-events", type=int, default=0)
        command.set_defaults(function=function)
    command = sub.add_parser("validate")
    command.add_argument("--config", required=True)
    command.add_argument("--bundle", required=True)
    command.add_argument("--group", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=validate)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
