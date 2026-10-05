#!/usr/bin/env python3
"""Build one grouped 2D histogram and its profile for a partition group."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import ROOT

from make_group_hist import (
    build_chain,
    build_selection,
    load_yaml,
    resolve_path,
    select_group_rows,
    selection_needs_friend_tree,
    sum_numeric_column,
    write_metadata,
)


ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)


def get_pair_config(hist_cfg: dict, pair: str) -> dict:
    try:
        return dict(hist_cfg["features2d"][pair])
    except KeyError as exc:
        raise KeyError(f"2D pair {pair!r} is not defined") from exc


def define_expression(dataframe, name: str, expression: str):
    expression = expression.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression):
        return dataframe, expression
    return dataframe.Define(name, expression), name


def make_histogram(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    metadata_csv = resolve_path(args.metadata_csv, repo_root)
    hist_cfg = load_yaml(resolve_path(args.hist_config, repo_root))
    eval_cfg = load_yaml(resolve_path(args.eval_config, repo_root))
    pair_cfg = get_pair_config(hist_cfg, args.pair)
    rows = select_group_rows(metadata_csv, args.region, args.particle_group)
    representative = rows.iloc[0].to_dict()

    selection, applied_cuts = build_selection(
        representative, pair_cfg, eval_cfg, args.base_cut_key, args.extra_cut_key
    )
    tree_name = eval_cfg.get("metadata", {}).get("tree_name", "sndData")
    friend_tree = eval_cfg.get("metadata", {}).get("friend_tree_name")
    chain, added_paths, skipped_paths = build_chain(
        rows,
        tree_name,
        repo_root,
        friend_tree_name=friend_tree if selection_needs_friend_tree(applied_cuts) else None,
    )
    dataframe = ROOT.RDataFrame(chain)
    if selection:
        dataframe = dataframe.Filter(selection, "em_shower_2d_selection")
    dataframe, x_column = define_expression(
        dataframe, "__eval_feature_x", str(pair_cfg["x_expression"])
    )
    dataframe, y_column = define_expression(
        dataframe, "__eval_feature_y", str(pair_cfg["y_expression"])
    )

    model = ROOT.RDF.TH2DModel(
        f"h2_{args.pair}",
        "",
        int(pair_cfg["x_bins"]),
        float(pair_cfg["x_min"]),
        float(pair_cfg["x_max"]),
        int(pair_cfg["y_bins"]),
        float(pair_cfg["y_min"]),
        float(pair_cfg["y_max"]),
    )
    selected = dataframe.Count()
    result = dataframe.Histo2D(model, x_column, y_column)
    selected_entries = int(selected.GetValue())
    hist = result.GetValue().Clone("hist2d")
    hist.SetDirectory(0)
    hist.GetXaxis().SetTitle(str(pair_cfg["x_title"]))
    hist.GetYaxis().SetTitle(str(pair_cfg["y_title"]))
    profile = hist.ProfileX("profile_x")
    profile.SetDirectory(0)
    profile.GetXaxis().SetTitle(str(pair_cfg["x_title"]))
    profile.GetYaxis().SetTitle(f"Mean {pair_cfg['y_title']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output), "RECREATE")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not create {output}")
    hist.Write("hist2d")
    profile.Write("profile_x")
    write_metadata(
        root_file,
        {
            "region": args.region,
            "particle_group": args.particle_group,
            "pair": args.pair,
            "x_expression": pair_cfg["x_expression"],
            "y_expression": pair_cfg["y_expression"],
            "selection": selection,
            "selected_entries": selected_entries,
            "n_added_files": len(added_paths),
            "n_skipped_files": len(skipped_paths),
            "sum_n_events": int(sum_numeric_column(rows, "n_events")),
            "lumi": sum_numeric_column(rows, "lumi_per_partition"),
        },
    )
    root_file.Close()
    print(f"Wrote {args.pair} for {args.region}/{args.particle_group}: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--hist-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--particle-group", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--base-cut-key", default="none")
    parser.add_argument("--extra-cut-key", default="none")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    make_histogram(parser.parse_args())


if __name__ == "__main__":
    main()
