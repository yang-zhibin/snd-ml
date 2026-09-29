#!/usr/bin/env python3
"""Fill passing-muon reconstructed-track inspection histograms."""

import argparse
from pathlib import Path

import pandas as pd
import ROOT

from reco_track_features import TRACK_TYPE_PREFIXES


CHI2NDF_BINS = 150
CHI2NDF_MIN = 0.0
CHI2NDF_MAX = 300.0


def _add_bin(histogram, target, source):
    """Add one bin, including its statistical uncertainty, then clear source."""
    content = histogram.GetBinContent(target) + histogram.GetBinContent(source)
    error = (
        histogram.GetBinError(target) ** 2 + histogram.GetBinError(source) ** 2
    ) ** 0.5
    histogram.SetBinContent(target, content)
    histogram.SetBinError(target, error)
    histogram.SetBinContent(source, 0.0)
    histogram.SetBinError(source, 0.0)


def fold_flow_bins(histogram):
    """Fold visible-axis underflows/overflows into the edge bins.

    ROOT does not draw flow bins. This inspection must not silently hide
    high-chi2, high-multiplicity, or out-of-range timing populations.
    """
    if histogram.GetDimension() == 1:
        _add_bin(histogram, 1, 0)
        _add_bin(histogram, histogram.GetNbinsX(), histogram.GetNbinsX() + 1)
        return
    if histogram.GetDimension() not in (2, 3):
        raise ValueError(f"Unsupported histogram dimension: {histogram.GetDimension()}")

    nx, ny = histogram.GetNbinsX(), histogram.GetNbinsY()
    nz = histogram.GetNbinsZ() if histogram.GetDimension() == 3 else 0
    if histogram.GetDimension() == 3:
        # Fold one axis at a time. Including the other axes' flow bins in the
        # earlier folds moves every edge/corner entry exactly once.
        for iy in range(0, ny + 2):
            for iz in range(0, nz + 2):
                _add_bin(
                    histogram,
                    histogram.GetBin(1, iy, iz),
                    histogram.GetBin(0, iy, iz),
                )
                _add_bin(
                    histogram,
                    histogram.GetBin(nx, iy, iz),
                    histogram.GetBin(nx + 1, iy, iz),
                )
        for ix in range(1, nx + 1):
            for iz in range(0, nz + 2):
                _add_bin(
                    histogram,
                    histogram.GetBin(ix, 1, iz),
                    histogram.GetBin(ix, 0, iz),
                )
                _add_bin(
                    histogram,
                    histogram.GetBin(ix, ny, iz),
                    histogram.GetBin(ix, ny + 1, iz),
                )
        for ix in range(1, nx + 1):
            for iy in range(1, ny + 1):
                _add_bin(
                    histogram,
                    histogram.GetBin(ix, iy, 1),
                    histogram.GetBin(ix, iy, 0),
                )
                _add_bin(
                    histogram,
                    histogram.GetBin(ix, iy, nz),
                    histogram.GetBin(ix, iy, nz + 1),
                )
        return

    # Fold x first, including both y flow rows. Corners consequently move to
    # (1,0)/(nx,ny+1), then the y folds move them exactly once to edge bins.
    for iy in range(0, ny + 2):
        _add_bin(histogram, histogram.GetBin(1, iy), histogram.GetBin(0, iy))
        _add_bin(
            histogram,
            histogram.GetBin(nx, iy),
            histogram.GetBin(nx + 1, iy),
        )
    for ix in range(1, nx + 1):
        _add_bin(
            histogram,
            histogram.GetBin(ix, 1),
            histogram.GetBin(ix, 0),
        )
        _add_bin(
            histogram,
            histogram.GetBin(ix, ny),
            histogram.GetBin(ix, ny + 1),
        )


def dataframe_from_args(args):
    if args.input:
        return ROOT.RDataFrame("sndData", args.input), [args.input], None, None

    metadata = pd.read_csv(args.partition_metadata, keep_default_na=False)
    required = {
        "region",
        "particle_group",
        "feature_partition_path",
        "feature_available",
        "n_events",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Partition metadata lacks columns: {sorted(missing)}")
    selected = metadata.loc[
        (metadata["region"].astype(str) == args.region)
        & (metadata["particle_group"].astype(str) == args.particle_group)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No partitions for region={args.region!r}, "
            f"particle_group={args.particle_group!r}"
        )
    available = selected["feature_available"].astype(str).str.lower().isin(
        {"1", "true", "yes"}
    )
    if not available.all():
        unavailable = selected.loc[~available, "partition_id"].astype(str).tolist()
        raise RuntimeError(f"Unavailable feature partitions: {unavailable}")

    paths = selected.sort_values("partition_id")["feature_partition_path"].astype(str).tolist()
    chain = ROOT.TChain("sndData")
    for path in paths:
        if chain.Add(path) <= 0:
            raise RuntimeError(f"Could not add partition to sndData TChain: {path}")
    # Keep the TChain alive for at least as long as the RDataFrame. PyROOT's
    # RDataFrame wrapper does not take Python ownership of this object.
    return ROOT.RDataFrame(chain), paths, selected, chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="Legacy single consolidated sndData ROOT file")
    inputs.add_argument(
        "--partition-metadata",
        help="Combined partition metadata CSV containing feature_partition_path",
    )
    parser.add_argument("--region", default="all_events")
    parser.add_argument("--particle-group", default="passing_muon")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    ROOT.ROOT.EnableImplicitMT(args.threads)
    dataframe, input_paths, partition_rows, input_owner = dataframe_from_args(args)
    columns = {str(name) for name in dataframe.GetColumnNames()}
    required = {"count_veto", "hitTime_veto_min_ns"}
    for prefix in TRACK_TYPE_PREFIXES.values():
        required.update(
            {
                f"{prefix}_n_all",
                f"{prefix}_has_best",
                f"{prefix}_best_chi2ndf",
                f"{prefix}_best_start_x",
                f"{prefix}_best_start_y",
            }
        )
    missing = required - columns
    if missing:
        raise RuntimeError(f"Input is missing required branches: {sorted(missing)}")

    actions = {}
    for prefix in TRACK_TYPE_PREFIXES.values():
        label = prefix.removeprefix("reco_").replace("_", " ")
        actions[f"{prefix}_n_all"] = dataframe.Histo1D(
            (
                f"{prefix}_n_all",
                f"{label};Tracks per event;Events",
                2,
                -0.5,
                1.5,
            ),
            f"{prefix}_n_all",
        )
        actions[f"{prefix}_veto_count_vs_n_all"] = dataframe.Histo2D(
            (
                f"{prefix}_veto_count_vs_n_all",
                f"{label};Veto hit count;Tracks per event",
                30,
                -0.5,
                29.5,
                2,
                -0.5,
                1.5,
            ),
            "count_veto",
            f"{prefix}_n_all",
        )
        valid_time = dataframe.Filter(
            "count_veto > 0 && std::isfinite(hitTime_veto_min_ns) "
            "&& hitTime_veto_min_ns > -900"
        )
        actions[f"{prefix}_veto_time_vs_n_all"] = valid_time.Histo2D(
            (
                f"{prefix}_veto_time_vs_n_all",
                f"{label};Earliest veto-hit time [ns];Tracks per event",
                123,
                -0.625,
                30.125,
                2,
                -0.5,
                1.5,
            ),
            "hitTime_veto_min_ns",
            f"{prefix}_n_all",
        )
        best = dataframe.Filter(f"{prefix}_has_best == 1")
        actions[f"{prefix}_best_chi2ndf"] = best.Histo1D(
            (
                f"{prefix}_best_chi2ndf",
                f"{label};Best track #chi^{{2}}/NDF (overflow in last bin);Events",
                CHI2NDF_BINS,
                CHI2NDF_MIN,
                CHI2NDF_MAX,
            ),
            f"{prefix}_best_chi2ndf",
        )
        actions[f"{prefix}_veto_count_vs_chi2ndf"] = best.Histo2D(
            (
                f"{prefix}_veto_count_vs_chi2ndf",
                f"{label};Veto hit count;Best track #chi^{{2}}/NDF (overflow in last bin)",
                30,
                -0.5,
                29.5,
                CHI2NDF_BINS,
                CHI2NDF_MIN,
                CHI2NDF_MAX,
            ),
            "count_veto",
            f"{prefix}_best_chi2ndf",
        )
        valid_best_time = best.Filter(
            "count_veto > 0 && std::isfinite(hitTime_veto_min_ns) "
            "&& hitTime_veto_min_ns > -900"
        )
        actions[f"{prefix}_veto_time_vs_chi2ndf"] = valid_best_time.Histo2D(
            (
                f"{prefix}_veto_time_vs_chi2ndf",
                f"{label};Earliest veto-hit time [ns];Best track #chi^{{2}}/NDF (overflow in last bin)",
                123,
                -0.625,
                30.125,
                CHI2NDF_BINS,
                CHI2NDF_MIN,
                CHI2NDF_MAX,
            ),
            "hitTime_veto_min_ns",
            f"{prefix}_best_chi2ndf",
        )
        valid_best_start = best.Filter(
            f"std::isfinite({prefix}_best_start_x) "
            f"&& std::isfinite({prefix}_best_start_y)"
        )
        actions[f"{prefix}_start_xy_2d"] = valid_best_start.Histo2D(
            (
                f"{prefix}_start_xy_2d",
                f"{label};Track start x [cm];Track start y [cm]",
                20,
                -80.0,
                20.0,
                16,
                0.0,
                80.0,
            ),
            f"{prefix}_best_start_x",
            f"{prefix}_best_start_y",
        )
        actions[f"{prefix}_start_xy_chi2ndf_3d"] = valid_best_start.Histo3D(
            (
                f"{prefix}_start_xy_chi2ndf_3d",
                f"{label};Track start x [cm];Track start y [cm];"
                "Best track #chi^{2}/NDF (overflow in last bin)",
                20,
                -80.0,
                20.0,
                16,
                0.0,
                80.0,
                CHI2NDF_BINS,
                CHI2NDF_MIN,
                CHI2NDF_MAX,
            ),
            f"{prefix}_best_start_x",
            f"{prefix}_best_start_y",
            f"{prefix}_best_chi2ndf",
        )

    for method, scifi_prefix, ds_prefix in (
        ("simple", "reco_scifi_simple", "reco_ds_simple"),
        ("hough", "reco_scifi_hough", "reco_ds_hough"),
    ):
        actions[f"reco_{method}_ds_n_all_vs_scifi_n_all"] = dataframe.Histo2D(
            (
                f"reco_{method}_ds_n_all_vs_scifi_n_all",
                f"{method};SciFi tracks per event;"
                "DS tracks per event",
                2,
                -0.5,
                1.5,
                2,
                -0.5,
                1.5,
            ),
            f"{scifi_prefix}_n_all",
            f"{ds_prefix}_n_all",
        )
        both_best = dataframe.Filter(
            f"{scifi_prefix}_has_best == 1 && {ds_prefix}_has_best == 1"
        )
        actions[
            f"reco_{method}_ds_best_chi2ndf_vs_scifi_best_chi2ndf"
        ] = both_best.Histo2D(
            (
                f"reco_{method}_ds_best_chi2ndf_vs_scifi_best_chi2ndf",
                f"{method};SciFi best track #chi^{{2}}/NDF "
                "(overflow in last bin);DS best track #chi^{2}/NDF "
                "(overflow in last bin)",
                CHI2NDF_BINS,
                CHI2NDF_MIN,
                CHI2NDF_MAX,
                CHI2NDF_BINS,
                CHI2NDF_MIN,
                CHI2NDF_MAX,
            ),
            f"{scifi_prefix}_best_chi2ndf",
            f"{ds_prefix}_best_chi2ndf",
        )

    ROOT.RDF.RunGraphs(list(actions.values()))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = ROOT.TFile.Open(args.output, "RECREATE")
    for name, action in actions.items():
        histogram = action.GetValue()
        fold_flow_bins(histogram)
        histogram.SetDirectory(output)
        histogram.Write(name)
    ROOT.TNamed("track_feature_schema", "flat_v1").Write()
    ROOT.TNamed(
        "track_inspection_input_mode",
        "partition_metadata" if args.partition_metadata else "consolidated_root",
    ).Write()
    ROOT.TParameter("long long")(
        "track_inspection_input_file_count", len(input_paths)
    ).Write()
    if partition_rows is not None:
        expected_entries = int(pd.to_numeric(partition_rows["n_events"]).sum())
        ROOT.TParameter("long long")(
            "track_inspection_expected_entries", expected_entries
        ).Write()
        ROOT.TNamed(
            "track_inspection_partition_metadata",
            str(args.partition_metadata),
        ).Write()
    output.Close()
    print(f"Wrote {len(actions)} histograms to {args.output}")


if __name__ == "__main__":
    main()
