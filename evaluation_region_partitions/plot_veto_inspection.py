#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import ROOT
import yaml


ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)


ROOT_COLORS = {
    "black": ROOT.kBlack,
    "blue": ROOT.kBlue + 1,
    "green": ROOT.kGreen + 2,
    "red": ROOT.kRed + 1,
    "magenta": ROOT.kMagenta + 1,
    "orange_dark": ROOT.kOrange + 7,
}


ONE_D_STUDIES = {
    "veto_count_all": {
        "hist": "h_veto_count_all",
        "title": "Veto hit multiplicity;Veto hit count;Events",
        "log_y": True,
        "has_no_veto_time_bin": False,
    },
    "veto_count_positive": {
        "hist": "h_veto_count_positive",
        "title": "Positive veto hit multiplicity;Veto hit count;Events",
        "log_y": True,
        "has_no_veto_time_bin": False,
    },
    "veto_earliest_time_all": {
        "hist": "h_veto_earliest_time_all",
        "title": "Earliest veto-hit time, including no-veto category;"
        "Earliest veto-hit time [ns];Events",
        "log_y": True,
        "has_no_veto_time_bin": True,
    },
    "veto_earliest_time_valid": {
        "hist": "h_veto_earliest_time_valid",
        "title": "Earliest veto-hit time for veto-active events;"
        "Earliest veto-hit time [ns];Events",
        "log_y": True,
        "has_no_veto_time_bin": False,
    },
}

TWO_D_STUDIES = {
    "veto_time_vs_count": {
        "hist": "h2_veto_time_vs_count",
        "has_no_veto_time_bin": True,
    },
    "veto_time_vs_scifi_count": {
        "hist": "h2_veto_time_vs_scifi_count",
        "has_no_veto_time_bin": True,
    },
    "veto_time_vs_scifi_density": {
        "hist": "h2_veto_time_vs_scifi_density",
        "has_no_veto_time_bin": True,
    },
    "veto_count_vs_scifi_count": {
        "hist": "h2_veto_count_vs_scifi_count",
        "has_no_veto_time_bin": False,
    },
    "veto_count_vs_scifi_density": {
        "hist": "h2_veto_count_vs_scifi_density",
        "has_no_veto_time_bin": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the six Region V2 veto-inspection histogram files."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--hist", action="append", required=True)
    parser.add_argument("--summary-row", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--combined-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--done", required=True)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def indexed_paths(values: list[str], kind: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{kind} must use DATASET=PATH syntax: {value!r}")
        dataset, path = value.split("=", 1)
        dataset = dataset.strip()
        if not dataset or dataset in result:
            raise ValueError(f"Invalid or duplicate {kind} dataset: {dataset!r}")
        result[dataset] = path
    return result


def read_summary_rows(paths: dict[str, str], order: list[str]) -> list[dict[str, str]]:
    rows = []
    expected_columns = None
    for dataset in order:
        with open(paths[dataset], "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            file_rows = list(reader)
            columns = reader.fieldnames
        if len(file_rows) != 1:
            raise ValueError(
                f"Expected one summary row for {dataset}; found {len(file_rows)}"
            )
        if file_rows[0].get("dataset_name") != dataset:
            raise ValueError(
                f"Summary row dataset mismatch for {dataset}: "
                f"{file_rows[0].get('dataset_name')!r}"
            )
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(f"Summary columns differ for dataset {dataset}")
        rows.append(file_rows[0])
    return rows


def clone_hist(root_file: ROOT.TFile, name: str, clone_name: str):
    source = root_file.Get(name)
    if not source:
        raise KeyError(f"Histogram {name!r} is missing in {root_file.GetName()}")
    hist = source.Clone(clone_name)
    hist.SetDirectory(0)
    return hist


def dataset_style(config: dict, dataset: str) -> dict:
    matches = [item for item in config["datasets"] if item["name"] == dataset]
    if len(matches) != 1:
        raise ValueError(f"Expected one style for dataset={dataset!r}")
    return matches[0]


def color(value: object) -> int:
    if isinstance(value, int):
        return value
    if str(value) not in ROOT_COLORS:
        raise KeyError(f"Unknown ROOT color name: {value!r}")
    return ROOT_COLORS[str(value)]


def style_hist(hist, style: dict, plots_cfg: dict) -> None:
    hist.SetLineColor(color(style["color"]))
    hist.SetMarkerColor(color(style["color"]))
    hist.SetMarkerStyle(int(style["marker"]))
    hist.SetMarkerSize(float(plots_cfg["marker_size"]))
    hist.SetLineWidth(int(plots_cfg["line_width"]))


def integral_with_flow(hist) -> float:
    return float(hist.Integral(0, hist.GetNbinsX() + 1))


def normalize_1d(hist) -> None:
    integral = integral_with_flow(hist)
    if integral <= 0 or not math.isfinite(integral):
        raise ValueError(f"Cannot normalize empty histogram {hist.GetName()}")
    hist.Scale(1.0 / integral)


def normalize_2d(hist) -> None:
    integral = float(
        hist.Integral(
            0,
            hist.GetNbinsX() + 1,
            0,
            hist.GetNbinsY() + 1,
        )
    )
    if integral <= 0 or not math.isfinite(integral):
        raise ValueError(f"Cannot normalize empty histogram {hist.GetName()}")
    hist.Scale(1.0 / integral)


def positive_minimum(hist) -> float | None:
    values = [
        hist.GetBinContent(index)
        for index in range(1, hist.GetNbinsX() + 1)
        if hist.GetBinContent(index) > 0
    ]
    return min(values) if values else None


def draw_header(config: dict) -> None:
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextSize(0.032)
    latex.DrawLatex(0.12, 0.965, str(config["plots"]["preliminary_label"]))
    latex.SetTextAlign(31)
    latex.DrawLatex(0.90, 0.965, str(config["plots"]["luminosity_label"]))


def save_canvas(canvas, output_dir: Path, stem: str, formats: list[str]) -> None:
    for extension in formats:
        final_path = output_dir / f"{stem}.{extension}"
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{stem}.",
            suffix=f".{extension}",
            dir=output_dir,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            canvas.Update()
            canvas.SaveAs(str(temporary_path))
            os.replace(temporary_path, final_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def plot_1d(
    config: dict,
    root_files: dict[str, ROOT.TFile],
    output_dir: Path,
    study_name: str,
    study: dict,
    mode: str,
) -> None:
    plots_cfg = config["plots"]
    datasets = [item["name"] for item in config["datasets"]]
    suffix = "raw" if mode == "unit_area" else "weighted"
    histograms = []
    for dataset in datasets:
        hist = clone_hist(
            root_files[dataset],
            f"{study['hist']}_{suffix}",
            f"{study_name}_{mode}_{dataset}",
        )
        if mode == "unit_area":
            normalize_1d(hist)
        style_hist(hist, dataset_style(config, dataset), plots_cfg)
        histograms.append((dataset, hist))

    width, height = [int(value) for value in plots_cfg["canvas_1d"]]
    canvas = ROOT.TCanvas(f"c_{study_name}_{mode}", "", width, height)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.04)
    canvas.SetBottomMargin(0.12)
    canvas.SetTopMargin(0.12)
    canvas.SetLogy(bool(study["log_y"] and plots_cfg.get("log_y", True)))

    maximum = max(hist.GetMaximum() for _, hist in histograms)
    positive = [
        value
        for _, hist in histograms
        for value in [positive_minimum(hist)]
        if value is not None
    ]
    minimum = min(positive) if positive else 1e-9
    if canvas.GetLogy():
        y_min = max(minimum * 0.35, 1e-12)
        y_max = max(maximum * 8.0, y_min * 100.0)
    else:
        y_min = 0.0
        y_max = maximum * 1.35

    draw_option = str(plots_cfg["draw_option_1d"])
    title_parts = str(study["title"]).split(";")
    main_title = title_parts[0]
    x_title = title_parts[1]
    for index, (_, hist) in enumerate(histograms):
        hist.SetTitle(f";{x_title};")
        hist.GetYaxis().SetTitle(
            "Fraction of events / bin" if mode == "unit_area" else "Weighted events / bin"
        )
        hist.GetXaxis().SetTitleOffset(1.15)
        hist.GetYaxis().SetTitleOffset(1.35)
        hist.SetMinimum(y_min)
        hist.SetMaximum(y_max)
        option = draw_option if index == 0 else f"{draw_option} SAME"
        hist.Draw(option)

    legend = ROOT.TLegend(0.57, 0.56, 0.92, 0.87)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.030)
    for dataset, hist in histograms:
        legend.AddEntry(hist, dataset_style(config, dataset)["label"], "lep")
    legend.Draw()

    draw_header(config)
    labels = ROOT.TLatex()
    labels.SetNDC(True)
    labels.SetTextFont(42)
    labels.SetTextSize(0.038)
    labels.DrawLatex(0.12, 0.905, main_title)
    labels.SetTextSize(0.028)
    labels.DrawLatex(
        0.14,
        0.845,
        "Independent unit-area shapes"
        if mode == "unit_area"
        else "Luminosity-weighted yields; passing muon shown separately",
    )
    if study["has_no_veto_time_bin"] and plots_cfg.get(
        "show_no_veto_label", True
    ):
        labels.DrawLatex(0.14, 0.800, str(plots_cfg["no_veto_label"]))

    save_canvas(
        canvas,
        output_dir,
        f"{study_name}_{mode}",
        [str(item) for item in plots_cfg["formats"]],
    )
    canvas.Close()


def combined_neutrino(
    config: dict,
    root_files: dict[str, ROOT.TFile],
    hist_name: str,
    suffix: str,
    name: str,
):
    neutrino_names = [
        item["name"] for item in config["datasets"] if item["category"] == "neutrino"
    ]
    if not neutrino_names:
        raise ValueError("No neutrino datasets are configured")
    combined = clone_hist(
        root_files[neutrino_names[0]],
        f"{hist_name}_{suffix}",
        name,
    )
    for dataset in neutrino_names[1:]:
        component = clone_hist(
            root_files[dataset],
            f"{hist_name}_{suffix}",
            f"{name}_{dataset}",
        )
        combined.Add(component)
    return combined


def positive_minimum_2d(hist) -> float | None:
    values = []
    for ix in range(1, hist.GetNbinsX() + 1):
        for iy in range(1, hist.GetNbinsY() + 1):
            value = hist.GetBinContent(ix, iy)
            if value > 0:
                values.append(value)
    return min(values) if values else None


def plot_2d(
    config: dict,
    root_files: dict[str, ROOT.TFile],
    output_dir: Path,
    study_name: str,
    study: dict,
    mode: str,
    combined_output: ROOT.TFile,
) -> None:
    categories = {}
    for item in config["datasets"]:
        categories[item["category"]] = item["name"]
    if "real_data" not in categories or "passing_muon" not in categories:
        raise ValueError("Both real_data and passing_muon categories are required")

    # Unit-area neutrino composition must still use physical luminosity weights;
    # normalize only after the four weighted components have been combined.
    suffix = "weighted"
    panels = [
        (
            "Real data",
            clone_hist(
                root_files[categories["real_data"]],
                f"{study['hist']}_{suffix}",
                f"h2_{study_name}_real_data_{mode}",
            ),
        ),
        (
            "Passing-muon data",
            clone_hist(
                root_files[categories["passing_muon"]],
                f"{study['hist']}_{suffix}",
                f"h2_{study_name}_passing_muon_{mode}",
            ),
        ),
        (
            "Combined neutrino MC",
            combined_neutrino(
                config,
                root_files,
                study["hist"],
                suffix,
                f"h2_{study_name}_combined_neutrino_{mode}",
            ),
        ),
    ]
    if mode == "unit_area":
        for _, hist in panels:
            normalize_2d(hist)

    maxima = [hist.GetMaximum() for _, hist in panels]
    minima = [
        value
        for _, hist in panels
        for value in [positive_minimum_2d(hist)]
        if value is not None
    ]
    z_max = max(maxima)
    z_min = min(minima) if minima else 1e-12
    if config["plots"].get("log_z", True):
        z_min = max(z_min * 0.5, 1e-12)

    width, height = [int(value) for value in config["plots"]["canvas_2d"]]
    canvas = ROOT.TCanvas(f"c_{study_name}_{mode}", "", width, height)
    histogram_cfg = config["histograms"][study_name]
    canvas.Divide(3, 1, 0.002, 0.002)
    for index, (title, hist) in enumerate(panels, start=1):
        pad = canvas.cd(index)
        pad.SetLeftMargin(float(config["plots"].get("margin_left_2d", 0.16)))
        pad.SetRightMargin(0.16)
        pad.SetBottomMargin(0.13)
        pad.SetTopMargin(0.13)
        pad.SetLogz(bool(config["plots"].get("log_z", True)))
        hist.SetTitle(
            f"{title};{histogram_cfg['x_title']};{histogram_cfg['y_title']};"
            + ("Fraction / bin" if mode == "unit_area" else "Weighted events / bin")
        )
        hist.SetMinimum(z_min)
        hist.SetMaximum(z_max)
        hist.GetXaxis().SetTitleOffset(1.15)
        hist.GetYaxis().SetTitleOffset(
            float(config["plots"].get("y_title_offset_2d", 1.55))
        )
        hist.Draw("COLZ")
        if study["has_no_veto_time_bin"] and config["plots"].get(
            "show_no_veto_label", True
        ):
            latex = ROOT.TLatex()
            latex.SetNDC(True)
            latex.SetTextFont(42)
            latex.SetTextSize(0.034)
            latex.DrawLatex(0.15, 0.91, str(config["plots"]["no_veto_label"]))

        combined_output.cd()
        hist.Write(f"{hist.GetName()}_display")

    save_canvas(
        canvas,
        output_dir,
        f"{study_name}_{mode}",
        [str(item) for item in config["plots"]["formats"]],
    )
    canvas.Close()


def combine_summary(rows: list[dict[str, str]], output: Path) -> None:
    if not rows:
        raise ValueError("No summary rows were provided")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    order = [item["name"] for item in config["datasets"]]
    hist_paths = indexed_paths(args.hist, "--hist")
    summary_paths = indexed_paths(args.summary_row, "--summary-row")
    for kind, paths in (("histogram", hist_paths), ("summary", summary_paths)):
        missing = set(order) - set(paths)
        extra = set(paths) - set(order)
        if missing or extra:
            raise ValueError(
                f"{kind} dataset mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_root_path = Path(args.combined_root)
    combined_root_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_summary_rows(summary_paths, order)
    combine_summary(rows, Path(args.summary))

    root_files = {}
    try:
        for dataset in order:
            root_file = ROOT.TFile.Open(hist_paths[dataset])
            if not root_file or root_file.IsZombie():
                raise OSError(f"Could not open histogram ROOT file: {hist_paths[dataset]}")
            root_files[dataset] = root_file

        combined_output = ROOT.TFile.Open(str(combined_root_path), "RECREATE")
        if not combined_output or combined_output.IsZombie():
            raise OSError(f"Could not create {combined_root_path}")
        try:
            one_dimensional = ONE_D_STUDIES
            if not config["plots"].get("include_inclusive_1d", True):
                one_dimensional = {
                    name: study
                    for name, study in ONE_D_STUDIES.items()
                    if name in {"veto_count_positive", "veto_earliest_time_valid"}
                }
            for study_name, study in one_dimensional.items():
                for mode in ("unit_area", "weighted"):
                    plot_1d(
                        config,
                        root_files,
                        output_dir,
                        study_name,
                        study,
                        mode,
                    )
            for study_name, study in TWO_D_STUDIES.items():
                for mode in ("unit_area", "weighted"):
                    plot_2d(
                        config,
                        root_files,
                        output_dir,
                        study_name,
                        study,
                        mode,
                        combined_output,
                    )
            combined_output.cd()
            ROOT.TNamed("inspection_version", str(config["version"])).Write()
            ROOT.TNamed(
                "neutrino_combination",
                "CC_nue + NC_nue + CC_numu + NC_numu after inspection weighting",
            ).Write()
        finally:
            combined_output.Close()
    finally:
        for root_file in root_files.values():
            root_file.Close()

    done = Path(args.done)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(
        f"{config['version']}: plots and summaries completed\n",
        encoding="utf-8",
    )
    print(f"Wrote veto inspection plots to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
