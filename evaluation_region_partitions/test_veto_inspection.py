#!/usr/bin/env python3

from __future__ import annotations

import csv
import argparse
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "snakemake/metadata/configs/region_v2_inspection/veto_inspection_v1.yaml"
V2_CONFIG = REPO_ROOT / "snakemake/metadata/configs/region_v2_inspection/veto_inspection_v2.yaml"
BUILDER = REPO_ROOT / "evaluation_region_partitions/make_veto_inspection_hist.py"
PLOTTER = REPO_ROOT / "evaluation_region_partitions/plot_veto_inspection.py"


def make_fixture(ROOT, path: Path, weight: float, include_invalid: bool) -> None:
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    tree = ROOT.TTree("sndData", "synthetic veto inspection")
    count_veto = array("i", [0])
    time_ns = array("d", [-999.0])
    count_scifi = array("i", [0])
    density_sndsw_scifi = array("d", [0.0])
    event_weight = array("d", [weight])
    tree.Branch("count_veto", count_veto, "count_veto/I")
    tree.Branch("hitTime_veto_min_ns", time_ns, "hitTime_veto_min_ns/D")
    tree.Branch("count_scifi", count_scifi, "count_scifi/I")
    tree.Branch(
        "density_sndsw_scifi",
        density_sndsw_scifi,
        "density_sndsw_scifi/D",
    )
    tree.Branch("inspection_event_weight", event_weight, "inspection_event_weight/D")
    events = [(0, -999.0), (0, -999.0), (1, -0.5), (2, 3.5), (4, 8.0), (45, 12.0)]
    if include_invalid:
        events.append((2, -999.0))
    for index, (count, time) in enumerate(events):
        count_veto[0] = count
        time_ns[0] = time
        count_scifi[0] = 20 + 10 * index
        density_sndsw_scifi[0] = 1000.0 + 500.0 * index
        event_weight[0] = weight
        tree.Fill()
    tree.Write()
    root_file.Close()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def make_fixtures(workdir: Path) -> None:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    datasets = [item["name"] for item in config["datasets"]]
    manifest = workdir / "manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_name", "output_path"])
        writer.writeheader()
        for index, dataset in enumerate(datasets):
            fixture = workdir / f"{dataset}.root"
            make_fixture(
                ROOT,
                fixture,
                1.0 if index < 2 else 0.1 * (index - 1),
                index == 0,
            )
            writer.writerow({"dataset_name": dataset, "output_path": fixture})


def run_workflow_test(workdir: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    datasets = [item["name"] for item in config["datasets"]]
    run([sys.executable, str(Path(__file__).resolve()), "--make-fixtures", str(workdir)])
    manifest = workdir / "manifest.csv"

    hist_args, summary_args, builder_commands = [], [], []
    for dataset in datasets:
        hist = workdir / "hists" / f"{dataset}.root"
        summary = workdir / "rows" / f"{dataset}.csv"
        builder_commands.append([
            sys.executable, str(BUILDER), "--config", str(CONFIG),
            "--manifest", str(manifest), "--dataset", dataset,
            "--output", str(hist), "--summary", str(summary),
            "--threads", "1", "--repo-root", str(REPO_ROOT),
        ])
        hist_args.extend(["--hist", f"{dataset}={hist}"])
        summary_args.extend(["--summary-row", f"{dataset}={summary}"])
    processes = [
        subprocess.Popen(command, cwd=REPO_ROOT) for command in builder_commands
    ]
    for process in processes:
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)

    plot_dir = workdir / "plots"
    combined, summary, done = workdir / "combined.root", workdir / "summary.csv", workdir / "done"
    run([
        sys.executable, str(PLOTTER), "--config", str(CONFIG),
        *hist_args, *summary_args, "--output-dir", str(plot_dir),
        "--combined-root", str(combined), "--summary", str(summary), "--done", str(done),
    ])
    stems = [
        "veto_count_all_unit_area", "veto_count_all_weighted",
        "veto_count_positive_unit_area", "veto_count_positive_weighted",
        "veto_earliest_time_all_unit_area", "veto_earliest_time_all_weighted",
        "veto_earliest_time_valid_unit_area", "veto_earliest_time_valid_weighted",
        "veto_time_vs_count_unit_area", "veto_time_vs_count_weighted",
        "veto_time_vs_scifi_count_unit_area",
        "veto_time_vs_scifi_count_weighted",
        "veto_time_vs_scifi_density_unit_area",
        "veto_time_vs_scifi_density_weighted",
        "veto_count_vs_scifi_count_unit_area",
        "veto_count_vs_scifi_count_weighted",
        "veto_count_vs_scifi_density_unit_area",
        "veto_count_vs_scifi_density_weighted",
    ]
    for stem in stems:
        for extension in config["plots"]["formats"]:
            path = plot_dir / f"{stem}.{extension}"
            assert path.is_file() and path.stat().st_size > 0, path
            if extension == "pdf":
                pdf_info = subprocess.run(
                    ["pdfinfo", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                pages = [
                    line for line in pdf_info.splitlines() if line.startswith("Pages:")
                ]
                assert len(pages) == 1 and int(pages[0].split(":", 1)[1]) == 1, (
                    path,
                    pages,
                )
    assert combined.is_file() and summary.is_file() and done.is_file()

    import ROOT

    ROOT.gROOT.SetBatch(True)
    first = ROOT.TFile.Open(str(workdir / "hists" / f"{datasets[0]}.root"))
    time_hist = first.Get("h_veto_earliest_time_all_raw")
    assert time_hist.GetBinContent(time_hist.FindBin(-1.0)) == 2.0
    assert time_hist.Integral() == 6.0
    count_hist = first.Get("h_veto_count_all_raw")
    assert count_hist.GetBinContent(count_hist.GetNbinsX()) == 1.0
    first.Close()
    combined_file = ROOT.TFile.Open(str(combined))
    for study in (
        "veto_time_vs_count",
        "veto_time_vs_scifi_count",
        "veto_time_vs_scifi_density",
        "veto_count_vs_scifi_count",
        "veto_count_vs_scifi_density",
    ):
        assert combined_file.Get(
            f"h2_{study}_combined_neutrino_weighted_display"
        )
    combined_file.Close()

    v2_hist = workdir / "v2" / "real_data.root"
    v2_summary = workdir / "v2" / "real_data.csv"
    run([
        sys.executable, str(BUILDER), "--config", str(V2_CONFIG),
        "--manifest", str(manifest), "--dataset", datasets[0],
        "--output", str(v2_hist), "--summary", str(v2_summary),
        "--threads", "1", "--repo-root", str(REPO_ROOT),
    ])
    with open(v2_summary, "r", encoding="utf-8", newline="") as handle:
        v2_row = next(csv.DictReader(handle))
    assert int(v2_row["source_events"]) == 7
    assert int(v2_row["total_events"]) == 5
    assert int(v2_row["no_veto_events"]) == 0

    v2_file = ROOT.TFile.Open(str(v2_hist))
    v2_count = v2_file.Get("h_veto_count_all_raw")
    assert v2_count.Integral() == 5.0
    assert v2_count.GetBinContent(v2_count.FindBin(0.0)) == 0.0
    v2_time = v2_file.Get("h_veto_earliest_time_all_raw")
    assert v2_time.Integral() == 4.0
    assert v2_time.GetBinContent(v2_time.FindBin(-1.0)) == 0.0
    v2_file.Close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-fixtures", type=Path)
    parser.add_argument("--keep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.make_fixtures:
        make_fixtures(args.make_fixtures)
        return
    if args.keep:
        value = Path(tempfile.mkdtemp(prefix="veto_inspection_test."))
        run_workflow_test(value)
        print(f"PASS; retained outputs at {value}")
        return
    with tempfile.TemporaryDirectory(prefix="veto_inspection_test.") as value:
        run_workflow_test(Path(value))
    print("PASS")


if __name__ == "__main__":
    main()
