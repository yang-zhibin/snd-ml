#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path

import ROOT

ROOT.gROOT.SetBatch(True)


DET_LABELS = {
    0: "SciFi",
    1: "Veto",
    2: "US",
    3: "DS",
}


def valid_time(value):
    try:
        value = float(value)
    except Exception:
        return False
    return math.isfinite(value) and value > -900.0


def make_hist(name, title, values, bins=100):
    values = list(values)
    if not values:
        hist = ROOT.TH1D(name, title, 1, 0.0, 1.0)
        hist.SetDirectory(0)
        return hist

    lo = min(values)
    hi = max(values)
    if hi <= lo:
        delta = max(abs(lo) * 1e-9, 1.0)
        lo -= delta
        hi += delta

    hist = ROOT.TH1D(name, title, bins, lo, hi)
    hist.SetDirectory(0)
    for value in values:
        hist.Fill(value)
    return hist


def save_canvas(hist, out_base, logy=False):
    canvas = ROOT.TCanvas("c", "c", 900, 700)
    if logy:
        canvas.SetLogy()
    hist.SetLineWidth(2)
    hist.Draw("hist")
    canvas.SaveAs(str(out_base.with_suffix(".png")))
    canvas.SaveAs(str(out_base.with_suffix(".pdf")))


def scan_file(path):
    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path}")

    tree = root_file.Get("sndData")
    if not tree:
        raise RuntimeError(f"Tree sndData not found in {path}")

    branches = {branch.GetName() for branch in tree.GetListOfBranches()}
    required = {"hitTime_raw_ns", "hitTime_raw_detType"}
    missing = sorted(required - branches)
    if missing:
        raise RuntimeError(f"Missing branches in {path}: {missing}")

    entries = int(tree.GetEntries())
    all_times = []
    times_by_det = {}
    event_ranges = []
    same_time_events = 0
    events_with_times = 0

    for entry_index, event in enumerate(tree):
        event_times = []
        for raw_time, det_type in zip(event.hitTime_raw_ns, event.hitTime_raw_detType):
            if not valid_time(raw_time):
                continue
            time_ns = float(raw_time)
            det_type = int(det_type)
            all_times.append(time_ns)
            event_times.append(time_ns)
            times_by_det.setdefault(det_type, []).append(time_ns)

        if not event_times:
            continue

        events_with_times += 1
        time_range = max(event_times) - min(event_times)
        event_ranges.append(time_range)
        if abs(time_range) < 1e-6:
            same_time_events += 1

    root_file.Close()
    return {
        "entries": entries,
        "events_with_times": events_with_times,
        "same_time_events": same_time_events,
        "all_times": all_times,
        "times_by_det": times_by_det,
        "event_ranges": event_ranges,
    }


def write_summary(summary_path, rows):
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample",
                "path",
                "entries",
                "events_with_times",
                "same_time_events",
                "same_time_fraction",
                "n_raw_times",
                "raw_time_min_ns",
                "raw_time_max_ns",
                "event_range_min_ns",
                "event_range_max_ns",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    combined_times = []
    combined_ranges = []
    combined_by_det = {}

    for input_path in args.inputs:
        path = Path(input_path)
        sample = path.stem
        print(f"Scanning {path}")
        result = scan_file(path)

        all_times = result["all_times"]
        event_ranges = result["event_ranges"]
        combined_times.extend(all_times)
        combined_ranges.extend(event_ranges)
        for det_type, values in result["times_by_det"].items():
            combined_by_det.setdefault(det_type, []).extend(values)

        same_fraction = (
            result["same_time_events"] / result["events_with_times"]
            if result["events_with_times"]
            else 0.0
        )
        summary_rows.append({
            "sample": sample,
            "path": str(path),
            "entries": result["entries"],
            "events_with_times": result["events_with_times"],
            "same_time_events": result["same_time_events"],
            "same_time_fraction": same_fraction,
            "n_raw_times": len(all_times),
            "raw_time_min_ns": min(all_times) if all_times else "",
            "raw_time_max_ns": max(all_times) if all_times else "",
            "event_range_min_ns": min(event_ranges) if event_ranges else "",
            "event_range_max_ns": max(event_ranges) if event_ranges else "",
        })

        save_canvas(
            make_hist(
                f"h_raw_time_{sample}",
                f"{sample};raw hit time [ns];hits",
                all_times,
            ),
            output_dir / f"{sample}__raw_hit_time",
            logy=True,
        )
        save_canvas(
            make_hist(
                f"h_event_range_{sample}",
                f"{sample};per-event raw hit time range [ns];events",
                event_ranges,
            ),
            output_dir / f"{sample}__event_raw_time_range",
            logy=True,
        )

        for det_type, values in sorted(result["times_by_det"].items()):
            label = DET_LABELS.get(det_type, f"detType{det_type}")
            save_canvas(
                make_hist(
                    f"h_raw_time_{sample}_{label}",
                    f"{sample} {label};raw hit time [ns];hits",
                    values,
                ),
                output_dir / f"{sample}__raw_hit_time__{label}",
                logy=True,
            )

    save_canvas(
        make_hist("h_raw_time_combined", "combined;raw hit time [ns];hits", combined_times),
        output_dir / "combined__raw_hit_time",
        logy=True,
    )
    save_canvas(
        make_hist(
            "h_event_range_combined",
            "combined;per-event raw hit time range [ns];events",
            combined_ranges,
        ),
        output_dir / "combined__event_raw_time_range",
        logy=True,
    )
    for det_type, values in sorted(combined_by_det.items()):
        label = DET_LABELS.get(det_type, f"detType{det_type}")
        save_canvas(
            make_hist(
                f"h_raw_time_combined_{label}",
                f"combined {label};raw hit time [ns];hits",
                values,
            ),
            output_dir / f"combined__raw_hit_time__{label}",
            logy=True,
        )

    summary_path = output_dir / "raw_hit_time_summary.csv"
    write_summary(summary_path, summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
