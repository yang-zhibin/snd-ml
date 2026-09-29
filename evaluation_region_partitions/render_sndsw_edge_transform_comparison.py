#!/usr/bin/env python3
"""Render an original/transformed comparison using the official SNDSW canvas."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path


def graph(ROOT, points, name, color, marker=20, line_style=1, line=False):
    value = ROOT.TGraphErrors() if not line else ROOT.TGraph()
    value.SetName(name)
    for index, point in enumerate(points):
        value.SetPoint(index, float(point["z_cm"]), float(point["u_cm"]))
        if not line:
            value.SetPointError(
                index, float(point.get("z_error_cm", 0.2)),
                float(point.get("u_error_cm", 0.2)),
            )
    value.SetMarkerStyle(marker)
    value.SetMarkerSize(1.25)
    value.SetMarkerColor(color)
    value.SetLineColor(color)
    value.SetLineWidth(2 if line else 1)
    value.SetLineStyle(line_style)
    return value


def draw_track(ROOT, namespace, payload, prefix, transformed=False):
    canvas, objects = namespace["h"]["simpleDisplay"], namespace["h"]
    color = ROOT.kAzure - 2
    style = 2 if transformed else 1
    for pad_index, projection in ((1, "X"), (2, "Y")):
        points = payload["tracks"]["transformed" if transformed else "original"][projection]
        item = graph(
            ROOT, points, f"{prefix}_track_{projection}", color,
            marker=20, line_style=style, line=True,
        )
        objects[item.GetName()] = item
        canvas.cd(pad_index)
        item.Draw("sameLP" if transformed else "sameL")


def clone_current_pads(ROOT, namespace, prefix):
    canvas = namespace["h"]["simpleDisplay"]
    canvas.Modified()
    canvas.Update()
    return [canvas.GetPad(index).Clone(f"{prefix}_pad_{index}") for index in (1, 2)]


def rebuild_transformed_canvas(ROOT, namespace, payload):
    h, canvas = namespace["h"], namespace["h"]["simpleDisplay"]
    for pad_index, histogram in ((1, "xz"), (2, "yz")):
        pad = canvas.cd(pad_index)
        pad.Clear()
        h[histogram].Draw("b")
    if namespace.get("withDetector", True):
        namespace["drawDetectors"]()

    colors = {
        "Scifi": ROOT.kBlue + 2,
        "Veto": ROOT.kRed + 1,
        "US": ROOT.kBlack,
        "DS": ROOT.kBlack,
    }
    pad_for_projection = {"X": 1, "Y": 2}
    for projection in ("X", "Y"):
        canvas.cd(pad_for_projection[projection])
        for system in ("Veto", "Scifi", "US", "DS"):
            points = payload["transformed_hits"][projection].get(system, [])
            if not points:
                continue
            item = graph(
                ROOT, points, f"transformed_{projection}_{system}",
                colors[system], marker=20,
            )
            h[item.GetName()] = item
            item.Draw("sameP")

    draw_track(ROOT, namespace, payload, "transformed", transformed=True)
    run_id, event_id = payload["run_id"], payload["event_id"]
    event_time = namespace["eventTree"].EventHeader.GetEventTime()
    info = [
        "TRANSFORMED DISPLAY",
        "SciFi, Veto, US and DS hits shifted",
        "Dashed line: transformed track reference",
    ]
    namespace["drawInfo"](canvas, 1, run_id, event_id, event_time, info)
    namespace["drawInfo"](canvas, 2, run_id, event_id, event_time, info)
    canvas.Modified()
    canvas.Update()


def draw_wide_canvas(ROOT, original_pads, transformed_pads, output_pdf, payload):
    canvas = ROOT.TCanvas("edge_transform_comparison", "edge transform comparison", 2400, 1500)
    canvas.Divide(2, 2, 0.002, 0.002)
    order = (
        (1, original_pads[0], "Original z-x"),
        (2, transformed_pads[0], "Transformed z-x"),
        (3, original_pads[1], "Original z-y"),
        (4, transformed_pads[1], "Transformed z-y"),
    )
    labels = []
    drawn_pads = []
    for target, source, title in order:
        canvas.cd(target)
        # The source pads occupy the upper/lower halves of the official
        # two-row canvas. Reset their NDC box after cloning so each one fills
        # its quadrant in this two-by-two comparison canvas.
        cloned = source.Clone(f"wide_{target}_{source.GetName()}")
        cloned.SetPad(0.0, 0.0, 1.0, 1.0)
        cloned.Draw()
        drawn_pads.append(cloned)
        label = ROOT.TLatex()
        label.SetNDC(True)
        label.SetTextFont(42)
        label.SetTextSize(0.035)
        label.SetTextColor(ROOT.kMagenta + 2 if "Transformed" in title else ROOT.kBlue + 2)
        label.DrawLatex(0.68, 0.94, title)
        labels.append(label)
    canvas.cd()
    heading = ROOT.TLatex()
    heading.SetNDC(True)
    heading.SetTextAlign(22)
    heading.SetTextFont(42)
    heading.SetTextSize(0.018)
    heading.DrawLatex(
        0.5, 0.995,
        f"Run {payload['run_id']} / Event {payload['event_id']} — "
        "Hough SciFi type 11",
    )
    canvas.Modified()
    canvas.Update()
    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    # ROOT's direct PDF backend gives a landscape canvas a portrait page box,
    # leaving most of the page blank.  The official display already uses EPS;
    # retain that route and crop the EPS bounding box during PDF conversion.
    with tempfile.TemporaryDirectory(prefix="sndsw_comparison_export_") as workdir:
        output_eps = Path(workdir) / "comparison.eps"
        canvas.Print(str(output_eps))
        subprocess.run(
            ["ps2pdf", "-dEPSCrop", str(output_eps), str(output_pdf)],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--display-script", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--geo", required=True)
    parser.add_argument("--event-index", required=True, type=int)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()
    with open(args.payload, encoding="utf-8") as stream:
        payload = json.load(stream)

    with tempfile.TemporaryDirectory(prefix="sndsw_transform_canvas_") as workdir:
        prefix = os.path.join(workdir, "official_original__")
        sys.argv = [
            args.display_script, "-f", args.input, "-p", "", "-g", args.geo,
            "--rootbatch", "--extension", "eps",
        ]
        namespace = runpy.run_path(args.display_script, run_name="__main__")
        ROOT = namespace["ROOT"]
        ROOT.gROOT.SetBatch(True)
        namespace["options"].storePic = prefix
        namespace["ut"].bookCanvas(
            namespace["h"], key="simpleDisplay", title="simple event display",
            nx=1600, ny=1000, cx=1, cy=2,
        )
        namespace["h"]["simpleDisplay"].SetCanvasSize(1600, 1000)
        namespace["loopEvents"](
            start=[args.event_index], auto=True, save=False, goodEvents=False,
            withTrack=0, withHoughTrack=0, nTracks=0, verbose=0,
            hitColour=None,
        )
        observed_run = int(namespace["eventTree"].EventHeader.GetRunId())
        observed_event = int(namespace["eventTree"].EventHeader.GetEventNumber())
        if (observed_run, observed_event) != (
            int(payload["run_id"]), int(payload["event_id"]),
        ):
            raise RuntimeError(
                f"Official display identity mismatch: "
                f"{observed_run}/{observed_event} != "
                f"{payload['run_id']}/{payload['event_id']}"
            )
        draw_track(ROOT, namespace, payload, "original", transformed=False)
        original_pads = clone_current_pads(ROOT, namespace, "original")
        rebuild_transformed_canvas(ROOT, namespace, payload)
        transformed_pads = clone_current_pads(ROOT, namespace, "transformed")
        draw_wide_canvas(
            ROOT, original_pads, transformed_pads, args.output_pdf, payload,
        )
    print(f"Wrote SNDSW-style transform comparison to {args.output_pdf}")


if __name__ == "__main__":
    main()
