#!/usr/bin/env python3
"""Render original, track-v4, and all-plane-average-v5 official displays."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from sndsw_event_display_common import (
    find_sndsw_2d_display_script,
    merge_pages_to_pdf,
    run_sndsw_batch,
)


def line_specification(audit: dict, shifted: bool, color: int, style: int = 1) -> dict:
    event = audit["event"]
    line = event["track"]
    shift_x = shift_y = 0.0
    if shifted:
        axis = event["axis_decision"]["selected_axis"]
        shift = float(event["reference"]["requested_shift_cm"])
        if axis == "x":
            shift_x = shift
        else:
            shift_y = shift
    return {
        **line,
        "shift_x_cm": shift_x,
        "shift_y_cm": shift_y,
        "color": color,
        "style": style,
        "width": 2,
    }


def render_one(row, display_script, workdir, tag, annotations, overlays):
    pages = run_sndsw_batch(
        [row], display_script, workdir, "eps", [annotations], 1600, 1000,
        with_track=0, with_hough_track=0, minimum_tracks=0, track_verbose=0,
        overlay_lines=[overlays],
    )
    if len(pages) != 1:
        raise RuntimeError(f"Expected one {tag} page, found {len(pages)}")
    return pages[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-audit-json", required=True)
    parser.add_argument("--track-root", required=True)
    parser.add_argument("--simple-audit-json", required=True)
    parser.add_argument("--simple-root", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--display-script", default="")
    args = parser.parse_args()
    track_audit = json.load(open(args.track_audit_json, encoding="utf-8"))
    simple_audit = json.load(open(args.simple_audit_json, encoding="utf-8"))
    if track_audit["identity"] != simple_audit["identity"]:
        raise RuntimeError("Track and simple transforms have different event identities")
    identity = track_audit["identity"]
    source = track_audit["source"]
    geofile = track_audit["geometry"]["path"]
    original = {
        "event_source_path": source["path"],
        "source_event_index": int(source["entry"]),
        "geo_path": geofile,
    }
    track_transformed = {
        "event_source_path": str(Path(args.track_root).resolve()),
        "source_event_index": 0,
        "geo_path": geofile,
    }
    simple_transformed = {
        "event_source_path": str(Path(args.simple_root).resolve()),
        "source_event_index": 0,
        "geo_path": geofile,
    }
    track_original_line = line_specification(track_audit, False, 4)
    activity_original_line = line_specification(simple_audit, False, 6, 2)
    track_shifted_line = line_specification(track_audit, True, 4)
    activity_shifted_line = line_specification(simple_audit, True, 6)
    track_decision = track_audit["event"]["axis_decision"]
    simple_decision = simple_audit["event"]["axis_decision"]
    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    display_script = find_sndsw_2d_display_script(args.display_script)
    with tempfile.TemporaryDirectory(prefix="detid_shift_v5_display_") as workdir:
        pages = [
            render_one(
                original, display_script, workdir, "original",
                ["ORIGINAL DIGIS", f"Run/Event {identity['runId']}/{identity['eventId']}",
                 "blue: frozen SciFi track", "magenta dashed: all-plane average fit"],
                [track_original_line, activity_original_line],
            ),
            render_one(
                track_transformed, display_script, workdir, "track transform",
                ["SCIFI-TRACK TRANSFORM", f"axis={track_decision['selected_axis']}, "
                 f"entry={track_audit['event']['reference']['target_side']}",
                 "blue: shifted frozen SciFi track"],
                [track_shifted_line],
            ),
            render_one(
                simple_transformed, display_script, workdir, "simple transform",
                ["ALL-PLANE AVERAGE TRANSFORM", f"axis={simple_decision['selected_axis']}, "
                 f"entry={simple_audit['event']['reference']['target_side']}",
                 "magenta: shifted activity line"],
                [activity_shifted_line],
            ),
        ]
        merge_pages_to_pdf(pages, str(output))
    print(f"Wrote v5 three-way official display to {output}")


if __name__ == "__main__":
    main()
