#!/usr/bin/env python3
"""Render original and transformed raw-digi events with official SNDSW."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--transformed-root", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--display-script", default="")
    parser.add_argument("--with-track", type=int, default=0)
    parser.add_argument("--with-hough-track", type=int, default=0)
    parser.add_argument("--minimum-tracks", type=int, default=0)
    parser.add_argument("--track-verbose", type=int, default=0)
    args = parser.parse_args()
    audit = json.load(open(args.audit_json, encoding="utf-8"))
    display_script = find_sndsw_2d_display_script(args.display_script)
    source = audit["source"]
    identity = audit["identity"]
    decision = audit.get("event", {}).get("axis_decision")
    geofile = audit["geometry"]["path"]
    original_row = {
        "event_source_path": source["path"],
        "source_event_index": int(source["entry"]),
        "geo_path": geofile,
    }
    transformed_row = {
        "event_source_path": str(Path(args.transformed_root).resolve()),
        "source_event_index": 0,
        "geo_path": geofile,
    }
    Path(args.output_pdf).parent.mkdir(parents=True, exist_ok=True)
    tracking_label = (
        f"tracking: simple={args.with_track}, Hough={args.with_hough_track}"
    )
    with tempfile.TemporaryDirectory(prefix="detid_shift_display_") as workdir:
        original_pages = run_sndsw_batch(
            [original_row], display_script, workdir, "eps",
            [["ORIGINAL DIGIS", f"Run/Event {identity['runId']}/{identity['eventId']}",
              tracking_label]],
            1600, 1000,
            args.with_track, args.with_hough_track,
            args.minimum_tracks, args.track_verbose,
        )
        transformed_label = "SciFi channel-sequence shift"
        if decision:
            transformed_label = (
                f"{decision['selected_axis'].upper()} only, "
                f"entry={decision['entry_side']}, {decision['direction_source']}"
            )
        transformed_pages = run_sndsw_batch(
            [transformed_row], display_script, workdir, "eps",
            [["TRANSFORMED DETECTOR IDs", transformed_label, tracking_label]],
            1600, 1000,
            args.with_track, args.with_hough_track,
            args.minimum_tracks, args.track_verbose,
        )
        if len(original_pages) != 1 or len(transformed_pages) != 1:
            raise RuntimeError(
                f"Expected one original and transformed page, got "
                f"{len(original_pages)}/{len(transformed_pages)}"
            )
        merge_pages_to_pdf(original_pages + transformed_pages, args.output_pdf)
    print(f"Wrote official SNDSW comparison to {args.output_pdf}")


if __name__ == "__main__":
    main()
