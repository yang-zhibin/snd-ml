#!/usr/bin/env python3
"""Render original/transformed SciFi channel pages with the V7 display code."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from detid_channel_shift_sndsw import public_eos_path
from scifi_sipm_event_display import create_sipm_geometry_context, render_plane_pages
from sndsw_event_display_common import merge_pages_to_pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--transformed-root", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--plane-page-width", type=float, default=66.0)
    args = parser.parse_args()
    audit = json.load(open(args.audit_json, encoding="utf-8"))
    source = audit["source"]
    identity = audit["identity"]
    geofile = audit["geometry"]["path"]
    decision = audit.get("event", {}).get("axis_decision")
    policy_label = ""
    if decision:
        policy_label = (
            f" — {decision['selected_axis'].upper()} only; "
            f"sx={decision['slope_xz']:.5g}, sy={decision['slope_yz']:.5g}; "
            f"entry={decision['entry_side']}; {decision['direction_source']}"
        )
    context = create_sipm_geometry_context(geofile)
    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="detid_shift_scifi_channels_") as workdir:
        original_pdf = os.path.join(workdir, "original.pdf")
        transformed_pdf = os.path.join(workdir, "transformed.pdf")
        original_audit = render_plane_pages(
            public_eos_path(source["path"]), source["tree"], int(source["entry"]),
            geofile, original_pdf,
            f"ORIGINAL — Run {identity['runId']} / Event {identity['eventId']}{policy_label}",
            args.plane_page_width, context,
        )
        transformed_audit = render_plane_pages(
            str(Path(args.transformed_root).resolve()), source["tree"], 0,
            geofile, transformed_pdf,
            f"TRANSFORMED DETECTOR IDs — Run {identity['runId']} / Event {identity['eventId']}{policy_label}",
            args.plane_page_width, context,
        )
        merge_pages_to_pdf([original_pdf, transformed_pdf], str(output))
    combined = {
        "version": audit["version"],
        "identity": identity,
        "axis_decision": decision,
        "page_order": [
            "original_horizontal", "original_vertical",
            "transformed_horizontal", "transformed_vertical",
        ],
        "original": original_audit,
        "transformed": transformed_audit,
        "output_pdf": str(output.resolve()),
    }
    audit_output = Path(args.output_audit)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{audit_output.name}.", dir=audit_output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(combined, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, audit_output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote V7-style SciFi channel comparison to {output}")


if __name__ == "__main__":
    main()
