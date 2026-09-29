#!/usr/bin/env python3
"""Render one MC-neutrino region/type as overview + two SciFi pages per event."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from plot_saved_events_sndsw import (
    collect_events,
    enrich_display_labels,
    find_sndsw_2d_display_script,
    load_eval_plot_config,
    load_yaml,
    merge_pages_to_pdf,
    resolve_display_source,
    resolve_path,
    run_sndsw_for_batch,
)
from scifi_sipm_event_display import create_sipm_geometry_context, render_plane_pages


PARTICLES = ("CC_nue", "CC_numu", "NC_nue", "NC_numu")
SUMMARY_FIELDS = (
    "region", "region_label", "particle_group", "particle_label",
    "event_sequence", "first_document_page", "partition_id",
    "feature_partition_path", "feature_entry", "original_entry",
    "source_feature_path", "event_source_path", "event_tree_name",
    "source_event_index", "runId", "eventId", "geofile", "output_pdf",
    "selection_expression", "total_recorded_scifi_hits",
    "total_valid_scifi_hits",
)


def write_csv(rows: list[dict], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--particle", required=True, choices=PARTICLES)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--sndsw-display-script")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config = load_yaml(resolve_path(args.config, repo_root))
    configured_particles = list(config["filters"]["particles"])
    configured_regions = list(config["filters"]["regions"])
    if args.particle not in configured_particles or args.region not in configured_regions:
        raise ValueError(f"Unconfigured region/particle: {args.region}/{args.particle}")

    # Each Snakemake job selects only its own group, allowing all 32 groups to
    # run independently while retaining the generic partition/source mapper.
    config["filters"] = {"regions": [args.region], "particles": [args.particle]}

    def prepare_display_source(event: dict) -> bool:
        try:
            event_source, tree_name, geofile = resolve_display_source(
                event, config, repo_root
            )
        except (OSError, RuntimeError) as error:
            print(
                f"[skip] unusable display candidate "
                f"{event.get('partition_id', '')}:{event.get('feature_entry', '')}: "
                f"{error}"
            )
            return False
        event["event_source_path"] = event_source
        event["event_tree_name"] = tree_name
        event["geofile"] = geofile
        return True

    # Count only candidates that have a readable event tree and an in-range
    # source entry toward the configured per-region limit.  A bad source row
    # therefore does not discard already rendered pages or the entire PDF.
    grouped = collect_events(
        config,
        args.metadata_csv,
        repo_root,
        event_validator=prepare_display_source,
    )
    enrich_display_labels(grouped, load_eval_plot_config(config, repo_root))
    events = grouped.get((args.region, args.particle), [])
    if not events:
        raise RuntimeError(f"No selected events for {args.region}/{args.particle}")

    display = config.get("display", {}) or {}
    display_script = find_sndsw_2d_display_script(
        args.sndsw_display_script
        or str(display.get("sndsw_display_script", "") or "").strip()
        or None
    )
    extension = "eps" if display.get("output_mode", "vector_pdf") == "vector_pdf" else "png"
    plane_width = float(display.get("plane_page_width", 66.0))
    output_pdf = Path(args.output_pdf).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    audits = []
    geometry_contexts = {}
    destination_created = False
    copy_succeeded = False
    try:
        with tempfile.TemporaryDirectory(prefix=f"mc_nu_{args.particle}_") as workdir:
            document_parts = []
            for sequence, event in enumerate(events, start=1):
                event_source = event["event_source_path"]
                tree_name = event["event_tree_name"]
                geofile = event["geofile"]

                event_workdir = os.path.join(workdir, f"event_{sequence:04d}")
                os.makedirs(event_workdir)
                overview_pages = run_sndsw_for_batch(
                    [event], display_script, event_workdir, extension, config
                )
                if len(overview_pages) != 1:
                    raise RuntimeError(
                        f"SNDSW made {len(overview_pages)} overview pages for event {sequence}"
                    )
                overview_pdf = os.path.join(event_workdir, "overview.pdf")
                merge_pages_to_pdf(overview_pages, overview_pdf)

                context = geometry_contexts.get(geofile)
                if context is None:
                    context = create_sipm_geometry_context(geofile)
                    geometry_contexts[geofile] = context
                plane_pdf = os.path.join(event_workdir, "scifi_planes.pdf")
                title = (
                    f"{event.get('particle_label', args.particle)} — "
                    f"{event.get('region_label', args.region)} — "
                    f"Run {event.get('runId', '')} / Event {event.get('eventId', '')}"
                )
                scifi_audit = render_plane_pages(
                    event_source, tree_name, int(event["source_event_index"]), geofile,
                    plane_pdf, title, plane_page_width=plane_width,
                    geometry_context=context,
                )
                if scifi_audit["unmapped_hits"]:
                    raise RuntimeError(
                        f"Event {sequence} has {len(scifi_audit['unmapped_hits'])} unmapped SciFi hits"
                    )
                document_parts.extend((overview_pdf, plane_pdf))
                summaries.append({
                    **event,
                    "event_sequence": sequence,
                    "first_document_page": 3 * sequence - 2,
                    "output_pdf": str(output_pdf),
                    "total_recorded_scifi_hits": scifi_audit["total_recorded_scifi_hits"],
                    "total_valid_scifi_hits": scifi_audit["total_valid_scifi_hits"],
                })
                audits.append({"event_sequence": sequence, **scifi_audit})

            scratch_pdf = os.path.join(workdir, "complete.pdf")
            merge_pages_to_pdf(document_parts, scratch_pdf)
            # All intermediate output remains in Condor's local scratch area.
            # Exclusive creation prevents this job from overwriting a PDF
            # already present at the declared AFS destination.
            with open(scratch_pdf, "rb") as source:
                with open(output_pdf, "xb") as destination:
                    destination_created = True
                    shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
                    destination.flush()
                    os.fsync(destination.fileno())
            source_size = os.path.getsize(scratch_pdf)
            destination_size = output_pdf.stat().st_size
            if destination_size != source_size:
                raise OSError(
                    f"Incomplete output copy: source={source_size} bytes, "
                    f"destination={destination_size} bytes"
                )
            copy_succeeded = True
    finally:
        if destination_created and not copy_succeeded and output_pdf.exists():
            output_pdf.unlink()

    write_csv(summaries, args.output_summary)
    audit_output = Path(args.output_audit)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps({
        "config": str(Path(args.config).resolve()),
        "region": args.region,
        "particle_group": args.particle,
        "events": audits,
        "event_count": len(audits),
        "pages_per_event": 3,
        "output_pdf": str(output_pdf),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_pdf} ({3 * len(events)} pages)")


if __name__ == "__main__":
    main()
