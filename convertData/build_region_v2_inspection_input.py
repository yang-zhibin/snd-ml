#!/usr/bin/env python3
"""Build one deterministic, consolidated region-v2 inspection ROOT file."""
import argparse
import csv
import gc
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from region_v2_inspection_common import (
    AUDIT_FIELDS, MANIFEST_FIELDS, load_config, load_selected_rows,
    source_luminosity, target_luminosity,
)


def write_csv(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def branch_names(tree):
    return [branch.GetName() for branch in tree.GetListOfBranches()]


def verify_friend_alignment(tree, cut, path):
    """Check identifiers at deterministic boundary/midpoint entries per file."""
    identifiers = {"runId", "eventId"}
    if not identifiers.issubset(branch_names(tree)) or not identifiers.issubset(branch_names(cut)):
        raise RuntimeError(f"Cannot verify sndData/cutFlowSummary alignment in {path}: missing runId/eventId")
    count = int(tree.GetEntries())
    indices = sorted({0, count // 2, count - 1}) if count else []
    for index in indices:
        tree.GetEntry(index); cut.GetEntry(index)
        feature_id = (int(tree.runId), int(tree.eventId))
        cut_id = (int(cut.runId), int(cut.eventId))
        if feature_id != cut_id:
            raise RuntimeError(
                f"Friend alignment failure in {path} at entry {index}: {feature_id} != {cut_id}"
            )


def inspect_sources(ROOT, dataset_name, selected, tree_name, need_cutflow):
    audits, schema, total = [], None, 0
    file_count = len(selected)
    started = time.monotonic()
    report_every = max(1, file_count // 20)
    print(f"Inspecting {file_count} source files for {dataset_name}...", flush=True)
    for position, (_, row) in enumerate(selected.iterrows(), start=1):
        path = row["resolved_feature_path"]
        audit = {
            "dataset_name": dataset_name, "source_metadata_row": int(row["source_metadata_row"]),
            "resolved_feature_path": path, "partition": row.get("partition", ""),
            "run": row.get("run", row.get("run_number", "")),
            "lumi_per_file": row["lumi_per_file"], "file_exists": int(os.path.exists(path)),
            "file_opened": 0, "sndData_entries": -1, "cutFlowSummary_entries": -1,
        }
        if not audit["file_exists"]:
            audits.append(audit)
            continue
        root_file = ROOT.TFile.Open(path, "READ")
        if not root_file or root_file.IsZombie():
            audits.append(audit)
            continue
        audit["file_opened"] = 1
        tree = root_file.Get(tree_name)
        cut = root_file.Get("cutFlowSummary") if need_cutflow else None
        if not tree:
            root_file.Close()
            audits.append(audit)
            continue
        audit["sndData_entries"] = int(tree.GetEntries())
        audit["cutFlowSummary_entries"] = int(cut.GetEntries()) if cut else -1
        names = branch_names(tree)
        if schema is None:
            schema = names
        elif names != schema:
            raise RuntimeError(f"Branch schema differs in {path}")
        if need_cutflow:
            required = {"StableBeams", "IP1", "EventDeltat_1_100"}
            if not cut or audit["cutFlowSummary_entries"] != audit["sndData_entries"]:
                raise RuntimeError(f"sndData/cutFlowSummary entry mismatch in {path}")
            missing = required.difference(branch_names(cut))
            if missing:
                raise RuntimeError(f"cutFlowSummary in {path} lacks {sorted(missing)}")
            verify_friend_alignment(tree, cut, path)
        total += audit["sndData_entries"]
        root_file.Close()
        audits.append(audit)
        if position == file_count or position % report_every == 0:
            elapsed = time.monotonic() - started
            rate = position / elapsed if elapsed else 0.0
            eta = (file_count - position) / rate if rate else 0.0
            print(
                f"  files {position:>{len(str(file_count))}}/{file_count} "
                f"({100.0 * position / file_count:5.1f}%) | "
                f"elapsed {elapsed:7.1f}s | ETA {eta:7.1f}s",
                flush=True,
            )
    bad = [a for a in audits if not a["file_opened"] or a["sndData_entries"] < 0]
    if bad:
        paths = "\n".join(a["resolved_feature_path"] for a in bad)
        raise FileNotFoundError(f"{len(bad)} selected inputs are missing/unreadable; no substitution is allowed:\n{paths}")
    return audits, schema or [], total


def rdf_input_path(path):
    """Use ROOT's canonical EOS identity so per-sample metadata lookup is stable."""
    if path.startswith("/eos/"):
        return f"root://eospublic.cern.ch/{path}"
    return path


def is_noop_selection(expression):
    """Return true when a configured selection is an unconditional pass."""
    return str(expression).strip().lower() in {"true", "1", "1 == 1", "1==1"}


def make_dataframe(ROOT, selected, tree_name, need_cutflow):
    spec = ROOT.RDF.Experimental.RDatasetSpec()
    paths = []
    for _, row in selected.iterrows():
        path = rdf_input_path(row["resolved_feature_path"])
        paths.append(path)
        metadata = ROOT.RDF.Experimental.RMetaData()
        metadata.Add("source_row", int(row["source_metadata_row"]))
        sample = ROOT.RDF.Experimental.RSample(
            f"source_row_{int(row['source_metadata_row'])}", tree_name, path, metadata
        )
        spec.AddSample(sample)
    if need_cutflow:
        spec.WithGlobalFriends("cutFlowSummary", paths, "cut")
    return ROOT.RDataFrame(spec).DefinePerSample(
        "inspection_source_row", 'rdfsampleinfo_.GetI("source_row")'
    )


def write_embedded_metadata(ROOT, output, numbers, strings):
    root_file = ROOT.TFile.Open(output, "UPDATE")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Cannot reopen output {output}")
    for name, value in numbers.items():
        ROOT.TParameter("double")(name, float(value)).Write(name, ROOT.TObject.kOverwrite)
    for name, value in strings.items():
        ROOT.TNamed(name, str(value)).Write(name, ROOT.TObject.kOverwrite)
    root_file.Close()


def build(args):
    import ROOT
    job_started = time.monotonic()
    timings = {}

    def start_step(number, total, message):
        print(f"\n[{number}/{total}] {message}", flush=True)
        return time.monotonic()

    def finish_step(name, started):
        timings[name] = time.monotonic() - started
        print(f"    completed in {timings[name]:.1f}s", flush=True)

    step = start_step(1, 5, "Loading configuration and selecting metadata rows")
    ROOT.ROOT.EnableImplicitMT(args.threads)
    config = load_config(args.config)
    spec, metadata_path, selected = load_selected_rows(config, args.dataset)
    tree_name = config["output"].get("tree_name", "sndData")
    need_cutflow = spec["selection_mode"] == "inclusive_data_quality"
    print(
        f"    dataset={args.dataset}, selected_files={len(selected)}, "
        f"threads={args.threads}, cutflow_friend={'yes' if need_cutflow else 'no'}",
        flush=True,
    )
    finish_step("configuration", step)

    step = start_step(2, 5, "Inspecting source ROOT files and checking their schemas")
    audits, columns, input_count = inspect_sources(
        ROOT, args.dataset, selected, tree_name, need_cutflow
    )
    collision = {"inspection_event_weight", "inspection_source_row"}.intersection(columns)
    if collision:
        raise RuntimeError(f"Input already contains reserved branches: {sorted(collision)}")
    print(f"    found {input_count} input events and {len(columns)} branches", flush=True)
    finish_step("source inspection", step)

    step = start_step(3, 5, "Calculating luminosity normalization and preparing RDataFrame")
    source_lumi = source_luminosity(selected)
    target_lumi = target_luminosity(config)
    if source_lumi <= 0:
        raise ValueError(f"Non-positive source luminosity for {args.dataset}")
    weight = target_lumi / source_lumi
    selection = spec.get("event_selection", "true")
    particle_filter = spec.get("particle_filter", "true")
    apply_filter = not (is_noop_selection(selection) and is_noop_selection(particle_filter))
    expression = f"({selection}) && ({particle_filter})"
    rdf = make_dataframe(ROOT, selected, tree_name, need_cutflow)
    # DefinePerSample returns a templated RInterface that PyROOT cannot pass to
    # AddProgressBar directly. Convert it explicitly to the accepted RNode type.
    ROOT.RDF.Experimental.AddProgressBar(ROOT.RDF.AsRNode(rdf))
    print(
        f"    source_luminosity={source_lumi:.12g} fb^-1, "
        f"target_luminosity={target_lumi:.12g} fb^-1",
        flush=True,
    )
    print(f"    event_weight={weight:.12g}", flush=True)
    print(
        f"    selection={'none (all events retained)' if not apply_filter else expression}",
        flush=True,
    )
    finish_step("dataframe setup", step)

    action = (
        "Copying all events and writing the consolidated ROOT tree"
        if not apply_filter
        else "Applying event selection and writing the consolidated ROOT tree"
    )
    step = start_step(4, 5, action)
    print(
        f"Processing {input_count} events from {len(selected)} files "
        f"with {args.threads} thread(s)...",
        flush=True,
    )
    rdf = rdf.Define("inspection_event_weight", repr(weight))
    if apply_filter:
        rdf = rdf.Filter(expression)
    snapshot_columns = columns + ["inspection_event_weight", "inspection_source_row"]
    options = ROOT.RDF.RSnapshotOptions()
    options.fLazy = True
    options.fCompressionLevel = int(config["output"].get("compression_level", 1))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    snapshot = rdf.Snapshot(tree_name, args.output, snapshot_columns, options)
    count = rdf.Count()
    ROOT.RDF.RunGraphs([snapshot, count])
    output_count = int(count.GetValue())
    # RSnapshot holds the output TFile through its RResultPtr/loop manager.
    # On large snapshots PyROOT may otherwise keep that handle alive until
    # function exit. Reopening the same path in UPDATE mode below then makes
    # ROOT treat the still-open file as unclean and attempt recovery.
    del snapshot
    del count
    del rdf
    gc.collect()
    open_snapshot = ROOT.gROOT.GetListOfFiles().FindObject(args.output)
    if open_snapshot and open_snapshot.IsOpen():
        open_snapshot.Close()
    processing_elapsed = time.monotonic() - step
    input_rate = input_count / processing_elapsed if processing_elapsed else 0.0
    output_rate = output_count / processing_elapsed if processing_elapsed else 0.0
    efficiency = output_count / input_count if input_count else 0.0
    print(
        f"    selected {output_count}/{input_count} events ({efficiency:.2%}); "
        f"input rate={input_rate:.1f} events/s, output rate={output_rate:.1f} events/s",
        flush=True,
    )
    finish_step("event processing and snapshot", step)

    step = start_step(5, 5, "Embedding provenance and writing audit/manifest CSV files")
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    row_definition = "zero-based row index in source metadata CSV before selection"
    runs = spec.get("selected_runs", "")
    note = spec.get("normalization_note", "Normalize to selected inclusive-data luminosity")
    numbers = {
        "inspection_source_luminosity_fb": source_lumi,
        "inspection_target_luminosity_fb": target_lumi,
        "inspection_normalization_weight": weight,
        "inspection_selected_file_count": len(selected),
        "inspection_input_entry_count": input_count,
        "inspection_output_entry_count": output_count,
        "inspection_mc_fraction": spec.get("mc_fraction", 1.0),
    }
    strings = {
        "inspection_version": config["version"], "inspection_dataset_name": args.dataset,
        "inspection_data_type": spec["data_type"],
        "inspection_source_metadata_csv": str(metadata_path),
        "inspection_source_row_definition": row_definition,
        "inspection_event_selection": selection, "inspection_particle_filter": particle_filter,
        "inspection_selected_runs": runs, "inspection_normalization_note": note,
        "inspection_production_timestamp_utc": timestamp,
    }
    write_embedded_metadata(ROOT, args.output, numbers, strings)
    manifest = {
        "inspection_version": config["version"], "dataset_name": args.dataset,
        "data_type": spec["data_type"], "output_path": args.published_path or args.output,
        "tree_name": tree_name, "source_metadata_csv": str(metadata_path),
        "source_row_definition": row_definition, "selected_file_count": len(selected),
        "missing_file_count": 0, "selected_runs": runs, "event_selection": selection,
        "particle_filter": particle_filter, "source_luminosity_fb": source_lumi,
        "target_luminosity_fb": target_lumi, "normalization_weight": weight,
        "input_entry_count": input_count, "output_entry_count": output_count,
        "mc_fraction": spec.get("mc_fraction", 1.0), "production_timestamp_utc": timestamp,
    }
    write_csv(args.manifest_row, MANIFEST_FIELDS, [manifest])
    write_csv(args.source_audit, AUDIT_FIELDS, audits)
    finish_step("metadata and audit output", step)
    total_elapsed = time.monotonic() - job_started
    print("\nTiming summary:", flush=True)
    for name, elapsed in timings.items():
        fraction = 100.0 * elapsed / total_elapsed if total_elapsed else 0.0
        print(f"  {name:<30} {elapsed:8.1f}s  ({fraction:5.1f}%)", flush=True)
    print(f"  {'total':<30} {total_elapsed:8.1f}s  (100.0%)", flush=True)
    print(
        f"Built {args.dataset}: {len(selected)} files, {output_count}/{input_count} events, "
        f"weight={weight:.12g}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--published-path")
    parser.add_argument("--manifest-row", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    try:
        build(parse_args())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
