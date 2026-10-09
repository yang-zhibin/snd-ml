#!/usr/bin/env python3
"""Write a sparse cutFlowSummary whose entry branch selects raw digi rows."""

from __future__ import annotations

import argparse
import array
import json
import sys
from pathlib import Path

import ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "snakemake/metadata"))
from prepare_unified_testbeam_2023_manifest import deterministic_entry_indices  # noqa: E402


def write_index(
    output_path: str | Path,
    summary_path: str | Path,
    source_uid: str,
    input_event_count: int,
    selected_event_count: int,
) -> dict:
    entries = deterministic_entry_indices(input_event_count, selected_event_count)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    root_file = ROOT.TFile(str(temporary), "RECREATE")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"could not create {temporary}")
    tree = ROOT.TTree("cutFlowSummary", "deterministic indexed selection")
    raw_entry = array.array("q", [0])
    tree.Branch("entry", raw_entry, "entry/L")
    for entry in entries:
        raw_entry[0] = entry
        tree.Fill()
    tree.Write()
    ROOT.TNamed("metadata_selection_policy", "evenly_spaced_entries_v1").Write()
    ROOT.TNamed("metadata_source_uid", str(source_uid)).Write()
    root_file.Close()
    temporary.replace(output)
    report = {
        "validation_passed": True,
        "selection_mode": "indexed_cutflow",
        "selection_policy": "evenly_spaced_entries_v1",
        "source_uid": str(source_uid),
        "input_event_count": int(input_event_count),
        "selected_event_count": int(selected_event_count),
        "first_entry": entries[0] if entries else None,
        "last_entry": entries[-1] if entries else None,
        "output_path": str(output),
    }
    summary = Path(summary_path)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary_tmp = summary.with_suffix(summary.suffix + ".tmp")
    summary_tmp.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_tmp.replace(summary)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--source-uid", required=True)
    parser.add_argument("--input-event-count", type=int, required=True)
    parser.add_argument("--selected-event-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_index(
        args.output,
        args.summary,
        args.source_uid,
        args.input_event_count,
        args.selected_event_count,
    )
    print(
        f"wrote {report['selected_event_count']}/"
        f"{report['input_event_count']} indexed entries"
    )


if __name__ == "__main__":
    main()
