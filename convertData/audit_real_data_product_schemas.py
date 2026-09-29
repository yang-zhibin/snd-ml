#!/usr/bin/env python3
"""Audit all real-data feature/hit3D files against trusted schemas."""
import argparse
import csv
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path


PRODUCTS = {
    "feature": ("feature_path", "sndData"),
    "hit3d": ("hit3d_path", "hit3D"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--feature-reference", required=True)
    parser.add_argument("--hit3d-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def inspect(ROOT, path, tree_name):
    if not os.path.exists(path):
        return "missing", -1, (), ""
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        return "unreadable", -1, (), ""
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        return "missing_tree", -1, (), ""
    # Sorting ignores harmless branch-order differences; name/class/leaf title
    # together retain the branch's effective ROOT type.
    schema = tuple(sorted(
        (b.GetName(), b.GetClassName(), b.GetTitle())
        for b in tree.GetListOfBranches()
    ))
    entries = int(tree.GetEntries())
    root_file.Close()
    digest = hashlib.sha256(repr(schema).encode()).hexdigest()[:16]
    return "readable", entries, schema, digest


def resolve(row, column):
    value = row[column]
    if value.startswith("/") or value.startswith("root://"):
        return value
    return row["output_base_path"].rstrip("/") + "/" + value.lstrip("/")


def main():
    import ROOT
    args = parse_args()
    references = {
        "feature": args.feature_reference,
        "hit3d": args.hit3d_reference,
    }
    expected = {}
    for product, (_, tree_name) in PRODUCTS.items():
        status, _, schema, digest = inspect(ROOT, references[product], tree_name)
        if status != "readable":
            raise RuntimeError(f"Bad {product} reference ({status}): {references[product]}")
        expected[product] = schema
        print(f"{product} reference: {len(schema)} branches, schema={digest}")

    with open(args.metadata, newline="", encoding="utf-8") as handle:
        metadata = list(csv.DictReader(handle))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for product, (column, tree_name) in PRODUCTS.items():
        report = []
        stale = []
        for index, row in enumerate(metadata, 1):
            path = resolve(row, column)
            status, entries, schema, digest = inspect(ROOT, path, tree_name)
            classification = (
                "current" if status == "readable" and schema == expected[product]
                else "different_schema" if status == "readable"
                else status
            )
            if classification != "current":
                stale.append(path)
            stat = os.stat(path) if os.path.exists(path) else None
            report.append({
                "product": product,
                "metadata_row": index - 1,
                "partition": row.get("partition", ""),
                "run_number": row.get("run_number", row.get("run", "")),
                "path": path,
                "classification": classification,
                "entries": entries,
                "branch_count": len(schema),
                "schema_fingerprint": digest,
                "size_bytes": stat.st_size if stat else -1,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else "",
                "audit_utc": generated,
            })
            if index % args.progress_every == 0 or index == len(metadata):
                print(f"{product}: {index}/{len(metadata)}", flush=True)

        fields = list(report[0])
        with open(outdir / f"real_data_{product}_schema_audit.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(report)
        with open(outdir / f"real_data_{product}_rerun_targets.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(stale) + ("\n" if stale else ""))
        counts = {name: sum(r["classification"] == name for r in report) for name in sorted({r["classification"] for r in report})}
        print(f"{product} summary: {counts}; rerun targets={len(stale)}")


if __name__ == "__main__":
    main()
