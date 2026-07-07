#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import ROOT
import yaml


ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def compact_region_name(region: str, region_label: str | None = None) -> str:
    label = str(region_label or "")
    match = re.match(r"^(SB\d+)\b", label)
    if match:
        return match.group(1)
    if str(region) == "signal_no_veto_has_us_fiducial_inside":
        return "Signal"
    return safe_name(region)


def is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_list_cell(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def resolve_path(path: object, repo_root: str | Path) -> str:
    text = str(path or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if text.startswith("root://") or os.path.isabs(text):
        return text
    return str(Path(repo_root) / text)


def resolve_row_path(row: dict, column: str, repo_root: str | Path) -> str:
    value = str(row.get(column, "") or "").strip()
    if not value or value.lower() == "nan":
        return ""
    if value.startswith("root://") or os.path.isabs(value):
        return value

    base = str(row.get("output_base_path", "") or "").strip()
    if base:
        return f"{base.rstrip('/')}/{value.lstrip('/')}"
    return resolve_path(value, repo_root)


def normalized_path(path: str) -> str:
    if path.startswith("root://"):
        return path
    return os.path.normpath(path)


def read_csv_rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_region_metadata(metadata_csv: str, repo_root: str | Path) -> list[dict]:
    rows = read_csv_rows(resolve_path(metadata_csv, repo_root))
    return [row for row in rows if is_truthy(row.get("feature_available", True))]


def load_eval_plot_config(config: dict, repo_root: str | Path) -> dict:
    eval_config_path = str((config.get("input", {}) or {}).get("eval_config", "") or "").strip()
    if not eval_config_path:
        eval_config_path = "snakemake/metadata/configs/region_evaluation/evaluation_options_v1.yaml"

    resolved = resolve_path(eval_config_path, repo_root)
    if not os.path.exists(resolved):
        print(f"[warn] evaluation config not found, region labels will use raw names: {resolved}")
        return {}
    return (load_yaml(resolved).get("plot", {}) or {})


def enrich_display_labels(events: dict[tuple[str, str], list[dict]], plot_cfg: dict) -> None:
    region_labels = plot_cfg.get("region_labels", {}) or {}
    particle_labels = plot_cfg.get("particle_labels", {}) or {}
    for (region, particle), group_events in events.items():
        region_label = str(region_labels.get(region, region))
        region_short = compact_region_name(region, region_label)
        particle_label = str(particle_labels.get(particle, particle))
        for event in group_events:
            event["region_label"] = region_label
            event["region_short"] = region_short
            event["particle_label"] = particle_label


def selected_metadata_rows(rows: list[dict], config: dict) -> list[dict]:
    filters = config.get("filters", {}) or {}
    configured_regions = filters.get("regions", []) or []
    configured_particles = filters.get("particles", []) or []
    if isinstance(configured_regions, str):
        configured_regions = [configured_regions]
    if isinstance(configured_particles, str):
        configured_particles = [configured_particles]
    regions = set(str(item) for item in configured_regions)
    particles = set(str(item) for item in configured_particles)

    out = []
    for row in rows:
        if regions and str(row.get("region", "")) not in regions:
            continue
        if particles and str(row.get("particle_group", "")) not in particles:
            continue
        out.append(row)
    return out


def get_limit(particle_group: str, config: dict) -> int | None:
    limits = config.get("limits", {}) or {}
    per_particle = limits.get("per_particle", {}) or {}
    value = per_particle.get(particle_group, limits.get("default_max_events_per_region_particle", 2))
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"all", "none", "unlimited", "-1"}:
        return None
    parsed = int(value)
    return None if parsed < 0 else parsed


def get_branch_value(tree, branch_name: str):
    if not branch_name or not hasattr(tree, branch_name):
        return None
    value = getattr(tree, branch_name)
    try:
        if hasattr(value, "size") and not isinstance(value, (str, bytes)):
            return [value.at(i) for i in range(value.size())]
    except Exception:
        pass
    return value


def format_value(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def tree_entries(path: str, tree_name: str) -> int:
    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open ROOT file: {path}")
    try:
        tree = root_file.Get(tree_name)
        if not tree:
            raise KeyError(f"Tree {tree_name!r} not found in {path}")
        return int(tree.GetEntries())
    finally:
        root_file.Close()


class SourceFeatureMapper:
    def __init__(self, source_paths: list[str], tree_name: str):
        self.source_paths = list(source_paths)
        self.tree_name = tree_name
        self._ranges: list[tuple[int, int, str]] = []

    def path_for_original_entry(self, original_entry: int) -> str | None:
        if not self.source_paths:
            return None
        if len(self.source_paths) == 1:
            return self.source_paths[0]

        while len(self._ranges) < len(self.source_paths) and (
            not self._ranges or self._ranges[-1][1] <= int(original_entry)
        ):
            start = self._ranges[-1][1] if self._ranges else 0
            path = self.source_paths[len(self._ranges)]
            entries = tree_entries(path, self.tree_name)
            self._ranges.append((start, start + entries, path))
            if len(self._ranges) >= len(self.source_paths):
                break

        for start, end, path in self._ranges:
            if start <= int(original_entry) < end:
                return path
        return None


def build_source_metadata_index(region_rows: list[dict], repo_root: str | Path) -> dict[str, dict]:
    metadata_paths = []
    seen = set()
    for row in region_rows:
        for metadata_path in split_list_cell(row.get("source_metadata_csv")):
            resolved = resolve_path(metadata_path, repo_root)
            if resolved and resolved not in seen:
                seen.add(resolved)
                metadata_paths.append(resolved)

    index = {}
    for metadata_path in metadata_paths:
        if not os.path.exists(metadata_path):
            print(f"[warn] source metadata CSV not found: {metadata_path}")
            continue
        for row in read_csv_rows(metadata_path):
            feature_path = resolve_row_path(row, "feature_path", repo_root)
            if not feature_path:
                continue
            item = dict(row)
            item["__metadata_csv"] = metadata_path
            item["__feature_path"] = feature_path
            index[normalized_path(feature_path)] = item
    return index


def feature_selection_entries(tree, selection: str, elist_name: str) -> list[int]:
    selection = str(selection or "").strip()
    if not selection or selection == "1":
        return list(range(int(tree.GetEntries())))

    tree.Draw(f">>{elist_name}", selection, "entrylist")
    elist = ROOT.gDirectory.Get(elist_name)
    if not elist:
        return []
    return [int(elist.GetEntry(i)) for i in range(int(elist.GetN()))]


def collect_partition_events(
    partition_row: dict,
    source_index: dict[str, dict],
    config: dict,
    repo_root: str | Path,
    remaining: int | None,
) -> list[dict]:
    feature_tree_name = config.get("input", {}).get("feature_tree", "sndData")
    selection = str((config.get("selection", {}) or {}).get("expression", "") or "").strip()
    annotation_features = list((config.get("annotations", {}) or {}).get("features", []) or [])

    feature_path = resolve_path(partition_row.get("feature_partition_path", ""), repo_root)
    if not feature_path:
        return []

    source_paths = [
        resolve_path(path, repo_root)
        for path in split_list_cell(partition_row.get("source_feature_paths"))
    ]
    mapper = SourceFeatureMapper(source_paths, feature_tree_name)

    root_file = ROOT.TFile.Open(feature_path)
    if not root_file or root_file.IsZombie():
        print(f"[skip] could not open feature partition: {feature_path}")
        return []

    tree = root_file.Get(feature_tree_name)
    if not tree:
        print(f"[skip] no {feature_tree_name} tree in {feature_path}")
        root_file.Close()
        return []

    entries = feature_selection_entries(
        tree,
        selection,
        f"elist_{safe_name(partition_row.get('partition_id'))}_{os.getpid()}",
    )
    selected = []
    for feature_entry in entries:
        if remaining is not None and len(selected) >= remaining:
            break
        tree.GetEntry(feature_entry)

        original_entry_value = get_branch_value(tree, "original_entry")
        if original_entry_value is None:
            original_entry_value = feature_entry
        original_entry = int(original_entry_value)
        source_feature_path = mapper.path_for_original_entry(original_entry)
        if not source_feature_path:
            print(
                "[skip] could not map original_entry "
                f"{original_entry} in {partition_row.get('partition_id')}"
            )
            continue

        source_row = source_index.get(normalized_path(source_feature_path))
        if not source_row:
            print(f"[skip] no source metadata row for feature path: {source_feature_path}")
            continue

        event_index = get_branch_value(tree, "eventIndex")
        if event_index is None:
            print(f"[skip] feature entry has no eventIndex: {feature_path}:{feature_entry}")
            continue

        features = {}
        for feature in annotation_features:
            features[feature] = get_branch_value(tree, feature)

        selected.append(
            {
                "region": partition_row.get("region", ""),
                "particle_group": partition_row.get("particle_group", ""),
                "particle_family": partition_row.get("particle_family", ""),
                "partition_id": partition_row.get("partition_id", ""),
                "feature_partition_path": feature_path,
                "feature_entry": int(feature_entry),
                "original_entry": original_entry,
                "source_feature_path": source_feature_path,
                "source_metadata_csv": source_row.get("__metadata_csv", ""),
                "source_metadata_row": source_row,
                "source_event_index": int(event_index),
                "runId": get_branch_value(tree, "runId"),
                "eventId": get_branch_value(tree, "eventId"),
                "features": features,
                "selection_label": (config.get("selection", {}) or {}).get("label", selection),
                "selection_expression": selection,
            }
        )

    root_file.Close()
    return selected


def collect_events(config: dict, metadata_csv: str, repo_root: str | Path) -> dict[tuple[str, str], list[dict]]:
    rows = selected_metadata_rows(load_region_metadata(metadata_csv, repo_root), config)
    source_index = build_source_metadata_index(rows, repo_root)

    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[(str(row.get("region", "")), str(row.get("particle_group", "")))].append(row)

    grouped_events = {}
    for (region, particle), partition_rows in grouped_rows.items():
        limit = get_limit(particle, config)
        collected = []
        print(f"[select] {region} / {particle}: limit={limit if limit is not None else 'all'}")
        for partition_row in partition_rows:
            remaining = None if limit is None else max(0, limit - len(collected))
            if remaining == 0:
                break
            collected.extend(
                collect_partition_events(
                    partition_row=partition_row,
                    source_index=source_index,
                    config=config,
                    repo_root=repo_root,
                    remaining=remaining,
                )
            )
        grouped_events[(region, particle)] = collected
        print(f"[select] {region} / {particle}: selected={len(collected)}")
    return grouped_events


def open_event_tree(root_file, tree_candidates: list[str]):
    for tree_name in tree_candidates:
        tree = root_file.Get(tree_name)
        if tree:
            return tree_name, tree
    return None, None


def event_tree_is_displayable(path: str, tree_candidates: list[str]) -> tuple[str | None, int]:
    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        return None, 0
    tree_name, tree = open_event_tree(root_file, tree_candidates)
    if not tree:
        root_file.Close()
        return None, 0
    if not (tree.GetBranch("EventHeader") or tree.GetBranch("EventHeader.")):
        root_file.Close()
        return None, 0
    entries = int(tree.GetEntries())
    root_file.Close()
    return tree_name, entries


def resolve_display_source(event: dict, config: dict, repo_root: str | Path) -> tuple[str, str, str]:
    source_row = event["source_metadata_row"]
    input_cfg = config.get("input", {}) or {}
    display_cfg = config.get("display", {}) or {}
    tree_candidates = list(input_cfg.get("event_tree_candidates", ["cbmsim", "rawConv"]))
    source_columns = list(input_cfg.get("event_source_columns", ["digi_path", "nueAnalysisFilter_path"]))
    configured_geofile = str(display_cfg.get("geofile", "") or "").strip()

    for column in source_columns:
        path = resolve_row_path(source_row, column, repo_root)
        if not path:
            continue
        tree_name, entries = event_tree_is_displayable(path, tree_candidates)
        if not tree_name:
            print(f"[skip] {column} is not displayable for {event['partition_id']}: {path}")
            continue
        if event["source_event_index"] >= entries:
            print(
                f"[skip] eventIndex {event['source_event_index']} out of range for "
                f"{column} entries={entries}: {path}"
            )
            continue
        geofile = configured_geofile or resolve_row_path(source_row, "geo_path", repo_root)
        if not geofile:
            raise RuntimeError(f"No geofile for source row from {source_row.get('__metadata_csv', '')}")
        return path, tree_name, geofile

    raise RuntimeError(
        "Could not find a displayable event source for "
        f"{event['region']} / {event['particle_group']} / {event['partition_id']}"
    )


def find_sndsw_2d_display_script(script_path: str | None) -> str:
    if script_path:
        return script_path

    sndsw_root = os.environ.get("SNDSW_ROOT")
    if sndsw_root:
        candidate = os.path.join(sndsw_root, "shipLHC", "scripts", "2dEventDisplay.py")
        if os.path.exists(candidate):
            return candidate

    for base in Path(__file__).resolve().parents:
        local_candidate = (
            base
            / "public"
            / "SndBuild"
            / "sndsw"
            / "shipLHC"
            / "scripts"
            / "2dEventDisplay.py"
        )
        if local_candidate.exists():
            return str(local_candidate)

    raise RuntimeError("Could not find SNDSW 2dEventDisplay.py; pass --sndsw-display-script or set it in config")


def make_branch_list(tree) -> ROOT.TList:
    names = set()
    for branch in tree.GetListOfBranches():
        branch_name = branch.GetName()
        names.add(branch_name)
        if branch_name.endswith("."):
            names.add(branch_name.rstrip("."))

    if tree.GetName() == "rawConv":
        names.update(["sndScifiHit", "MuFilterHit", "SNDLHCEventHeader"])
    else:
        names.update([
            "MCTrack",
            "vetoPoint",
            "EmulsionDetPoint",
            "ScifiPoint",
            "MuFilterPoint",
            "SNDLHCEventHeader",
            "MCEventHeader.",
        ])

    branch_list = ROOT.TList()
    branch_list.SetName("BranchList")
    for name in sorted(names):
        branch_list.Add(ROOT.TObjString(name))
    return branch_list


def write_fairroot_bookkeeping(input_file, output_file, tree):
    branch_list = input_file.Get("BranchList")
    if branch_list:
        branch_list.Write("BranchList", ROOT.TObject.kSingleKey)
    else:
        make_branch_list(tree).Write("BranchList", ROOT.TObject.kSingleKey)

    timebased_branch_list = input_file.Get("TimeBasedBranchList")
    if timebased_branch_list:
        timebased_branch_list.Write("TimeBasedBranchList", ROOT.TObject.kSingleKey)
    else:
        empty_list = ROOT.TList()
        empty_list.SetName("TimeBasedBranchList")
        empty_list.Write("TimeBasedBranchList", ROOT.TObject.kSingleKey)

    for header_name in ("FileHeader", "FileHeaderHeader"):
        header = input_file.Get(header_name)
        if header:
            header.Write(header_name, ROOT.TObject.kSingleKey)

    if not input_file.Get("FileHeader") and hasattr(ROOT, "FairFileHeader"):
        file_header = ROOT.FairFileHeader()
        file_header.SetTitle("FileHeader")
        file_header.Write("FileHeader", ROOT.TObject.kSingleKey)

    _ = output_file


def prepare_selected_display_file(
    input_file_path: str,
    output_file_path: str,
    tree_name: str,
    source_event_indices: list[int],
):
    input_file = ROOT.TFile.Open(input_file_path)
    if not input_file or input_file.IsZombie():
        raise RuntimeError(f"Could not open event source file: {input_file_path}")

    input_tree = input_file.Get(tree_name)
    if not input_tree:
        input_file.Close()
        raise RuntimeError(f"Could not find tree {tree_name} in {input_file_path}")

    output_file = ROOT.TFile.Open(output_file_path, "RECREATE")
    if not output_file or output_file.IsZombie():
        input_file.Close()
        raise RuntimeError(f"Could not create temporary display file: {output_file_path}")

    output_file.cd()
    write_fairroot_bookkeeping(input_file, output_file, input_tree)
    copied_tree = input_tree.CloneTree(0)
    for source_event_index in source_event_indices:
        if source_event_index < 0 or source_event_index >= input_tree.GetEntries():
            print(f"[skip] source event index out of range: {source_event_index} in {input_file_path}")
            continue
        input_tree.GetEntry(int(source_event_index))
        copied_tree.Fill()
    copied_tree.Write("", ROOT.TObject.kOverwrite)

    output_file.Close()
    input_file.Close()


def latex_safe(value: object) -> str:
    text = str(value)
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def annotation_lines(event: dict, max_lines: int) -> list[str]:
    lines = [
        f"Region: {event.get('region_label', event['region'])}",
        f"Particle: {event.get('particle_label', event['particle_group'])}",
    ]

    feature_parts = [
        f"{name}={format_value(value)}"
        for name, value in (event.get("features") or {}).items()
        if value is not None
    ]
    while feature_parts and len(lines) < max_lines:
        chunk = ", ".join(feature_parts[:3])
        lines.append(chunk)
        feature_parts = feature_parts[3:]
    return [latex_safe(line) for line in lines[:max_lines]]


def write_sndsw_runner(
    runner_path: str,
    display_script: str,
    input_file: str,
    geofile: str,
    output_prefix: str,
    event_indices: list[int],
    output_extension: str,
    annotations: list[dict],
    canvas_width: int,
    canvas_height: int,
):
    annotation_payload = [
        {"lines": annotation_lines(event, int(event.get("max_annotation_lines", 8)))}
        for event in annotations
    ]
    runner = f"""#!/usr/bin/env python3
import runpy
import sys

sys.argv = [
    {display_script!r},
    "-f", {input_file!r},
    "-p", "",
    "-g", {geofile!r},
    "--rootbatch",
    "--extension", {output_extension!r},
]

event_annotations = {annotation_payload!r}
namespace = runpy.run_path({display_script!r}, run_name="__main__")
namespace["ROOT"].gStyle.SetPaperSize(32.0, 20.0)
namespace["options"].storePic = {output_prefix!r}
namespace["ut"].bookCanvas(
    namespace["h"],
    key="simpleDisplay",
    title="simple event display",
    nx={int(canvas_width)},
    ny={int(canvas_height)},
    cx=1,
    cy=2,
)
namespace["h"]["simpleDisplay"].SetCanvasSize({int(canvas_width)}, {int(canvas_height)})

def drawInfo_region_partition(pad, k, run, event, timestamp, moreEventInfo=[]):
    ROOT = namespace["ROOT"]
    eventTree = namespace["eventTree"]
    mc = namespace["mc"]
    datetime = namespace["datetime"]

    timestamp_print = False
    if not mc and hasattr(eventTree.EventHeader, "GetUTCtimestamp"):
        timestamp_print = True
        time_event = datetime.utcfromtimestamp(eventTree.EventHeader.GetUTCtimestamp())

    pad_text = ROOT.TPad("region_info", "region_info", 0.16, 0.03, 0.94, 0.36)
    pad_text.SetFillStyle(4000)
    pad_text.Draw()
    pad_text.cd()
    text = ROOT.TLatex()
    text.SetTextAlign(11)
    text.SetTextFont(42)
    text.SetTextSize(0.090)

    event_number = event
    if hasattr(eventTree.EventHeader, "GetEventNumber"):
        try:
            event_number = eventTree.EventHeader.GetEventNumber()
        except Exception:
            event_number = event

    lines = ["SND@LHC Experiment, CERN", "Run / Event: " + str(run) + " / " + str(event_number)]
    if timestamp_print:
        lines.append("Time (GMT): " + str(time_event))

    try:
        annotation = event_annotations[int(event)]
        lines.extend(annotation.get("lines", []))
    except Exception:
        pass

    y = 0.92
    dy = 0.105 if len(lines) > 6 else 0.13
    for line in lines[:10]:
        text.DrawLatex(0.0, y, str(line))
        y -= dy
    pad.cd(k)

namespace["drawInfo"] = drawInfo_region_partition
namespace["loopEvents"].__globals__["drawInfo"] = drawInfo_region_partition
namespace["loopEvents"](
    start={event_indices!r},
    auto=True,
    save=False,
    goodEvents=False,
    withTrack=0,
    withHoughTrack=0,
    nTracks=0,
    verbose=0,
    hitColour="q",
)
"""
    with open(runner_path, "w", encoding="utf-8") as handle:
        handle.write(runner)


def merge_pngs_to_pdf(input_pngs: list[str], output_pdf: str):
    if not input_pngs:
        raise RuntimeError(f"No PNG pages to merge for {output_pdf}")

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(output_pdf) as pdf:
            for png_path in input_pngs:
                image = plt.imread(png_path)
                height = image.shape[0]
                width = image.shape[1]
                dpi = 100.0
                fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(image)
                ax.axis("off")
                pdf.savefig(fig, dpi=dpi)
                plt.close(fig)
        return
    except Exception as error:
        print(f"[warn] matplotlib PNG-to-PDF conversion failed: {error}")

    convert = shutil.which("convert")
    if convert:
        subprocess.run([convert, *input_pngs, output_pdf], check=True)
        return

    raise RuntimeError("Could not convert PNG event displays to PDF")


def merge_pages_to_pdf(input_pages: list[str], output_pdf: str):
    if not input_pages:
        raise RuntimeError(f"No pages to merge for {output_pdf}")

    extensions = {os.path.splitext(page)[1].lower() for page in input_pages}
    if extensions == {".png"}:
        merge_pngs_to_pdf(input_pages, output_pdf)
        return

    if extensions in ({".ps"}, {".eps"}):
        ps2pdf = shutil.which("ps2pdf")
        if not ps2pdf:
            raise RuntimeError("Could not convert PostScript/EPS pages to PDF: ps2pdf not found")
        with tempfile.TemporaryDirectory(prefix="region_event_ps2pdf_") as pdf_workdir:
            converted_pages = []
            crop_option = ["-dEPSCrop"] if extensions == {".eps"} else []
            for index, page in enumerate(input_pages):
                pdf_page = os.path.join(pdf_workdir, f"page_{index:04d}.pdf")
                subprocess.run([ps2pdf, *crop_option, page, pdf_page], check=True)
                converted_pages.append(pdf_page)
            merge_pages_to_pdf(converted_pages, output_pdf)
        return

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    pdfunite = shutil.which("pdfunite")
    if pdfunite and extensions == {".pdf"}:
        subprocess.run([pdfunite, *input_pages, output_pdf], check=True)
        return

    ghostscript = shutil.which("gs")
    if ghostscript and extensions == {".pdf"}:
        subprocess.run(
            [
                ghostscript,
                "-dBATCH",
                "-dNOPAUSE",
                "-q",
                "-sDEVICE=pdfwrite",
                f"-sOutputFile={output_pdf}",
                *input_pages,
            ],
            check=True,
        )
        return

    raise RuntimeError(f"Could not merge pages with extensions: {sorted(extensions)}")


def run_sndsw_for_batch(
    events: list[dict],
    display_script: str,
    workdir: str,
    output_extension: str,
    config: dict,
) -> list[str]:
    if not events:
        return []

    first = events[0]
    event_source = first["event_source_path"]
    tree_name = first["event_tree_name"]
    geofile = first["geofile"]

    batch_name = safe_name(f"{first['region']}__{first['particle_group']}__{len(events)}__{abs(hash(event_source))}")
    display_input = os.path.join(workdir, f"display_input_{batch_name}.root")
    prepare_selected_display_file(
        input_file_path=event_source,
        output_file_path=display_input,
        tree_name=tree_name,
        source_event_indices=[event["source_event_index"] for event in events],
    )

    prefix = os.path.join(workdir, f"{batch_name}__")
    runner_path = os.path.join(workdir, f"run_{batch_name}.py")
    display_cfg = config.get("display", {}) or {}
    annotation_cfg = config.get("annotations", {}) or {}
    for event in events:
        event["max_annotation_lines"] = int(annotation_cfg.get("max_lines", 8))

    write_sndsw_runner(
        runner_path=runner_path,
        display_script=display_script,
        input_file=display_input,
        geofile=geofile,
        output_prefix=prefix,
        event_indices=list(range(len(events))),
        output_extension=output_extension,
        annotations=events,
        canvas_width=int(display_cfg.get("canvas_width", 1600)),
        canvas_height=int(display_cfg.get("canvas_height", 1000)),
    )

    print(
        f"[sndsw] {first['region']} / {first['particle_group']} "
        f"source={os.path.basename(event_source)} draw={len(events)}"
    )
    completed = subprocess.run([sys.executable, runner_path], cwd=workdir, check=False)
    page_files = sorted(glob.glob(prefix + f"*.{output_extension}"))
    if completed.returncode != 0:
        if completed.returncode == -signal.SIGTERM and page_files:
            print("[sndsw] 2dEventDisplay.py exited with SIGTERM after writing pages; accepting output")
        else:
            raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return page_files


def write_summary_csv(summary_rows: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "region",
        "region_label",
        "region_short",
        "particle_group",
        "particle_label",
        "partition_id",
        "feature_partition_path",
        "feature_entry",
        "original_entry",
        "source_feature_path",
        "event_source_path",
        "event_tree_name",
        "source_event_index",
        "runId",
        "eventId",
        "geofile",
        "output_pdf",
        "selection_expression",
        "feature_values",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def plot_group(
    region: str,
    particle: str,
    events: list[dict],
    outdir: str,
    config: dict,
    display_script: str,
    repo_root: str | Path,
) -> list[dict]:
    if not events:
        return []

    display_cfg = config.get("display", {}) or {}
    output_mode = display_cfg.get("output_mode", "vector_pdf")
    output_extension = "eps" if output_mode == "vector_pdf" else "png"

    resolved_events = []
    for event in events:
        if event.get("event_source_path") and event.get("event_tree_name") and event.get("geofile"):
            event_source = event["event_source_path"]
            tree_name = event["event_tree_name"]
            geofile = event["geofile"]
        else:
            try:
                event_source, tree_name, geofile = resolve_display_source(event, config, repo_root)
            except Exception as error:
                print(f"[skip] {error}")
                continue
            event["event_source_path"] = event_source
            event["event_tree_name"] = tree_name
            event["geofile"] = geofile
        resolved_events.append(event)

    if not resolved_events:
        return []

    region_short = resolved_events[0].get("region_short", compact_region_name(region))
    output_pdf = os.path.join(
        outdir,
        f"events__{safe_name(region_short)}__{safe_name(region)}__{safe_name(particle)}.pdf",
    )
    page_files = []
    with tempfile.TemporaryDirectory(prefix=f"region_events_{safe_name(region)}_{safe_name(particle)}_") as workdir:
        batches = defaultdict(list)
        for event in resolved_events:
            key = (event["event_source_path"], event["event_tree_name"], event["geofile"])
            batches[key].append(event)

        for batch_events in batches.values():
            page_files.extend(
                run_sndsw_for_batch(
                    events=batch_events,
                    display_script=display_script,
                    workdir=workdir,
                    output_extension=output_extension,
                    config=config,
                )
            )

        if page_files:
            merge_pages_to_pdf(page_files, output_pdf)

    summary_rows = []
    for event in resolved_events:
        feature_values = ";".join(
            f"{name}={format_value(value)}"
            for name, value in (event.get("features") or {}).items()
        )
        summary_rows.append(
            {
                "region": region,
                "region_label": event.get("region_label", ""),
                "region_short": event.get("region_short", ""),
                "particle_group": particle,
                "particle_label": event.get("particle_label", ""),
                "partition_id": event.get("partition_id", ""),
                "feature_partition_path": event.get("feature_partition_path", ""),
                "feature_entry": event.get("feature_entry", ""),
                "original_entry": event.get("original_entry", ""),
                "source_feature_path": event.get("source_feature_path", ""),
                "event_source_path": event.get("event_source_path", ""),
                "event_tree_name": event.get("event_tree_name", ""),
                "source_event_index": event.get("source_event_index", ""),
                "runId": event.get("runId", ""),
                "eventId": event.get("eventId", ""),
                "geofile": event.get("geofile", ""),
                "output_pdf": output_pdf if page_files else "",
                "selection_expression": event.get("selection_expression", ""),
                "feature_values": feature_values,
            }
        )

    if page_files:
        print(f"[saved] {output_pdf} pages={len(page_files)}")
    return summary_rows


def configured_follow_options(args, config: dict, outdir: str) -> dict:
    follow_cfg = dict((config.get("follow_event", {}) or {}))
    cli_requested = (
        args.follow_run is not None
        or args.follow_event is not None
        or args.follow_source_index is not None
        or args.follow_summary_csv is not None
        or args.n_following is not None
    )
    enabled = bool(follow_cfg.get("enabled", False)) or cli_requested
    if not enabled:
        return {"enabled": False}

    if args.follow_run is not None:
        follow_cfg["runId"] = args.follow_run
    if args.follow_event is not None:
        follow_cfg["eventId"] = args.follow_event
    if args.follow_source_index is not None:
        follow_cfg["source_event_index"] = args.follow_source_index
    if args.n_following is not None:
        follow_cfg["n_following"] = args.n_following
    if args.follow_summary_csv is not None:
        follow_cfg["summary_csv"] = args.follow_summary_csv

    if "n_following" not in follow_cfg:
        follow_cfg["n_following"] = 1
    if "summary_csv" not in follow_cfg:
        summary_name = (config.get("outputs", {}) or {}).get("summary_csv", "summary.csv")
        follow_cfg["summary_csv"] = os.path.join(outdir, summary_name)
    if "output_summary_csv" not in follow_cfg:
        follow_cfg["output_summary_csv"] = "follow_summary.csv"
    follow_cfg["enabled"] = True
    return follow_cfg


def _same_cell(row: dict, key: str, value: object) -> bool:
    if value is None:
        return True
    return str(row.get(key, "")).strip() == str(value).strip()


def find_follow_anchor(summary_rows: list[dict], follow_cfg: dict) -> dict:
    matches = []
    for row in summary_rows:
        if not _same_cell(row, "runId", follow_cfg.get("runId")):
            continue
        if not _same_cell(row, "eventId", follow_cfg.get("eventId")):
            continue
        if not _same_cell(row, "source_event_index", follow_cfg.get("source_event_index")):
            continue
        matches.append(row)

    if not matches:
        criteria = {
            key: follow_cfg.get(key)
            for key in ("runId", "eventId", "source_event_index")
            if follow_cfg.get(key) is not None
        }
        raise RuntimeError(f"Could not find follow-event anchor row in summary.csv for {criteria}")
    if len(matches) > 1:
        print(f"[warn] follow-event anchor matched {len(matches)} rows; using the first one")
    return matches[0]


def read_event_identity(path: str, tree_name: str, source_event_index: int) -> dict:
    root_file = ROOT.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        print(f"[warn] could not open event source for identity lookup: {path}")
        return {}
    try:
        tree = root_file.Get(tree_name)
        if not tree:
            print(f"[warn] tree {tree_name!r} not found for identity lookup: {path}")
            return {}
        if source_event_index < 0 or source_event_index >= int(tree.GetEntries()):
            print(f"[skip] follow-up source event index out of range: {source_event_index} in {path}")
            return {"out_of_range": True}
        tree.GetEntry(int(source_event_index))
        header = None
        for header_name in ("EventHeader", "SNDLHCEventHeader"):
            if not tree.GetBranch(header_name):
                continue
            try:
                header = getattr(tree, header_name)
                break
            except Exception as error:
                print(f"[warn] could not read {header_name} for identity lookup: {error}")
        if not header:
            return {}

        values = {}
        for output_key, methods in {
            "runId": ("GetRunId", "GetRunID", "GetRunNumber"),
            "eventId": ("GetEventNumber", "GetEventId", "GetEventID"),
        }.items():
            for method in methods:
                if hasattr(header, method):
                    try:
                        values[output_key] = getattr(header, method)()
                        break
                    except Exception:
                        pass
        return values
    finally:
        root_file.Close()


def find_feature_row_by_event_index(
    feature_path: str,
    tree_name: str,
    source_event_index: int,
    annotation_features: list[str],
) -> dict:
    if not feature_path:
        return {}
    root_file = ROOT.TFile.Open(feature_path)
    if not root_file or root_file.IsZombie():
        print(f"[warn] could not open source feature file for follow-up lookup: {feature_path}")
        return {}
    try:
        tree = root_file.Get(tree_name)
        if not tree:
            print(f"[warn] no {tree_name} tree in source feature file: {feature_path}")
            return {}
        entries = feature_selection_entries(
            tree,
            f"eventIndex == {int(source_event_index)}",
            f"elist_follow_{int(source_event_index)}_{os.getpid()}",
        )
        if not entries:
            print(f"[follow] no feature row found for eventIndex={source_event_index}")
            return {}

        feature_entry = int(entries[0])
        tree.GetEntry(feature_entry)
        features = {
            feature: get_branch_value(tree, feature)
            for feature in annotation_features
        }
        out = {
            "feature_entry": feature_entry,
            "features": features,
            "runId": get_branch_value(tree, "runId"),
            "eventId": get_branch_value(tree, "eventId"),
            "original_entry": get_branch_value(tree, "original_entry"),
        }
        return {key: value for key, value in out.items() if value is not None}
    finally:
        root_file.Close()


def build_follow_events(anchor: dict, follow_cfg: dict, config: dict, repo_root: str | Path) -> list[dict]:
    n_following = int(follow_cfg.get("n_following", 1))
    if n_following < 1:
        raise RuntimeError("n_following must be at least 1")

    event_source_path = resolve_path(anchor.get("event_source_path", ""), repo_root)
    event_tree_name = str(anchor.get("event_tree_name", "") or "").strip()
    geofile = resolve_path(anchor.get("geofile", ""), repo_root)
    anchor_index = int(anchor["source_event_index"])
    feature_tree_name = (config.get("input", {}) or {}).get("feature_tree", "sndData")
    annotation_features = list((config.get("annotations", {}) or {}).get("features", []) or [])
    source_feature_path = resolve_path(anchor.get("source_feature_path", ""), repo_root)

    if not event_source_path or not event_tree_name or not geofile:
        raise RuntimeError("Anchor row is missing event_source_path, event_tree_name, or geofile")

    anchor_region_label = anchor.get("region_label") or anchor.get("region") or "selected event"
    particle_group = anchor.get("particle_group") or "follow_event"
    particle_label = anchor.get("particle_label") or particle_group
    follow_region = f"follow_after_{anchor.get('region') or 'selected_event'}"
    follow_region_short = f"{anchor.get('region_short') or 'event'}_follow"
    anchor_run = anchor.get("runId", "")
    anchor_event = anchor.get("eventId", "")

    events = []
    for offset in range(1, n_following + 1):
        source_event_index = anchor_index + offset
        identity = read_event_identity(event_source_path, event_tree_name, source_event_index)
        if identity.get("out_of_range"):
            continue
        feature_info = find_feature_row_by_event_index(
            source_feature_path,
            feature_tree_name,
            source_event_index,
            annotation_features,
        )

        event = {
            "region": follow_region,
            "region_label": f"Following +{offset} after {anchor_region_label}",
            "region_short": follow_region_short,
            "particle_group": particle_group,
            "particle_label": particle_label,
            "partition_id": anchor.get("partition_id", ""),
            "feature_partition_path": source_feature_path,
            "feature_entry": feature_info.get("feature_entry", ""),
            "original_entry": feature_info.get("original_entry", ""),
            "source_feature_path": source_feature_path,
            "event_source_path": event_source_path,
            "event_tree_name": event_tree_name,
            "source_event_index": source_event_index,
            "runId": feature_info.get("runId", identity.get("runId", "")),
            "eventId": feature_info.get("eventId", identity.get("eventId", "")),
            "geofile": geofile,
            "features": feature_info.get("features", {}),
            "selection_expression": (
                f"following +{offset} after Run/Event {anchor_run}/{anchor_event}; "
                f"anchor source_event_index={anchor_index}"
            ),
        }
        events.append(event)
    return events


def plot_follow_events(
    follow_cfg: dict,
    outdir: str,
    config: dict,
    display_script: str,
    repo_root: str | Path,
) -> list[dict]:
    summary_csv = resolve_path(follow_cfg["summary_csv"], repo_root)
    summary_rows = read_csv_rows(summary_csv)
    anchor = find_follow_anchor(summary_rows, follow_cfg)
    events = build_follow_events(anchor, follow_cfg, config, repo_root)
    if not events:
        print("[follow] no follow-up events to plot")
        return []

    anchor_label = f"run{anchor.get('runId', '')}_event{anchor.get('eventId', '')}"
    follow_outdir = os.path.join(outdir, "follow_events", safe_name(anchor_label))
    os.makedirs(follow_outdir, exist_ok=True)
    print(
        "[follow] anchor "
        f"Run/Event {anchor.get('runId', '')}/{anchor.get('eventId', '')}; "
        f"plotting {len(events)} following event(s)"
    )
    return plot_group(
        region=events[0]["region"],
        particle=events[0]["particle_group"],
        events=events,
        outdir=follow_outdir,
        config=config,
        display_script=display_script,
        repo_root=repo_root,
    )


def main(args):
    repo_root = Path(args.repo_root).resolve()
    config = load_yaml(resolve_path(args.config, repo_root))

    metadata_csv = args.metadata_csv or (config.get("input", {}) or {}).get("metadata_csv")
    if not metadata_csv:
        raise RuntimeError("No metadata CSV was provided")

    outdir = args.outdir or (config.get("outputs", {}) or {}).get("outdir", "evaluation_region_partitions/event_display")
    outdir = resolve_path(outdir, repo_root)
    os.makedirs(outdir, exist_ok=True)

    display_cfg = config.get("display", {}) or {}
    display_script = find_sndsw_2d_display_script(
        args.sndsw_display_script or str(display_cfg.get("sndsw_display_script", "") or "").strip() or None
    )

    follow_cfg = configured_follow_options(args, config, outdir)
    if follow_cfg.get("enabled"):
        summary_rows = plot_follow_events(
            follow_cfg=follow_cfg,
            outdir=outdir,
            config=config,
            display_script=display_script,
            repo_root=repo_root,
        )
        summary_name = follow_cfg.get("output_summary_csv", "follow_summary.csv")
        summary_path = os.path.join(outdir, summary_name)
        write_summary_csv(summary_rows, summary_path)
        print(f"[saved] {summary_path} rows={len(summary_rows)}")
        return

    grouped_events = collect_events(config=config, metadata_csv=metadata_csv, repo_root=repo_root)
    enrich_display_labels(grouped_events, load_eval_plot_config(config, repo_root))

    all_summary_rows = []
    for (region, particle), events in grouped_events.items():
        all_summary_rows.extend(
            plot_group(
                region=region,
                particle=particle,
                events=events,
                outdir=outdir,
                config=config,
                display_script=display_script,
                repo_root=repo_root,
            )
        )

    summary_name = (config.get("outputs", {}) or {}).get("summary_csv", "summary.csv")
    summary_path = os.path.join(outdir, summary_name)
    write_summary_csv(all_summary_rows, summary_path)
    print(f"[saved] {summary_path} rows={len(all_summary_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw SNDSW event displays for selected region-partition feature events."
    )
    parser.add_argument("--config", required=True, help="Event-display YAML config")
    parser.add_argument("--metadata-csv", default=None, help="Override region partition metadata CSV")
    parser.add_argument("--outdir", default=None, help="Override output directory")
    parser.add_argument("--sndsw-display-script", default=None, help="Path to SNDSW 2dEventDisplay.py")
    parser.add_argument(
        "--follow-summary-csv",
        default=None,
        help="Use an existing display summary CSV as the anchor source for follow-up event displays",
    )
    parser.add_argument("--follow-run", type=int, default=None, help="Anchor runId for follow-up event display")
    parser.add_argument("--follow-event", type=int, default=None, help="Anchor eventId for follow-up event display")
    parser.add_argument(
        "--follow-source-index",
        type=int,
        default=None,
        help="Anchor source_event_index for follow-up event display",
    )
    parser.add_argument(
        "--n-following",
        type=int,
        default=None,
        help="Number of events immediately after the anchor source_event_index to display",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root used to resolve relative paths",
    )
    main(parser.parse_args())
