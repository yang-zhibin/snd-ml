#!/usr/bin/env python3
"""Shared batch helpers for producing official SNDSW 2D event displays."""

from __future__ import annotations

import glob
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import ROOT


ROOT.gROOT.SetBatch(True)


def find_sndsw_2d_display_script(script_path: str | None) -> str:
    if script_path:
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"SNDSW display script does not exist: {script_path}")
        return script_path
    sndsw_root = os.environ.get("SNDSW_ROOT")
    if sndsw_root:
        candidate = os.path.join(sndsw_root, "shipLHC", "scripts", "2dEventDisplay.py")
        if os.path.exists(candidate):
            return candidate
    for base in Path(__file__).resolve().parents:
        candidate = base / "public" / "SndBuild" / "sndsw" / "shipLHC" / "scripts" / "2dEventDisplay.py"
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("Could not find SNDSW 2dEventDisplay.py")


def open_event_tree(root_file, candidates: list[str]):
    for name in candidates:
        tree = root_file.Get(name)
        if tree:
            return name, tree
    return None, None


def displayable_event_tree(path: str, candidates: list[str]) -> tuple[str, int]:
    source = ROOT.TFile.Open(path)
    if not source or source.IsZombie():
        raise OSError(f"Cannot open event-display source {path}")
    try:
        name, tree = open_event_tree(source, candidates)
        if not tree:
            raise RuntimeError(f"No event tree {candidates} in {path}")
        if not (tree.GetBranch("EventHeader") or tree.GetBranch("EventHeader.")):
            raise RuntimeError(f"Event tree {name} has no EventHeader in {path}")
        return str(name), int(tree.GetEntries())
    finally:
        source.Close()


def latex_safe(value: object) -> str:
    return str(value).replace("\\", "/").replace("{", "(").replace("}", ")")


def write_sndsw_runner(
    runner_path: str, display_script: str, input_file: str, geofile: str,
    output_prefix: str, event_indices: list[int], output_extension: str,
    annotation_lines: list[list[str]], canvas_width: int, canvas_height: int,
    with_track: int = 0, with_hough_track: int = 0,
    minimum_tracks: int = 0, track_verbose: int = 0,
    overlay_lines: list[list[dict]] | None = None,
):
    if with_track not in (0, 1, 2, 3):
        raise ValueError(f"Invalid simple-tracking mode: {with_track}")
    if with_hough_track not in (0, 1, 2, 3, 4):
        raise ValueError(f"Invalid Hough-tracking mode: {with_hough_track}")
    if minimum_tracks < 0 or track_verbose < 0:
        raise ValueError("minimum_tracks and track_verbose must be non-negative")
    annotations = [[latex_safe(line) for line in lines] for lines in annotation_lines]
    overlays = overlay_lines or [[] for _ in event_indices]
    if len(overlays) != len(event_indices):
        raise ValueError("overlay_lines must contain one entry per displayed event")
    runner = f'''#!/usr/bin/env python3
import runpy
import sys

sys.argv = [
    {display_script!r}, "-f", {input_file!r}, "-p", "", "-g", {geofile!r},
    "--rootbatch", "--extension", {output_extension!r},
]
event_annotations = {annotations!r}
event_overlay_lines = {overlays!r}
namespace = runpy.run_path({display_script!r}, run_name="__main__")
ROOT = namespace["ROOT"]
# Older muon-MC productions store FairEventHeader, whereas the current
# 2dEventDisplay.py unconditionally calls GetEventNumber().  In those files
# the equivalent stable identifier is GetMCEntryNumber().
if (
    hasattr(ROOT, "FairEventHeader")
    and not hasattr(ROOT.FairEventHeader, "GetEventNumber")
    and hasattr(ROOT.FairEventHeader, "GetMCEntryNumber")
):
    ROOT.FairEventHeader.GetEventNumber = lambda self: self.GetMCEntryNumber()
namespace["ROOT"].gStyle.SetPaperSize(32.0, 20.0)
namespace["options"].storePic = {output_prefix!r}
namespace["ut"].bookCanvas(
    namespace["h"], key="simpleDisplay", title="simple event display",
    nx={int(canvas_width)}, ny={int(canvas_height)}, cx=1, cy=2,
)
namespace["h"]["simpleDisplay"].SetCanvasSize({int(canvas_width)}, {int(canvas_height)})

def drawInfo_manifest(pad, k, run, event, timestamp, moreEventInfo=[]):
    ROOT = namespace["ROOT"]
    eventTree = namespace["eventTree"]
    mc = namespace["mc"]
    datetime = namespace["datetime"]
    lines = ["SND@LHC Experiment, CERN"]
    event_number = event
    if hasattr(eventTree.EventHeader, "GetEventNumber"):
        try:
            event_number = eventTree.EventHeader.GetEventNumber()
        except Exception:
            pass
    lines.append("Run / Event: " + str(run) + " / " + str(event_number))
    if not mc and hasattr(eventTree.EventHeader, "GetUTCtimestamp"):
        try:
            lines.append("Time (GMT): " + str(datetime.utcfromtimestamp(eventTree.EventHeader.GetUTCtimestamp())))
        except Exception:
            pass
    try:
        lines.extend(event_annotations[int(event)])
    except Exception:
        pass
    info = ROOT.TPad("manifest_info", "manifest_info", 0.16, 0.03, 0.94, 0.39)
    info.SetFillStyle(4000)
    info.Draw()
    info.cd()
    text = ROOT.TLatex()
    text.SetTextAlign(11)
    text.SetTextFont(42)
    text.SetTextSize(0.074 if len(lines) > 8 else 0.086)
    y = 0.94
    dy = 0.086 if len(lines) > 8 else 0.105
    for line in lines[:12]:
        text.DrawLatex(0.0, y, str(line))
        y -= dy
    pad.cd(k)

namespace["drawInfo"] = drawInfo_manifest
namespace["loopEvents"].__globals__["drawInfo"] = drawInfo_manifest

overlay_state = {{"event": 0}}
def drawFrozenLines(event):
    ROOT = namespace["ROOT"]
    h = namespace["h"]
    index = overlay_state["event"]
    overlay_state["event"] += 1
    if index >= len(event_overlay_lines):
        return
    h.setdefault("frozen_line_overlays", [])
    for specification in event_overlay_lines[index]:
        z0 = float(specification["reference_z_cm"])
        for pad_number, coordinate in ((1, "x"), (2, "y")):
            pad = h["simpleDisplay"].cd(pad_number)
            z_low, z_high = float(pad.GetUxmin()), float(pad.GetUxmax())
            origin = float(specification[f"reference_{{coordinate}}_cm"])
            slope = float(specification[f"slope_{{coordinate}}z"])
            shift = float(specification.get(f"shift_{{coordinate}}_cm", 0.0))
            low_value = origin + slope * (z_low - z0) + shift
            high_value = origin + slope * (z_high - z0) + shift
            line = ROOT.TLine(z_low, low_value, z_high, high_value)
            line.SetLineColor(int(specification.get("color", ROOT.kBlue)))
            line.SetLineWidth(int(specification.get("width", 2)))
            line.SetLineStyle(int(specification.get("style", 1)))
            line.Draw("same")
            h["frozen_line_overlays"].append(line)
            pad.Update()

namespace["userProcessing"] = drawFrozenLines
namespace["loopEvents"].__globals__["userProcessing"] = drawFrozenLines
# SNDSW's Hough path reloads GetEvent(N), where N is the position inside a
# list-valued ``start`` rather than the requested tree entry.  Preserve the
# official display code but map that reload back to the selected entries.
if {int(with_hough_track)} > 0:
    class _SelectedEntryTree:
        def __init__(self, tree, entries):
            self._tree = tree
            self._entries = entries

        def GetEvent(self, index):
            return self._tree.GetEvent(self._entries[int(index)])

    class _SelectedEntrySource:
        def __init__(self, source, entries):
            self._source = source
            self._tree = _SelectedEntryTree(source.GetInTree(), entries)

        def GetInTree(self):
            return self._tree

    namespace["loopEvents"].__globals__["source"] = _SelectedEntrySource(
        namespace["source"], {event_indices!r}
    )
namespace["loopEvents"](
    start={event_indices!r}, auto=True, save=False, goodEvents=False,
    withTrack={int(with_track)}, withHoughTrack={int(with_hough_track)},
    nTracks={int(minimum_tracks)}, verbose={int(track_verbose)}, hitColour="q",
)
'''
    with open(runner_path, "w", encoding="utf-8") as stream:
        stream.write(runner)


def merge_pngs_to_pdf(pages: list[str], output_pdf: str):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(output_pdf) as pdf:
            for page in pages:
                image = plt.imread(page)
                height, width = image.shape[:2]
                figure = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
                axis = figure.add_axes([0, 0, 1, 1])
                axis.imshow(image)
                axis.axis("off")
                pdf.savefig(figure, dpi=100)
                plt.close(figure)
        return
    except Exception as error:
        print(f"[warn] matplotlib PNG conversion failed: {error}")
    convert = shutil.which("convert")
    if not convert:
        raise RuntimeError("Neither matplotlib nor ImageMagick can convert PNG pages")
    subprocess.run([convert, *pages, output_pdf], check=True)


def merge_pages_to_pdf(pages: list[str], output_pdf: str):
    if not pages:
        raise RuntimeError(f"No pages to merge for {output_pdf}")
    extensions = {Path(page).suffix.lower() for page in pages}
    if extensions == {".png"}:
        merge_pngs_to_pdf(pages, output_pdf)
        return
    if extensions in ({".eps"}, {".ps"}):
        ps2pdf = shutil.which("ps2pdf")
        if not ps2pdf:
            raise RuntimeError("ps2pdf is required for vector event-display output")
        with tempfile.TemporaryDirectory(prefix="sndsw_ps2pdf_") as workdir:
            converted = []
            for index, page in enumerate(pages):
                target = os.path.join(workdir, f"page_{index:04d}.pdf")
                crop = ["-dEPSCrop"] if extensions == {".eps"} else []
                subprocess.run([ps2pdf, *crop, page, target], check=True)
                converted.append(target)
            merge_pages_to_pdf(converted, output_pdf)
        return
    if extensions == {".pdf"}:
        pdfunite = shutil.which("pdfunite")
        if pdfunite:
            subprocess.run([pdfunite, *pages, output_pdf], check=True)
            return
        ghostscript = shutil.which("gs")
        if ghostscript:
            subprocess.run([
                ghostscript, "-dBATCH", "-dNOPAUSE", "-q", "-sDEVICE=pdfwrite",
                f"-sOutputFile={output_pdf}", *pages,
            ], check=True)
            return
    raise RuntimeError(f"Cannot merge page extensions {sorted(extensions)}")


def run_sndsw_batch(
    rows: list[dict], display_script: str, workdir: str, output_extension: str,
    annotation_lines: list[list[str]], canvas_width: int, canvas_height: int,
    with_track: int = 0, with_hough_track: int = 0,
    minimum_tracks: int = 0, track_verbose: int = 0,
    overlay_lines: list[list[dict]] | None = None,
) -> list[str]:
    if not rows:
        return []
    source = rows[0]["event_source_path"]
    geofile = rows[0]["geo_path"]
    source_event_indices = [int(row["source_event_index"]) for row in rows]
    tag = f"batch_{abs(hash((source, tuple(source_event_indices))))}"
    prefix = os.path.join(workdir, f"{tag}__")
    runner = os.path.join(workdir, f"run_{tag}.py")
    write_sndsw_runner(
        runner, display_script, source, geofile, prefix,
        source_event_indices, output_extension, annotation_lines,
        canvas_width, canvas_height, with_track, with_hough_track,
        minimum_tracks, track_verbose, overlay_lines,
    )
    completed = subprocess.run([sys.executable, runner], cwd=workdir, check=False)
    pages = sorted(glob.glob(prefix + f"*.{output_extension}"))
    if completed.returncode != 0:
        if completed.returncode == -signal.SIGTERM and pages:
            print("[sndsw] SIGTERM after page output; accepting completed pages")
        else:
            raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return pages
