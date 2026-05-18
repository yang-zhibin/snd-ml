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

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


KAON_BINS = [
    "MC_kaon_5-10GeV",
    "MC_kaon_10-20GeV",
    "MC_kaon_20-30GeV",
    "MC_kaon_30-40GeV",
    "MC_kaon_50-60GeV",
    "MC_kaon_60-70GeV",
    "MC_kaon_70-80GeV",
    "MC_kaon_80-90GeV",
    "MC_kaon_90-100GeV",
    "MC_kaon_100-150GeV",
]

NEUTRON_BINS = [
    "MC_neutron_5-10GeV",
    "MC_neutron_10-20GeV",
    "MC_neutron_20-30GeV",
    "MC_neutron_30-40GeV",
    "MC_neutron_50-60GeV",
    "MC_neutron_60-70GeV",
    "MC_neutron_70-80GeV",
    "MC_neutron_80-90GeV",
    "MC_neutron_90-100GeV",
    "MC_neutron_100-150GeV",
]

NEUTRINO_CATS = [
    "MC_NC_nue",
    "MC_NC_numu",
    "MC_CC_nue",
    "MC_CC_numu",
]

ORDERED_CATEGORIES = (
    ["real_data", "MC_muonDIS"]
    + KAON_BINS
    + NEUTRON_BINS
    + NEUTRINO_CATS
)


def partition_from_filename(filepath: str) -> str | None:
    match = re.match(r"^eff_(.+)\.root$", os.path.basename(filepath))
    return match.group(1) if match else None


def classify_partition(partition: str) -> str | None:
    if partition.startswith("real_data"):
        return "real_data"

    if partition in set(NEUTRINO_CATS):
        return partition

    match = re.match(r"^(MC_(?:kaon|neutron)_\d+-\d+GeV)(?:_Max\d+-\d+)?$", partition)
    if match:
        return match.group(1)

    match = re.match(r"^(MC_muonDIS)(?:_Max\d+-\d+)?$", partition)
    if match:
        return match.group(1)

    return None


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse_event_limit(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value

    text = str(value).strip().lower()
    if text in {"all", "none", "unlimited", "-1"}:
        return None

    limit = int(text)
    if limit < 0:
        return None
    return limit


def event_limit_for_category(category: str, args) -> int | None:
    if category == "real_data":
        return parse_event_limit(args.max_events_real_data)
    return parse_event_limit(args.max_events)


def collect_category_files(input_dir: str) -> dict[str, list[str]]:
    filepaths = sorted(glob.glob(os.path.join(input_dir, "eff_*.root")))
    if not filepaths:
        raise RuntimeError(f"No files matching eff_*.root found in {input_dir}")

    category_files = defaultdict(list)
    for filepath in filepaths:
        partition = partition_from_filename(filepath)
        if partition is None:
            continue

        category = classify_partition(partition)
        if category is None:
            print(f"[skip] unrecognized partition: {partition}")
            continue

        category_files[category].append(filepath)

    return dict(category_files)


def infer_geofile(metadata_dir: str | None) -> str | None:
    if not metadata_dir:
        return None

    metadata_path = Path(metadata_dir)
    preferred = [
        "real_data_2024_skim_runs_metadata.csv",
        "MC_muonDIS_cvilela_metadata.csv",
        "MC_kaon_FTFP_BERT_metadata.csv",
        "MC_neutron_FTFP_BERT_metadata.csv",
        "MC_neutrino_2024_ve_metadata.csv",
        "MC_neutrino_2024_vm_metadata.csv",
        "MC_muon_up_metadata.csv",
    ]

    csv_paths = [metadata_path / name for name in preferred]
    csv_paths += sorted(path for path in metadata_path.glob("*.csv") if path not in csv_paths)

    for csv_path in csv_paths:
        if not csv_path.exists():
            continue

        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            if "geo_path" not in (reader.fieldnames or []):
                continue
            for row in reader:
                geofile = (row.get("geo_path") or "").strip()
                if geofile:
                    print(f"[geo] inferred geofile from {csv_path}: {geofile}")
                    return geofile

    return None


def setup_scifi_geometry(geofile: str):
    root_file = ROOT.TFile.Open(geofile)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open geofile: {geofile}")

    from rootpyPickler import Unpickler
    import shipLHC_conf as sndDet_conf

    ship_geo = Unpickler(root_file).load("ShipGeo")
    run = ROOT.FairRunSim()
    modules = sndDet_conf.configure(run, ship_geo)
    return root_file, run, modules["Scifi"]


def open_event_tree(root_file):
    for tree_name in ("cbmsim", "rawConv"):
        tree = root_file.Get(tree_name)
        if tree:
            return tree_name, tree
    return None, None


def count_saved_events(filepath: str) -> tuple[str | None, int]:
    root_file = ROOT.TFile.Open(filepath)
    if not root_file or root_file.IsZombie():
        print(f"[skip] could not open {filepath}")
        return None, 0

    tree_name, tree = open_event_tree(root_file)
    if tree is None:
        print(f"[skip] no cbmsim/rawConv tree in {filepath}")
        root_file.Close()
        return None, 0

    if not (tree.GetBranch("EventHeader") or tree.GetBranch("EventHeader.")):
        print(f"[skip] {filepath} has no EventHeader branch; official 2dEventDisplay.py requires it")
        root_file.Close()
        return tree_name, 0

    entries = int(tree.GetEntries())
    root_file.Close()
    return tree_name, entries


def find_sndsw_2d_display_script(script_path: str | None) -> str:
    if script_path:
        return script_path

    sndsw_root = os.environ.get("SNDSW_ROOT")
    if sndsw_root:
        candidate = os.path.join(sndsw_root, "shipLHC", "scripts", "2dEventDisplay.py")
        if os.path.exists(candidate):
            return candidate

    local_candidate = (
        Path(__file__).resolve().parents[2]
        / "public"
        / "SndBuild"
        / "sndsw"
        / "shipLHC"
        / "scripts"
        / "2dEventDisplay.py"
    )
    if local_candidate.exists():
        return str(local_candidate)

    raise RuntimeError("Could not find SNDSW 2dEventDisplay.py; pass --sndsw-display-script")


def write_sndsw_runner(
    runner_path: str,
    display_script: str,
    input_file: str,
    geofile: str,
    output_prefix: str,
    event_indices: list[int],
    output_extension: str,
    disable_logo: bool,
):
    text_only_draw_info = ""
    if disable_logo:
        text_only_draw_info = """
def drawInfo_text_only(pad, k, run, event, timestamp, moreEventInfo=[]):
    ROOT = namespace["ROOT"]
    eventTree = namespace["eventTree"]
    mc = namespace["mc"]
    options = namespace["options"]
    datetime = namespace["datetime"]

    if k == 1 or len(moreEventInfo) < 5:
        timestamp_print = False
        if not mc and hasattr(eventTree.EventHeader, "GetUTCtimestamp"):
            timestamp_print = True
            time_event = datetime.utcfromtimestamp(eventTree.EventHeader.GetUTCtimestamp())

        padText = ROOT.TPad("info", "info", 0.19, 0.1, 0.6, 0.3)
        padText.SetFillStyle(4000)
        padText.Draw()
        padText.cd()
        textInfo = ROOT.TLatex()
        textInfo.SetTextAlign(11)
        textInfo.SetTextFont(42)
        textInfo.SetTextSize(.15)
        textInfo.DrawLatex(0, 0.6, "SND@LHC Experiment, CERN")
        if hasattr(eventTree.EventHeader, "GetEventNumber"):
            event_number = eventTree.EventHeader.GetEventNumber()
        else:
            event_number = event
        textInfo.DrawLatex(0, 0.4, "Run / Event: " + str(run) + " / " + str(event_number))
        if timestamp_print:
            textInfo.DrawLatex(0, 0.2, "Time (GMT): {}".format(time_event))
        pad.cd(k)
    elif options.extraInfo:
        padText = ROOT.TPad("info", "info", 0.29, 0.12, 0.9, 0.35)
        padText.SetFillStyle(4000)
        padText.Draw()
        padText.cd()
        textInfo = ROOT.TLatex()
        textInfo.SetTextAlign(11)
        textInfo.SetTextFont(42)
        textInfo.SetTextSize(.1)
        textInfo.SetTextColor(ROOT.kMagenta + 2)
        dely = 0.12
        for i in range(min(7, len(moreEventInfo))):
            textInfo.DrawLatex(0.4, 0.9 - dely * i, moreEventInfo[i])
        pad.cd(k)

namespace["drawInfo"] = drawInfo_text_only
namespace["loopEvents"].__globals__["drawInfo"] = drawInfo_text_only
"""

    runner = f"""#!/usr/bin/env python3
import os
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

namespace = runpy.run_path({display_script!r}, run_name="__main__")
namespace["ROOT"].gStyle.SetPaperSize(32.0, 20.0)
namespace["options"].storePic = {output_prefix!r}
namespace["ut"].bookCanvas(
    namespace["h"],
    key="simpleDisplay",
    title="simple event display",
    nx=1600,
    ny=1000,
    cx=1,
    cy=2,
)
namespace["h"]["simpleDisplay"].SetCanvasSize(1600, 1000)
{text_only_draw_info}
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
    with open(runner_path, "w") as handle:
        handle.write(runner)


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


def prepare_sndsw_display_file(input_file_path: str, output_file_path: str, tree_name: str, n_events: int):
    input_file = ROOT.TFile.Open(input_file_path)
    if not input_file or input_file.IsZombie():
        raise RuntimeError(f"Could not open saved event file: {input_file_path}")

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
    copied_tree = input_tree.CloneTree(n_events)
    copied_tree.Write("", ROOT.TObject.kOverwrite)

    output_file.Close()
    input_file.Close()


def run_sndsw_display(
    display_script: str,
    input_file: str,
    geofile: str,
    workdir: str,
    category: str,
    file_index: int,
    max_events: int | None,
    output_extension: str,
    disable_logo: bool,
) -> list[str]:
    tree_name, entries = count_saved_events(input_file)
    if entries <= 0:
        print(f"[skip] {category:25s} {os.path.basename(input_file):55s} {tree_name} entries={entries}")
        return []

    n_events = entries if max_events is None else min(max_events, entries)
    display_input = os.path.join(workdir, f"display_input_{safe_name(category)}_{file_index:04d}.root")
    prepare_sndsw_display_file(input_file, display_input, tree_name, n_events)

    event_indices = list(range(n_events))
    prefix = os.path.join(workdir, f"{safe_name(category)}__file{file_index:04d}__")
    runner_path = os.path.join(workdir, f"run_{safe_name(category)}_{file_index:04d}.py")

    write_sndsw_runner(
        runner_path=runner_path,
        display_script=display_script,
        input_file=display_input,
        geofile=geofile,
        output_prefix=prefix,
        event_indices=event_indices,
        output_extension=output_extension,
        disable_logo=disable_logo,
    )

    print(
        f"[sndsw] {category:25s} {os.path.basename(input_file):55s} "
        f"{tree_name} entries={entries} draw={n_events}"
    )
    completed = subprocess.run([sys.executable, runner_path], cwd=workdir, check=False)
    page_files = sorted(glob.glob(prefix + f"*.{output_extension}"))

    if completed.returncode != 0:
        if completed.returncode == -signal.SIGTERM and page_files:
            print("[sndsw] 2dEventDisplay.py exited with SIGTERM after writing pages; accepting output")
        else:
            raise subprocess.CalledProcessError(completed.returncode, completed.args)

    return page_files


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
    except Exception as matplotlib_error:
        print(f"[warn] matplotlib PNG-to-PDF conversion failed: {matplotlib_error}")

    convert = shutil.which("convert")
    if convert:
        subprocess.run([convert, *input_pngs, output_pdf], check=True)
        return

    raise RuntimeError("Could not convert PNG event displays to PDF")


def merge_pages_to_pdf(input_pages: list[str], output_pdf: str):
    if not input_pages:
        raise RuntimeError(f"No PDF pages to merge for {output_pdf}")

    extensions = {os.path.splitext(page)[1].lower() for page in input_pages}
    if extensions == {".png"}:
        merge_pngs_to_pdf(input_pages, output_pdf)
        return

    if extensions in ({".ps"}, {".eps"}):
        ps2pdf = shutil.which("ps2pdf")
        if not ps2pdf:
            raise RuntimeError("Could not convert vector PostScript/EPS pages to PDF: ps2pdf not found")

        with tempfile.TemporaryDirectory(prefix="saved_events_ps2pdf_") as pdf_workdir:
            converted_pages = []
            crop_option = ["-dEPSCrop"] if extensions == {".eps"} else []
            for index, ps_page in enumerate(input_pages):
                pdf_page = os.path.join(pdf_workdir, f"page_{index:04d}.pdf")
                subprocess.run([ps2pdf, *crop_option, ps_page, pdf_page], check=True)
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

    raise RuntimeError(f"Could not merge event-display pages with extensions: {sorted(extensions)}")


def get_event_id(tree, local_index: int) -> str:
    header = getattr(tree, "EventHeader", None)
    if header:
        for method in ("GetEventNumber", "GetEventID", "GetEventId"):
            if hasattr(header, method):
                try:
                    return str(getattr(header, method)())
                except Exception:
                    pass
    return str(local_index)


def make_graph(points: list[tuple[float, float]], name: str, color: int) -> ROOT.TGraph:
    graph = ROOT.TGraph(len(points))
    graph.SetName(name)
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(0.75)
    graph.SetMarkerColor(color)
    graph.SetLineColor(color)
    for index, (z_pos, coord) in enumerate(points):
        graph.SetPoint(index, z_pos, coord)
    return graph


def make_frame(name: str, title: str, points: list[tuple[float, float]], x_title: str, y_title: str) -> ROOT.TH2F:
    if points:
        z_values = [point[0] for point in points]
        c_values = [point[1] for point in points]
        z_min, z_max = min(z_values), max(z_values)
        c_min, c_max = min(c_values), max(c_values)
        z_pad = max(5.0, 0.10 * max(1.0, z_max - z_min))
        c_pad = max(5.0, 0.15 * max(1.0, c_max - c_min))
        z_low, z_high = z_min - z_pad, z_max + z_pad
        c_low, c_high = c_min - c_pad, c_max + c_pad
    else:
        z_low, z_high = 250.0, 350.0
        c_low, c_high = -80.0, 90.0

    frame = ROOT.TH2F(name, title, 10, z_low, z_high, 10, c_low, c_high)
    frame.SetStats(False)
    frame.GetXaxis().SetTitle(x_title)
    frame.GetYaxis().SetTitle(y_title)
    return frame


def decode_scifi_channel(det_id: int) -> tuple[int, int, int]:
    orientation = (det_id // 100000) % 10
    plane = 2 * (det_id // 1000000 - 1) + orientation
    mat = (det_id % 100000) // 10000
    channel = det_id % 1000 + (det_id % 10000) // 1000 * 128
    return plane, mat, channel


def collect_scifi_points(event, scifi_det, coordinate_mode: str):
    vec_a = ROOT.TVector3()
    vec_b = ROOT.TVector3()
    x_points = []
    y_points = []
    total_signal = 0.0
    n_valid = 0

    if not hasattr(event, "Digi_ScifiHits"):
        return x_points, y_points, n_valid, total_signal

    for hit in event.Digi_ScifiHits:
        if not hit.isValid():
            continue

        det_id = hit.GetDetectorID()
        if coordinate_mode == "geometry":
            scifi_det.GetSiPMPosition(det_id, vec_a, vec_b)
            z_coord = 0.5 * (vec_a.Z() + vec_b.Z())
            if hit.isVertical():
                x_points.append((z_coord, 0.5 * (vec_a.X() + vec_b.X())))
            else:
                y_points.append((z_coord, 0.5 * (vec_a.Y() + vec_b.Y())))
        else:
            plane, mat, channel = decode_scifi_channel(det_id)
            point = (float(plane), float(mat * 512 + channel))
            if hit.isVertical():
                x_points.append(point)
            else:
                y_points.append(point)

        try:
            total_signal += float(hit.GetSignal(0))
        except Exception:
            pass
        n_valid += 1

    return x_points, y_points, n_valid, total_signal


def draw_event(
    canvas,
    event,
    scifi_det,
    label: str,
    source: str,
    tree_name: str,
    local_index: int,
    shown_index: int,
    coordinate_mode: str,
):
    x_points, y_points, n_valid, total_signal = collect_scifi_points(event, scifi_det, coordinate_mode)

    canvas.Clear()
    canvas.Divide(1, 2)

    header = (
        f"{label}    page event {shown_index}    "
        f"tree {tree_name} entry {local_index} event {get_event_id(event, local_index)}    "
        f"SciFi hits {n_valid} QDC {total_signal:.1f}"
    )

    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextSize(0.035)

    if coordinate_mode == "geometry":
        x_axis_title = "z [cm]"
        x_projection_y_title = "x [cm]"
        y_projection_y_title = "y [cm]"
    else:
        x_axis_title = "SciFi plane index"
        x_projection_y_title = "vertical mat-channel"
        y_projection_y_title = "horizontal mat-channel"

    canvas.cd(1)
    ROOT.gPad.SetGrid()
    frame_x = make_frame(
        f"frame_x_{shown_index}",
        header,
        x_points,
        x_axis_title,
        x_projection_y_title,
    )
    frame_x.Draw()
    graph_x = make_graph(x_points, f"graph_x_{shown_index}", ROOT.kAzure + 2)
    graph_x.Draw("P SAME")
    latex.DrawLatex(0.12, 0.92, "SciFi vertical fibres")

    canvas.cd(2)
    ROOT.gPad.SetGrid()
    frame_y = make_frame(
        f"frame_y_{shown_index}",
        os.path.basename(source),
        y_points,
        x_axis_title,
        y_projection_y_title,
    )
    frame_y.Draw()
    graph_y = make_graph(y_points, f"graph_y_{shown_index}", ROOT.kOrange + 7)
    graph_y.Draw("P SAME")
    latex.DrawLatex(0.12, 0.92, "SciFi horizontal fibres")

    canvas.Update()


def draw_empty_pdf_page(canvas, pdf_path: str, category: str, message: str):
    canvas.Clear()
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.12, 0.62, category)
    latex.DrawLatex(0.12, 0.54, message)
    canvas.Print(pdf_path)


def plot_category(category: str, files: list[str], outdir: str, scifi_det, max_events: int | None, coordinate_mode: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, f"events_{safe_name(category)}.pdf")
    canvas = ROOT.TCanvas(f"c_{safe_name(category)}", "", 1100, 850)

    canvas.Print(pdf_path + "[")
    shown = 0

    for filepath in files:
        if max_events is not None and shown >= max_events:
            break

        root_file = ROOT.TFile.Open(filepath)
        if not root_file or root_file.IsZombie():
            print(f"[skip] could not open {filepath}")
            continue

        tree_name, tree = open_event_tree(root_file)
        if tree is None:
            print(f"[skip] no cbmsim/rawConv tree in {filepath}")
            root_file.Close()
            continue

        entries = tree.GetEntries()
        print(f"[read] {category:25s} {os.path.basename(filepath):55s} {tree_name} entries={entries}")

        for local_index in range(entries):
            if max_events is not None and shown >= max_events:
                break
            tree.GetEntry(local_index)
            shown += 1
            draw_event(
                canvas=canvas,
                event=tree,
                scifi_det=scifi_det,
                label=category,
                source=filepath,
                tree_name=tree_name,
                local_index=local_index,
                shown_index=shown,
                coordinate_mode=coordinate_mode,
            )
            canvas.Print(pdf_path)

        root_file.Close()

    pages = shown
    if shown == 0:
        draw_empty_pdf_page(canvas, pdf_path, category, "No saved events found in this category.")
        pages = 1

    canvas.Print(pdf_path + "]")
    print(f"[saved] {pdf_path} pages={pages}")
    return pages


def plot_category_with_sndsw(
    category: str,
    files: list[str],
    outdir: str,
    geofile: str,
    display_script: str,
    max_events: int | None,
    output_mode: str,
) -> int:
    pdf_path = os.path.join(outdir, f"events_{safe_name(category)}.pdf")
    remaining = max_events
    page_files = []
    output_extension = "eps" if output_mode == "vector_pdf" else "png"
    disable_logo = output_mode == "vector_pdf"

    with tempfile.TemporaryDirectory(prefix=f"saved_events_{safe_name(category)}_") as workdir:
        for file_index, filepath in enumerate(files):
            if remaining is not None and remaining <= 0:
                break

            new_pages = run_sndsw_display(
                display_script=display_script,
                input_file=filepath,
                geofile=geofile,
                workdir=workdir,
                category=category,
                file_index=file_index,
                max_events=remaining,
                output_extension=output_extension,
                disable_logo=disable_logo,
            )
            page_files.extend(new_pages)
            if remaining is not None:
                remaining -= len(new_pages)

        if not page_files:
            canvas = ROOT.TCanvas(f"c_empty_{safe_name(category)}", "", 1100, 850)
            empty_pdf = os.path.join(workdir, f"empty_{safe_name(category)}.pdf")
            draw_empty_pdf_page(canvas, empty_pdf, category, "No saved events found in this category.")
            page_files.append(empty_pdf)

        merge_pages_to_pdf(page_files, pdf_path)

    print(f"[saved] {pdf_path} pages={len(page_files)}")
    return len(page_files)


def main(args):
    geo_file = None
    run = None
    scifi_det = None
    geofile = args.geofile or infer_geofile(args.metadata_dir)
    if args.backend == "sndsw" and not geofile:
        raise RuntimeError("No geofile provided and no geo_path could be inferred from metadata")

    if args.backend == "simple" and args.coordinate_mode == "geometry":
        geofile = args.geofile or infer_geofile(args.metadata_dir)
        if not geofile:
            raise RuntimeError("No geofile provided and no geo_path could be inferred from metadata")
        geo_file, run, scifi_det = setup_scifi_geometry(geofile)

    category_files = collect_category_files(args.input_dir)
    display_script = find_sndsw_2d_display_script(args.sndsw_display_script) if args.backend == "sndsw" else None

    selected_categories = args.categories or ORDERED_CATEGORIES
    total_pages = 0

    for category in selected_categories:
        files = category_files.get(category, [])
        if not files:
            continue
        if args.backend == "sndsw":
            max_events = event_limit_for_category(category, args)
            total_pages += plot_category_with_sndsw(
                category=category,
                files=files,
                outdir=args.outdir,
                geofile=geofile,
                display_script=display_script,
                max_events=max_events,
                output_mode=args.output_mode,
            )
        else:
            max_events = event_limit_for_category(category, args)
            total_pages += plot_category(
                category,
                files,
                args.outdir,
                scifi_det,
                max_events,
                args.coordinate_mode,
            )

    if total_pages == 0:
        print("[warn] no event pages were produced from the selected categories")

    _ = (geo_file, run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw saved event displays from eff_*.root files.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="/eos/experiment/sndlhc/users/zhibin/nueAnalysis",
        help="Directory containing eff_*.root files",
    )
    parser.add_argument(
        "--outdir",
        default="event_displays",
        help="Output directory for multipage PDF files",
    )
    parser.add_argument(
        "--geofile",
        default="/eos/experiment/sndlhc/convertedData/commissioning/2024/geofile_sndlhc_TI18_V0_2024.root",
        help="Geometry ROOT file. If omitted, the first geo_path in --metadata-dir is used.",
    )
    parser.add_argument(
        "--metadata-dir",
        default=None,
        help="Metadata directory used to infer a geofile when --geofile is not provided.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=2,
        help="Maximum number of events to draw per category PDF",
    )
    parser.add_argument(
        "--max-events-real-data",
        default="all",
        help="Maximum number of real-data events to draw. Use 'all' for no limit.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional category list, e.g. real_data MC_muonDIS MC_kaon_5-10GeV.",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=["channel", "geometry"],
        default="channel",
        help="Simple backend only: use detector-ID channel coordinates or geometry positions.",
    )
    parser.add_argument(
        "--backend",
        choices=["sndsw", "simple"],
        default="sndsw",
        help="Use the official SNDSW 2dEventDisplay.py backend, or the fallback simple plotter.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["png_pdf", "vector_pdf"],
        default="png_pdf",
        help="For the SNDSW backend: use PNG pages merged to PDF, or direct vector PDF pages without the image logo.",
    )
    parser.add_argument(
        "--sndsw-display-script",
        default=None,
        help="Path to SNDSW shipLHC/scripts/2dEventDisplay.py. Defaults to $SNDSW_ROOT/shipLHC/scripts/2dEventDisplay.py.",
    )

    main(parser.parse_args())
