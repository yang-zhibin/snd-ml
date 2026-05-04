import os
import re
import ROOT
from collections import defaultdict

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# =========================================================
# Config
# =========================================================

data_dir = "/eos/experiment/sndlhc/users/zhibin/nueAnalysis/"
tree_name = "sndData"
output_dir = "PG_hadron_plots"

max_kaon_partitions = 20
max_neutron_partitions = 20

nueAnalysis_partitions = (
    [f"MC_kaon_5-10GeV_Max{max_kaon_partitions}-{i}" for i in range(1, max_kaon_partitions + 1)] +
    [f"MC_neutron_5-10GeV_Max{max_neutron_partitions}-{i}" for i in range(1, max_neutron_partitions + 1)] +
    [
        "MC_kaon_10-20GeV", "MC_kaon_20-30GeV", "MC_kaon_30-40GeV",
        "MC_kaon_50-60GeV", "MC_kaon_60-70GeV", "MC_kaon_70-80GeV",
        "MC_kaon_80-90GeV", "MC_kaon_90-100GeV",

        "MC_neutron_10-20GeV", "MC_neutron_20-30GeV", "MC_neutron_30-40GeV",
        "MC_neutron_50-60GeV", "MC_neutron_60-70GeV", "MC_neutron_70-80GeV",
        "MC_neutron_80-90GeV", "MC_neutron_90-100GeV",
    ]
)

energy_order = [
    "5-10GeV", "10-20GeV", "20-30GeV", "30-40GeV",
    "50-60GeV", "60-70GeV", "70-80GeV", "80-90GeV", "90-100GeV"
]
particle_order = ["kaon", "neutron"]

energy_colors = {
    "5-10GeV": ROOT.kBlack,
    "10-20GeV": ROOT.kRed + 1,
    "20-30GeV": ROOT.kBlue + 1,
    "30-40GeV": ROOT.kGreen + 2,
    "50-60GeV": ROOT.kMagenta + 1,
    "60-70GeV": ROOT.kOrange + 7,
    "70-80GeV": ROOT.kCyan + 1,
    "80-90GeV": ROOT.kViolet + 1,
    "90-100GeV": ROOT.kGray + 2,
}

particle_styles = {
    "kaon": 1,
    "neutron": 2,
}

# 2D ranges you requested
XMIN_2D = -80.0
XMAX_2D = 20.0
YMIN_2D = -20.0
YMAX_2D = 80.0
NBINS_2D_X = 200
NBINS_2D_Y = 200

# theta ranges
THETA_MIN = -0.5
THETA_MAX = 0.5
THETA_BINS = 200

# keep objects alive
all_hists = {}
all_canvases = {}

# =========================================================
# Helpers
# =========================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sanitize_name(text):
    return text.replace("-", "_").replace(".", "p").replace("/", "_")

def parse_partition(partition):
    """
    Examples:
      MC_kaon_5-10GeV_Max20-1 -> ('kaon', '5-10GeV')
      MC_neutron_10-20GeV     -> ('neutron', '10-20GeV')
    """
    pattern = r"^MC_(kaon|neutron)_([0-9]+-[0-9]+GeV)(?:_Max[0-9]+-[0-9]+)?$"
    match = re.match(pattern, partition)
    if not match:
        raise ValueError(f"Cannot parse partition name: {partition}")
    return match.group(1), match.group(2)

def build_file_groups(partitions, data_dir):
    grouped_files = defaultdict(list)

    for partition in partitions:
        try:
            particle, energy = parse_partition(partition)
        except ValueError as e:
            print(f"[WARNING] {e}")
            continue

        filepath = os.path.join(data_dir, f"hist_{partition}.root")
        if not os.path.exists(filepath):
            print(f"[WARNING] Missing file: {filepath}")
            continue

        grouped_files[(particle, energy)].append(filepath)

    return grouped_files

def build_chains(grouped_files, tree_name):
    chains = {}
    for key, files in grouped_files.items():
        chain = ROOT.TChain(tree_name)
        for filepath in files:
            chain.Add(filepath)
        chains[key] = chain
    return chains

def get_chain_entries(chain):
    try:
        return chain.GetEntries()
    except Exception:
        return 0

def normalize_hist(hist):
    integral = hist.Integral()
    if integral > 0:
        hist.Scale(1.0 / integral)

def style_hist(hist, particle, energy):
    hist.SetLineColor(energy_colors.get(energy, ROOT.kBlack))
    hist.SetLineStyle(particle_styles.get(particle, 1))
    hist.SetLineWidth(2)

def save_canvas(canvas, outdir, basename):
    # png_path = os.path.join(outdir, f"{basename}.png")
    pdf_path = os.path.join(outdir, f"{basename}.pdf")
    # canvas.SaveAs(png_path)
    canvas.SaveAs(pdf_path)

def remove_old_hist_if_exists(name):
    obj = ROOT.gROOT.FindObject(name)
    if obj:
        ROOT.gROOT.GetListOfSpecials().Remove(obj)
        ROOT.gROOT.GetList().Remove(obj)

def draw_th2_from_chain(chain, hist_name, hist_title, expr_y, expr_x,
                        nbins_x, x_min, x_max, nbins_y, y_min, y_max,
                        selection=""):
    """
    Uses TTree::Draw to create/fill the histogram, then fetches it.
    This avoids the detached-hist problem.
    """
    remove_old_hist_if_exists(hist_name)

    draw_cmd = (
        f"{expr_y}:{expr_x} >> "
        f"{hist_name}({nbins_x},{x_min},{x_max},{nbins_y},{y_min},{y_max})"
    )
    n_drawn = chain.Draw(draw_cmd, selection, "goff")
    if n_drawn <= 0:
        return None, n_drawn

    hist = ROOT.gROOT.FindObject(hist_name)
    if not hist:
        return None, n_drawn

    hist.SetTitle(hist_title)
    hist.SetDirectory(0)
    return hist, n_drawn

def draw_th1_from_chain(chain, hist_name, hist_title, expr,
                        nbins, x_min, x_max, selection=""):
    """
    Uses TTree::Draw to create/fill the histogram, then fetches it.
    """
    remove_old_hist_if_exists(hist_name)

    draw_cmd = f"{expr} >> {hist_name}({nbins},{x_min},{x_max})"
    n_drawn = chain.Draw(draw_cmd, selection, "goff")
    if n_drawn <= 0:
        return None, n_drawn

    hist = ROOT.gROOT.FindObject(hist_name)
    if not hist:
        return None, n_drawn

    hist.SetTitle(hist_title)
    hist.SetDirectory(0)
    return hist, n_drawn

# =========================================================
# Build grouped chains
# =========================================================

ensure_dir(output_dir)

grouped_files = build_file_groups(nueAnalysis_partitions, data_dir)
chains = build_chains(grouped_files, tree_name)

print("\n=== Built chains ===")
for particle in particle_order:
    for energy in energy_order:
        key = (particle, energy)
        if key not in chains:
            continue
        print(
            f"{particle:8s} {energy:10s} : "
            f"files = {len(grouped_files[key]):2d}, "
            f"entries = {get_chain_entries(chains[key])}"
        )

# =========================================================
# 1) 2D plot: avg_scifi_x vs avg_scifi_y
# =========================================================

scifi_outdir = os.path.join(output_dir, "avg_scifi_xy")
ensure_dir(scifi_outdir)

for particle in particle_order:
    for energy in energy_order:
        key = (particle, energy)
        if key not in chains:
            continue

        chain = chains[key]
        safe_tag = f"{particle}_{sanitize_name(energy)}"
        hname = f"h_avg_scifi_xy_{safe_tag}"
        htitle = f"{particle} {energy};avg_scifi_x;avg_scifi_y"

        hist, n_drawn = draw_th2_from_chain(
            chain=chain,
            hist_name=hname,
            hist_title=htitle,
            expr_y="avg_scifi_y",
            expr_x="avg_scifi_x",
            nbins_x=NBINS_2D_X,
            x_min=XMIN_2D,
            x_max=XMAX_2D,
            nbins_y=NBINS_2D_Y,
            y_min=YMIN_2D,
            y_max=YMAX_2D,
            selection=""
        )

        if not hist:
            print(f"[INFO] No entries drawn for avg_scifi plot: {particle}, {energy}")
            continue

        all_hists[hname] = hist

        cname = f"c_{hname}"
        canvas = ROOT.TCanvas(cname, "", 800, 700)
        canvas.SetRightMargin(0.14)
        hist.Draw("COLZ")
        save_canvas(canvas, scifi_outdir, hname)
        all_canvases[cname] = canvas

# =========================================================
# 2) 2D plot: x vs y
# =========================================================

xy_outdir = os.path.join(output_dir, "avg_xy")
ensure_dir(xy_outdir)

for particle in particle_order:
    for energy in energy_order:
        key = (particle, energy)
        if key not in chains:
            continue

        chain = chains[key]
        safe_tag = f"{particle}_{sanitize_name(energy)}"
        hname = f"h_avg_xy_{safe_tag}"
        htitle = f"{particle} {energy};x;y"

        hist, n_drawn = draw_th2_from_chain(
            chain=chain,
            hist_name=hname,
            hist_title=htitle,
            expr_y="y",
            expr_x="x",
            nbins_x=NBINS_2D_X,
            x_min=XMIN_2D,
            x_max=XMAX_2D,
            nbins_y=NBINS_2D_Y,
            y_min=YMIN_2D,
            y_max=YMAX_2D,
            selection=""
        )

        if not hist:
            print(f"[INFO] No entries drawn for x-y plot: {particle}, {energy}")
            continue

        all_hists[hname] = hist

        cname = f"c_{hname}"
        canvas = ROOT.TCanvas(cname, "", 800, 700)
        canvas.SetRightMargin(0.14)
        hist.Draw("COLZ")
        save_canvas(canvas, xy_outdir, hname)
        all_canvases[cname] = canvas

# =========================================================
# 3) theta_x and theta_y histograms
# =========================================================

theta_outdir = os.path.join(output_dir, "theta")
ensure_dir(theta_outdir)

theta_x_hists = {}
theta_y_hists = {}

for particle in particle_order:
    for energy in energy_order:
        key = (particle, energy)
        if key not in chains:
            continue

        chain = chains[key]
        safe_tag = f"{particle}_{sanitize_name(energy)}"

        hname_x = f"h_theta_x_{safe_tag}"
        htitle_x = f"{particle} {energy};theta_x = px/pz;Normalized entries"

        hname_y = f"h_theta_y_{safe_tag}"
        htitle_y = f"{particle} {energy};theta_y = py/pz;Normalized entries"

        hist_x, n_x = draw_th1_from_chain(
            chain=chain,
            hist_name=hname_x,
            hist_title=htitle_x,
            expr="(px/pz)",
            nbins=THETA_BINS,
            x_min=THETA_MIN,
            x_max=THETA_MAX,
            selection="pz!=0"
        )

        hist_y, n_y = draw_th1_from_chain(
            chain=chain,
            hist_name=hname_y,
            hist_title=htitle_y,
            expr="(py/pz)",
            nbins=THETA_BINS,
            x_min=THETA_MIN,
            x_max=THETA_MAX,
            selection="pz!=0"
        )

        if hist_x:
            normalize_hist(hist_x)
            style_hist(hist_x, particle, energy)
            theta_x_hists[key] = hist_x
            all_hists[hname_x] = hist_x
        else:
            print(f"[INFO] No entries drawn for theta_x: {particle}, {energy}")

        if hist_y:
            normalize_hist(hist_y)
            style_hist(hist_y, particle, energy)
            theta_y_hists[key] = hist_y
            all_hists[hname_y] = hist_y
        else:
            print(f"[INFO] No entries drawn for theta_y: {particle}, {energy}")

# =========================================================
# Overlay: all theta_x in one plot
# =========================================================

if theta_x_hists:
    cname = "c_theta_x_all"
    canvas = ROOT.TCanvas(cname, "", 1000, 800)
    legend = ROOT.TLegend(0.62, 0.55, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    first = True
    ymax = 0.0
    for energy in energy_order:
        for particle in particle_order:
            key = (particle, energy)
            if key in theta_x_hists:
                ymax = max(ymax, theta_x_hists[key].GetMaximum())

    for energy in energy_order:
        for particle in particle_order:
            key = (particle, energy)
            if key not in theta_x_hists:
                continue

            hist = theta_x_hists[key]
            hist.SetTitle("theta_x overlays;theta_x = px/pz;Normalized entries")
            hist.SetMaximum(1.15 * ymax if ymax > 0 else 1.0)

            if first:
                hist.Draw("hist")
                first = False
            else:
                hist.Draw("hist same")

            legend.AddEntry(hist, f"{particle}, {energy}", "l")

    legend.Draw()
    save_canvas(canvas, theta_outdir, "theta_x_all")
    all_canvases[cname] = canvas

# =========================================================
# Overlay: all theta_y in one plot
# =========================================================

if theta_y_hists:
    cname = "c_theta_y_all"
    canvas = ROOT.TCanvas(cname, "", 1000, 800)
    legend = ROOT.TLegend(0.62, 0.55, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    first = True
    ymax = 0.0
    for energy in energy_order:
        for particle in particle_order:
            key = (particle, energy)
            if key in theta_y_hists:
                ymax = max(ymax, theta_y_hists[key].GetMaximum())

    for energy in energy_order:
        for particle in particle_order:
            key = (particle, energy)
            if key not in theta_y_hists:
                continue

            hist = theta_y_hists[key]
            hist.SetTitle("theta_y overlays;theta_y = py/pz;Normalized entries")
            hist.SetMaximum(1.15 * ymax if ymax > 0 else 1.0)

            if first:
                hist.Draw("hist")
                first = False
            else:
                hist.Draw("hist same")

            legend.AddEntry(hist, f"{particle}, {energy}", "l")

    legend.Draw()
    save_canvas(canvas, theta_outdir, "theta_y_all")
    all_canvases[cname] = canvas

# =========================================================
# Optional: theta overlays by particle
# =========================================================

for particle in particle_order:
    # theta_x
    cname = f"c_theta_x_{particle}"
    canvas = ROOT.TCanvas(cname, "", 1000, 800)
    legend = ROOT.TLegend(0.62, 0.55, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    first = True
    ymax = 0.0
    for energy in energy_order:
        key = (particle, energy)
        if key in theta_x_hists:
            ymax = max(ymax, theta_x_hists[key].GetMaximum())

    for energy in energy_order:
        key = (particle, energy)
        if key not in theta_x_hists:
            continue

        hist = theta_x_hists[key]
        hist.SetTitle(f"theta_x overlays ({particle});theta_x = px/pz;Normalized entries")
        hist.SetMaximum(1.15 * ymax if ymax > 0 else 1.0)

        if first:
            hist.Draw("hist")
            first = False
        else:
            hist.Draw("hist same")

        legend.AddEntry(hist, energy, "l")

    if not first:
        legend.Draw()
        save_canvas(canvas, theta_outdir, f"theta_x_{particle}")
        all_canvases[cname] = canvas

    # theta_y
    cname = f"c_theta_y_{particle}"
    canvas = ROOT.TCanvas(cname, "", 1000, 800)
    legend = ROOT.TLegend(0.62, 0.55, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    first = True
    ymax = 0.0
    for energy in energy_order:
        key = (particle, energy)
        if key in theta_y_hists:
            ymax = max(ymax, theta_y_hists[key].GetMaximum())

    for energy in energy_order:
        key = (particle, energy)
        if key not in theta_y_hists:
            continue

        hist = theta_y_hists[key]
        hist.SetTitle(f"theta_y overlays ({particle});theta_y = py/pz;Normalized entries")
        hist.SetMaximum(1.15 * ymax if ymax > 0 else 1.0)

        if first:
            hist.Draw("hist")
            first = False
        else:
            hist.Draw("hist same")

        legend.AddEntry(hist, energy, "l")

    if not first:
        legend.Draw()
        save_canvas(canvas, theta_outdir, f"theta_y_{particle}")
        all_canvases[cname] = canvas

# =========================================================
# Optional: theta overlays by energy
# =========================================================

for energy in energy_order:
    # theta_x
    cname = f"c_theta_x_{sanitize_name(energy)}"
    canvas = ROOT.TCanvas(cname, "", 1000, 800)
    legend = ROOT.TLegend(0.62, 0.70, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    first = True
    ymax = 0.0
    for particle in particle_order:
        key = (particle, energy)
        if key in theta_x_hists:
            ymax = max(ymax, theta_x_hists[key].GetMaximum())

    for particle in particle_order:
        key = (particle, energy)
        if key not in theta_x_hists:
            continue

        hist = theta_x_hists[key]
        hist.SetTitle(f"theta_x overlays ({energy});theta_x = px/pz;Normalized entries")
        hist.SetMaximum(1.15 * ymax if ymax > 0 else 1.0)

        if first:
            hist.Draw("hist")
            first = False
        else:
            hist.Draw("hist same")

        legend.AddEntry(hist, particle, "l")

    if not first:
        legend.Draw()
        save_canvas(canvas, theta_outdir, f"theta_x_{sanitize_name(energy)}")
        all_canvases[cname] = canvas

    # theta_y
    cname = f"c_theta_y_{sanitize_name(energy)}"
    canvas = ROOT.TCanvas(cname, "", 1000, 800)
    legend = ROOT.TLegend(0.62, 0.70, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    first = True
    ymax = 0.0
    for particle in particle_order:
        key = (particle, energy)
        if key in theta_y_hists:
            ymax = max(ymax, theta_y_hists[key].GetMaximum())

    for particle in particle_order:
        key = (particle, energy)
        if key not in theta_y_hists:
            continue

        hist = theta_y_hists[key]
        hist.SetTitle(f"theta_y overlays ({energy});theta_y = py/pz;Normalized entries")
        hist.SetMaximum(1.15 * ymax if ymax > 0 else 1.0)

        if first:
            hist.Draw("hist")
            first = False
        else:
            hist.Draw("hist same")

        legend.AddEntry(hist, particle, "l")

    if not first:
        legend.Draw()
        save_canvas(canvas, theta_outdir, f"theta_y_{sanitize_name(energy)}")
        all_canvases[cname] = canvas

print(f"\nDone. Plots saved under: {output_dir}")