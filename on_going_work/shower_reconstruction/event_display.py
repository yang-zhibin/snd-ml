import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import array
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import re
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo

def get_detector_layout(geo_file):
    """
    Extract detector component geometry from an SND@LHC/FairRoot geofile.

    Returns
    -------
    dict[str, list[dict]]:
        Keys: wall, mat, veto_bar, us_bar, ds_bar, block
        Each entry has: name, x, y, z, dx, dy, dz, ver, hor, path
    """

    # ---- Regex patterns for relevant subdetectors ----
    WALL_RE      = re.compile(r"^volWallborder(_\d+)?$", re.IGNORECASE)
    MAT_RE       = re.compile(r"^(Hor|Vert)?MatVolume(_\d+)?$", re.IGNORECASE)
    VETO_BAR_RE  = re.compile(r"^volVetoBar(_ver)?(_\d+)?$", re.IGNORECASE)
    US_BAR_RE    = re.compile(r"^volMuUpstreamBar(_(hor|ver))?(_\d+)?$", re.IGNORECASE)
    DS_BAR_RE    = re.compile(r"^volMuDownstreamBar(_(hor|ver))?(_\d+)?$", re.IGNORECASE)
    BLOCK_RE     = re.compile(r"^volFeBlock(_\d+)?$", re.IGNORECASE)

    # ---- Ancestor name hints ----
    IS_TARGET   = lambda path: any(n.startswith("volTarget") for n in path)
    IS_VETO     = lambda path: any(n.startswith("volVeto") for n in path)
    IS_VETO_PL  = lambda name: name.startswith("volVetoPlane")
    IS_MUFILT   = lambda path: any(n.startswith("volMuFilter") for n in path)
    IS_US_DET   = lambda name: name.startswith("volMuUpstreamDet")
    IS_DS_DET   = lambda name: name.startswith("volMuDownstreamDet")

    def get_global_xyz(mat):
        t = mat.GetTranslation()
        return float(t[0]), float(t[1]), float(t[2])

    # ---- open geometry ----
    f = ROOT.TFile.Open(geo_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open geometry file: {geo_file}")
    geom = f.Get("FAIRGeom")
    if not geom:
        raise RuntimeError("TGeoManager 'FAIRGeom' not found in file")
    top = geom.GetTopNode()
    if not top:
        raise RuntimeError("No top node in geometry")

    out = defaultdict(list)

    # ---- DFS traversal ----
    stack = [(top, ROOT.TGeoHMatrix(), [top.GetVolume().GetName()])]
    while stack:
        node, M, path = stack.pop()
        vol = node.GetVolume()
        if not vol:
            continue

        for i in range(node.GetNdaughters()):
            ch = node.GetDaughter(i)
            ch_vol = ch.GetVolume()
            if not ch_vol:
                continue
            ch_name = ch_vol.GetName()

            # compose global transform
            M_child = ROOT.TGeoHMatrix(M)
            M_child.Multiply(ch.GetMatrix())
            path_child = path + [ch_name]
            x, y, z = get_global_xyz(M_child)

            # get bounding box dimensions
            shp = ch_vol.GetShape()
            dx = getattr(shp, "GetDX", lambda: 0)()
            dy = getattr(shp, "GetDY", lambda: 0)()
            dz = getattr(shp, "GetDZ", lambda: 0)()

            # --- Orientation logic ---
            ver, hor = 0, 0

            if WALL_RE.match(ch_name):
                ver, hor = 1, 1
            elif US_BAR_RE.match(ch_name):
                ver, hor = 0, 1
            elif VETO_BAR_RE.match(ch_name):
                if "ver" in ch_name.lower():
                    ver, hor = 1, 0
                else:
                    ver, hor = 0, 1
            elif BLOCK_RE.match(ch_name):
                ver, hor = 1, 1
            else:
                if "ver" in ch_name.lower() or any("ver" in p.lower() for p in path_child):
                    ver, hor = 1, 0
                elif "hor" in ch_name.lower() or any("hor" in p.lower() for p in path_child):
                    ver, hor = 0, 1

            # ---- Classification ----
            # 1) Wallborder under Target
            if WALL_RE.match(ch_name) and IS_TARGET(path_child):
                out["wall"].append(dict(
                    name=ch_name, x=x, y=y, z=z,
                    dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
                ))

            # 2) SciFi material layers under Target
            if MAT_RE.match(ch_name) and IS_TARGET(path_child):
                out["mat"].append(dict(
                    name=ch_name, x=x, y=y, z=z,
                    dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
                ))

            # 3) Veto bars
            if VETO_BAR_RE.match(ch_name) and IS_VETO(path_child):
                if any(IS_VETO_PL(n) for n in path_child):
                    out["veto_bar"].append(dict(
                        name=ch_name, x=x, y=y, z=z,
                        dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
                    ))

            # 4) MuFilter upstream bars
            if US_BAR_RE.match(ch_name) and IS_MUFILT(path_child):
                if any(IS_US_DET(n) for n in path_child):
                    out["us_bar"].append(dict(
                        name=ch_name, x=x, y=y, z=z,
                        dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
                    ))

            # 5) MuFilter downstream bars
            if DS_BAR_RE.match(ch_name) and IS_MUFILT(path_child):
                if any(IS_DS_DET(n) for n in path_child):
                    out["ds_bar"].append(dict(
                        name=ch_name, x=x, y=y, z=z,
                        dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
                    ))

            # 6) MuFilter Fe blocks
            if BLOCK_RE.match(ch_name) and IS_MUFILT(path_child):
                out["block"].append(dict(
                    name=ch_name, x=x, y=y, z=z,
                    dx=dx, dy=dy, dz=dz, ver=1, hor=1, path=tuple(path_child)
                ))

            # continue recursion
            stack.append((ch, M_child, path_child))

    return dict(out)


def get_detector_layout_testbeam(geo_file):
    """
    Extract detector component geometry from an SND@LHC/FairRoot geofile.

    Returns
    -------
    dict[str, list[dict]]:
        Keys: wall, mat
        Each entry has: name, x, y, z, dx, dy, dz, ver, hor, path
    """

    # ---- Regex patterns for relevant subdetectors ----
    WALL_RE      = re.compile(r"^volWallborder(_\d+)?$", re.IGNORECASE)
    MAT_RE       = re.compile(r"^(Hor|Vert)?MatVolume(_\d+)?$", re.IGNORECASE)

    # ---- Ancestor name hints ----
    IS_TARGET   = lambda path: any(n.startswith("volTarget") for n in path)
   

    def get_global_xyz(mat):
        t = mat.GetTranslation()
        return float(t[0]), float(t[1]), float(t[2])

    # ---- open geometry ----
    f = ROOT.TFile.Open(geo_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open geometry file: {geo_file}")
    geom = f.Get("FAIRGeom")
    if not geom:
        raise RuntimeError("TGeoManager 'FAIRGeom' not found in file")
    top = geom.GetTopNode()
    if not top:
        raise RuntimeError("No top node in geometry")

    out = defaultdict(list)

    # ---- DFS traversal ----
    stack = [(top, ROOT.TGeoHMatrix(), [top.GetVolume().GetName()])]
    while stack:
        node, M, path = stack.pop()
        vol = node.GetVolume()
        if not vol:
            continue

        for i in range(node.GetNdaughters()):
            ch = node.GetDaughter(i)
            ch_vol = ch.GetVolume()
            if not ch_vol:
                continue
            ch_name = ch_vol.GetName()

            # compose global transform
            M_child = ROOT.TGeoHMatrix(M)
            M_child.Multiply(ch.GetMatrix())
            path_child = path + [ch_name]
            x, y, z = get_global_xyz(M_child)

            # get bounding box dimensions
            shp = ch_vol.GetShape()
            dx = getattr(shp, "GetDX", lambda: 0)()
            dy = getattr(shp, "GetDY", lambda: 0)()
            dz = getattr(shp, "GetDZ", lambda: 0)()

            # --- Orientation logic ---
            ver, hor = 0, 0

            if WALL_RE.match(ch_name):
                ver, hor = 1, 1
            else:
                if "ver" in ch_name.lower() or any("ver" in p.lower() for p in path_child):
                    ver, hor = 1, 0
                elif "hor" in ch_name.lower() or any("hor" in p.lower() for p in path_child):
                    ver, hor = 0, 1

            # ---- Classification ----
            # 1) Wallborder under Target
            if WALL_RE.match(ch_name) and IS_TARGET(path_child):
                out["wall"].append(dict(
                    name=ch_name, x=x, y=y, z=z,
                    dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
                ))

            # 2) SciFi material layers under Target
            if MAT_RE.match(ch_name) and IS_TARGET(path_child):
                out["mat"].append(dict(
                    name=ch_name, x=x, y=y, z=z,
                    dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
                ))

            # continue recursion
            stack.append((ch, M_child, path_child))

    return dict(out)



def open_root_file(file_path, mode='read'):
    """Open and return a ROOT file and its primary tree."""
    file = ROOT.TFile(file_path, mode)
    tree = file.Get('cbmsim')
    return file, tree

def create_output_file(path, mode):
    """Create and return a new ROOT file and a new tree for output."""
    out_file = ROOT.TFile(path, mode)
    new_tree = ROOT.TTree('sndData', 'converted cbmsim tree')
    return out_file, new_tree

def shower_id_legend_handles():
    """Legend entries matching the fixed mapping above."""
    items = [
        ("EM shower (11)",          shower_color(11)),
        ("Muon (13)",               shower_color(13)),
        ("Tau shower (15)",         shower_color(15)),
        ("NC shower (112/114/116)", shower_color(112)),
        ("Hadronic shower (0)",     shower_color(0)),
        ("Combine (-1)",            shower_color(-1)),
        ("Unassigned/invalid (-2)", shower_color(-2)),
    ]
    return [Line2D([0],[0], marker='o', linestyle='',
                   markersize=8, markerfacecolor=c, markeredgecolor='none')
            for _, c in items], [t for t, _ in items]
    
def shower_color(sid: int):
    """
    Fixed discrete colors for showerId:
        11  -> blue   (EM shower)
        13  -> red    (muon)
        15  -> orange (Tau shower)
        112/114/116 -> green (NC shower)
        0   -> purple (Hadronic shower)
        -1  -> grey   (combine)
        -2  -> black  (unassigned or invalid)
    """
    sid = int(sid)
    if sid in (112, 114, 116):   # NC-like
        return mcolors.to_rgba("tab:green")
    if sid == 11:
        return mcolors.to_rgba("tab:blue")
    if sid == 13:
        return mcolors.to_rgba("tab:red")
    if sid == 15:
        return mcolors.to_rgba("tab:orange")
    if sid == 0:
        return mcolors.to_rgba("tab:purple")
    if sid == -1:
        return mcolors.to_rgba("tab:gray")
    if sid == -2:
        return mcolors.to_rgba("black")
    # fallback
    return mcolors.to_rgba("tab:gray")

    

def plot_clustering(all_hits, det_layout, event_dict, args):
    group_color = {
        "wall":      "tab:gray",
        "mat":       "tab:blue",
        "veto_bar":  "tab:red",
        "us_bar":    "tab:orange",
        "ds_bar":    "tab:purple",
        "block":     "tab:green",
    }

    # ---- helpers -------------------------------------------------------------

    def setup_ax(ax, title, xlabel, ylabel):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.tick_params(direction="in")
        ax.grid(False)

    def add_rect(ax, z, c, half_w, half_h, color, fill=False, lw=0.6, alpha=0.8):
        ax.add_patch(Rectangle(
            (z - half_w, c - half_h),
            2 * half_w, 2 * half_h,
            fill=fill,
            linewidth=lw,
            edgecolor=color,
            facecolor=color if fill else "none",
            alpha=alpha,
        ))

    # Discrete color for shower IDs. We assign fixed colors to the IDs you listed
    # (positives and negatives are distinguished but share the same color family with different alpha).


    # ---- figure layout (left = QDC, right = ShowerID) -----------------------

    fig, axes = plt.subplots(2, 2, figsize=(30, 16), facecolor="white")
    (ax_xz_qdc, ax_xz_shw), (ax_yz_qdc, ax_yz_shw) = axes

    setup_ax(ax_xz_qdc, "XZ View (QDC)", "Z [cm]", "X [cm]")
    setup_ax(ax_yz_qdc, "YZ View (QDC)", "Z [cm]", "Y [cm]")
    setup_ax(ax_xz_shw, "XZ View (Shower ID)", "Z [cm]", "X [cm]")
    setup_ax(ax_yz_shw, "YZ View (Shower ID)", "Z [cm]", "Y [cm]")

    # ---- draw detector layout on all four axes ------------------------------

    for group, items in det_layout.items():
        if group == 'mat':
            continue
        color = group_color.get(group, "k")
        for it in items:
            x, y, z = it["x"], it["y"], it["z"]
            dx, dy, dz = float(it["dx"]), float(it["dy"]), float(it["dz"])
            ver, hor   = int(it["ver"]), int(it["hor"])
            fill = group in {"wall", "block"}

            if ver == 1:
                add_rect(ax_xz_qdc, z, x, dz, dx, color, fill=fill, alpha=0.4 if fill else 0.8)
                add_rect(ax_xz_shw, z, x, dz, dx, color, fill=fill, alpha=0.4 if fill else 0.8)
            if hor == 1:
                add_rect(ax_yz_qdc, z, y, dz, dy, color, fill=fill, alpha=0.4 if fill else 0.8)
                add_rect(ax_yz_shw, z, y, dz, dy, color, fill=fill, alpha=0.4 if fill else 0.8)

    # ---- bar dimensions per orientation (veto/us/ds) ------------------------

    dim_lookup = {}
    for group in ["veto_bar", "us_bar", "ds_bar"]:
        dim_lookup[group] = {"ver": None, "hor": None}
        for it in det_layout.get(group, []):
            if it["ver"] == 1 and dim_lookup[group]["ver"] is None:
                dim_lookup[group]["ver"] = (it["dx"], it["dy"], it["dz"])
            if it["hor"] == 1 and dim_lookup[group]["hor"] is None:
                dim_lookup[group]["hor"] = (it["dx"], it["dy"], it["dz"])

    # ---- QDC colormaps & scales (SciFi vs bars) -----------------------------

    scifi_qdcs = [h["qdc"] for h in all_hits if h["detType"] == 0 and h["qdc"] > 0]
    bar_qdcs   = [h["qdc"] for h in all_hits if h["detType"] in [1,2,3] and h["qdc"] > 0]
    sci_min, sci_max = (min(scifi_qdcs), max(scifi_qdcs)) if scifi_qdcs else (0, 1)
    bar_min, bar_max = (min(bar_qdcs),   max(bar_qdcs))   if bar_qdcs   else (0, 1)
    cmap_scifi = plt.cm.viridis   # SciFi continuous
    cmap_bar   = plt.cm.viridis  # Bars continuous, as requested

    # ---- draw hits ----------------------------------------------------------

    for hit in all_hits:
        det_type = hit["detType"]   # 0=scifi, 1=veto, 2=us, 3=ds
        x, y, z  = hit["x"], hit["y"], hit["z"]
        qdc      = hit["qdc"]
        sid      = hit["hit_shower_id"]
        is_vert  = hit["isVertical"]

        # QDC colors (left figure)
        if det_type == 0:
            cval = (qdc - sci_min) / (sci_max - sci_min + 1e-9)
            c_qdc = cmap_scifi(np.clip(cval, 0, 1))
            ax_xz_qdc.scatter(z, x, s=10, c=[c_qdc], marker="o", edgecolors="none")
            ax_yz_qdc.scatter(z, y, s=10, c=[c_qdc], marker="o", edgecolors="none")
        else:
            cval = (qdc - bar_min) / (bar_max - bar_min + 1e-9)
            c_qdc = cmap_bar(np.clip(cval, 0, 1))
            group = {1: "veto_bar", 2: "us_bar", 3: "ds_bar"}[det_type]
            orient = "ver" if is_vert else "hor"
            dx, dy, dz = dim_lookup.get(group, {}).get(orient, (0.5, 0.5, 0.5))
            if is_vert:
                add_rect(ax_xz_qdc, z, x, dz, dx, c_qdc, fill=True, alpha=0.9)
            else:
                add_rect(ax_yz_qdc, z, y, dz, dy, c_qdc, fill=True, alpha=0.9)

        # Shower-ID colors (right figure)
        c_shw = shower_color(sid)
        if det_type == 0:
            ax_xz_shw.scatter(z, x, s=10, c=[c_shw], marker="o", edgecolors="none")
            ax_yz_shw.scatter(z, y, s=10, c=[c_shw], marker="o", edgecolors="none")
        else:
            group = {1: "veto_bar", 2: "us_bar", 3: "ds_bar"}[det_type]
            orient = "ver" if is_vert else "hor"
            dx, dy, dz = dim_lookup.get(group, {}).get(orient, (0.5, 0.5, 0.5))
            if is_vert:
                add_rect(ax_xz_shw, z, x, dz, dx, c_shw, fill=True, alpha=0.9)
            else:
                add_rect(ax_yz_shw, z, y, dz, dy, c_shw, fill=True, alpha=0.9)

    # ---- colorbars for QDC-only (left column) -------------------------------

    sm_scifi = plt.cm.ScalarMappable(cmap=cmap_scifi, norm=plt.Normalize(vmin=sci_min, vmax=sci_max))
    sm_bar   = plt.cm.ScalarMappable(cmap=cmap_bar,   norm=plt.Normalize(vmin=bar_min,  vmax=bar_max))
    #fig.colorbar(sm_scifi, ax=[ax_xz_qdc, ax_yz_qdc], fraction=0.015, pad=0.01, label="SciFi QDC")
    #fig.colorbar(sm_bar,   ax=[ax_xz_qdc, ax_yz_qdc], fraction=0.015, pad=0.06, label="Veto/US/DS QDC")
    handles, labels = shower_id_legend_handles()
    ax_yz_shw.legend(
        handles, labels,
        title="Shower ID colors",
        loc="center left",            # anchor to the left edge of bbox_to_anchor
        bbox_to_anchor=(1.02, 0.5),   # x offset = 1.02 moves it outside the axes
        frameon=False,
        fontsize=10,
        title_fontsize=11,
    )

    # ---- annotation text ----------------------------------------------------

    run_id = event_dict.get("runId", [None])[0]
    evt_id = event_dict.get("eventId", [None])[0]
    pdg    = event_dict.get("pdgCode", [None])[0]  # if available in your tree
    pname = pdg_to_name.get(pdg)
    header = f"type: {getattr(args, 'type', 'NA')}   run: {run_id}   evtId: {evt_id}   pdgCode: {pdg}, particle: {pname}"
    # Put a single header across the top
    fig.suptitle(header, y=0.985, fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.96, 0.97])

    # ---- finalize & save (vector) ------------------------------------------
    
    os.makedirs("./eventDisplay", exist_ok=True)
    output_path = f"./eventDisplay/{getattr(args,'type','evt')}_run-{run_id}_evtId-{evt_id}_{pname}.pdf"
    plt.savefig(output_path, format="pdf")
    
    plt.close()
    print(f"Saved event display to {output_path}")


def build_mother_to_daughters(event):
    """Return dict: mother_index -> [daughter_indices] for event.MCTrack."""
    m2d = {}
    n = event.MCTrack.GetEntries()
    for i in range(n):
        mid = event.MCTrack[i].GetMotherId()
        m2d.setdefault(mid, []).append(i)
        
    print(m2d)
    return m2d

def _categorize_seed_shower_id(track, event_level_pdg):
    """
    Decide the showerId for a *seed* track (motherId==0).
    Rules:
      - e/μ/τ -> ±11 / ±13 / ±15
      - νe/νμ/ντ -> ±112/±114/±116 if event is NC-like; else keep ±12/±14/±16
      - hadronic/other -> 0
    """
    pdg = int(track.GetPdgCode()) if hasattr(track, "GetPdgCode") else 0
    apdg = abs(pdg)

    # Heuristic: event-level NC flags often look like 112/114/116
    # If event pdgCode matches those (or starts with '11' and length 3), use 100+flavor for neutrinos.
    is_nc_event = str(event_level_pdg) in {"112", "114", "116", "-112", "-114", "-116"} 

    if apdg in (11, 13, 15):
        return pdg  # ±11/±13/±15
    if apdg in (12, 14, 16):
        return int(math.copysign(100 + apdg, pdg)) if is_nc_event else pdg
    return 0  # hadronic/other

def _assign_shower_ids(event, m2d, event_level_pdg):
    """
    Build dict: trackId -> showerId, seeding on tracks with motherId==0
    and propagating showerId to all descendants.
    """
    n = event.MCTrack.GetEntries()
    shower = {}

    # Seeds = immediate daughters of the primary (MCTrack[0] usually has mother=-1).
    seed_ids = [i for i in range(n) if event.MCTrack[i].GetMotherId() == 0]

    print('seed:',seed_ids)
    # Determine showerId for each seed and flood-fill to its descendants
    for sid in seed_ids:
        seed_track = event.MCTrack[sid]
        shower_id = _categorize_seed_shower_id(seed_track, event_level_pdg)

        # DFS
        stack = [sid]
        while stack:
            t = stack.pop()
            if t in shower:
                continue
            shower[t] = shower_id
            stack.extend(m2d.get(t, []))

    # Any leftover tracks (not reachable from seeds) -> hadron(0) by default
    for i in range(n):
        if i not in shower:
            shower[i] = 0

    return shower

def _fmt_track_line(tr, idx, shower_id):
    """Nice one-line summary with start pos + momentum."""
    pdg = tr.GetPdgCode() if hasattr(tr, "GetPdgCode") else None
    x = tr.GetStartX() if hasattr(tr, "GetStartX") else float("nan")
    y = tr.GetStartY() if hasattr(tr, "GetStartY") else float("nan")
    z = tr.GetStartZ() if hasattr(tr, "GetStartZ") else float("nan")
    mother_id = tr.GetMotherId()
    # momentum
    vec = None
    if hasattr(tr, "GetMomentum"):
        try:
            vec = tr.GetMomentum()
        except Exception:
            vec = None
    if not isinstance(vec, ROOT.TVector3):
        px = tr.GetPx() if hasattr(tr, "GetPx") else 0.0
        py = tr.GetPy() if hasattr(tr, "GetPy") else 0.0
        pz = tr.GetPz() if hasattr(tr, "GetPz") else 0.0
        vec = ROOT.TVector3(px, py, pz)

    return f"Track {idx:>3}  PDG={pdg:>4}, mother_id={mother_id:>4}  showerId={shower_id:>4}  start=({x:.3g},{y:.3g},{z:.3g})  p=({vec.X():.3g},{vec.Y():.3g},{vec.Z():.3g}) |p|={vec.Mag():.3g}"

def print_track_tree(event, m2d, track_idx, shower_map, indent=0):
    """Recursive pretty-printer including showerId, start pos, momentum."""
    tr = event.MCTrack[track_idx]
    pad = " " * indent
    line = _fmt_track_line(tr, track_idx, shower_map.get(track_idx, 0))
    print(pad + line)
    for d in m2d.get(track_idx, []):
        print_track_tree(event, m2d, d, shower_map, indent + 4)


    
def process_hits(event, snd_geo, event_dict, det_layout, args, pdf):
    """Process all hits in the event and update hits array and averages."""
    MC = args.type
    Scifi = snd_geo.modules['Scifi']
    MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()
    
    eventId = event_dict["eventId"]
    pdgCode = event_dict["pdgCode"]
    
    #read plane positions, vertical->top_x->A.x, horizontal->right_y->A.y


    # pdgCode, ve:12/-12, vm:14/-14, vt:16/-16, NC:112/-112/114/-114/116/-116
    # need a dictionary: trackId -> showerId (e:11/-11, muon:13/-13, tau:15/-15, NC:12/-12/13/-13/15/-15, hadron: 0)
    # 1.find all the track from motherId=0
    # 2. assign showerId to these track
    # 3. assign the same showerId to the sub-track of these track
    
    # --- Build hierarchy and shower mapping ---
    m2d = build_mother_to_daughters(event)


    shower_map = _assign_shower_ids(event, m2d, pdgCode)

    # Now actually print the tree(s)
    primary_tracks = m2d.get(-1, [])
    for idx in primary_tracks:
        print_track_tree(event, m2d, idx, shower_map, indent=0)
    
    print(dir(event.MCTrack[0]))
    #print(shower_map)

    # Temporary storage for all hits with positions
    all_hits = []
    # SciFi hits
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        station = detID // 1000000

        Scifi.GetSiPMPosition(detID, A, B)

        max_QDC = 200 * 16
        this_qdc = 0
        ns = max(1,aHit.GetnSides())
        for side in range(ns):
            for m in  range(aHit.GetnSiPMs()):
                qdc = aHit.GetSignal(m+side*aHit.GetnSiPMs())
                if not qdc < 0:
                    this_qdc += qdc
        if this_qdc > max_QDC :
            this_qdc = max_QDC
        hit_time = aHit.GetTime()
        
        if ('MC' in  args.type):
            hit2MC = event.Digi_ScifiHits2MCPoints[0]
            linksToMCPoints = hit2MC.wList(detID)
            shower_id_list = []
            for mc_point_i, weight in linksToMCPoints:  
                scifi_point = event.ScifiPoint[mc_point_i]
                track_id = scifi_point.GetTrackID()
                shower_id = shower_map.get(track_id)
                if shower_id is None:
                    continue
                shower_id_list.append(shower_id)
            
            unique_ids = set(shower_id_list)
            if len(unique_ids) == 1:
                hit_shower_id = next(iter(unique_ids))   
            elif len(unique_ids) > 1:   
                hit_shower_id = -1  
            else:
                hit_shower_id = -2
                
            
            
            # break
        
        all_hits.append({
            "detType": 0,
            "station": station,
            "isVertical": aHit.isVertical(),
            "x":A.x(),
            "y":A.y(),
            "z":A.z(),
            "qdc":this_qdc,
            "hit_time": hit_time,
            "hit_shower_id": hit_shower_id
            
        })

    # MuFilter hits
    

    n_veto_hit = 0
    
    for aHit in event.Digi_MuFilterHits:
        
        
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        detType = aHit.GetSystem()
        station = (detID // 1000) % 10

        MuFilter.GetPosition(detID, A, B)

        max_QDC = 200 * 16
        this_qdc = 0
        ns = max(1,aHit.GetnSides())
        for side in range(ns):
            for m in  range(aHit.GetnSiPMs()):
                qdc = aHit.GetSignal(m+side*aHit.GetnSiPMs())
                if not qdc < 0:
                    this_qdc += qdc
        if this_qdc > max_QDC :
            this_qdc = max_QDC
        hit_time = aHit.GetTime()
        
        if ('MC' in  args.type):
            hit2MC = event.Digi_MuFilterHits2MCPoints[0]
            linksToMCPoints = hit2MC.wList(detID)
            start_z = (
                event.MCTrack[1].GetStartZ()
                if event.MCTrack.GetEntries() > 1
                else -999
                )
            event_dict["start_z"] = start_z
                
                
  
        all_hits.append({
            "detType": detType,
            "station": station+1,
            "isVertical": aHit.isVertical(),
            "x":A.x(),
            "y":A.y(),
            "z":A.z(),
            "qdc":this_qdc,
            "hit_time": hit_time,
            "hit_shower_id": -2
        })
    
    #plot_clustering(all_hits, det_layout, event_dict, args)
    
    return 

def main(args):
    print(f"start plotting events")
    snd_geo = setup_geometry(args.geo_path )
    print("getting geo layout")
    if args.beam == "TI18":
        det_layout = get_detector_layout(args.geo_path)    
    else:
        det_layout = get_detector_layout_testbeam(args.geo_path)
    
    raw_data, raw_tree = open_root_file(args.digi_path)
    

    
    # Process each event
    scifi_count_threshold = args.n_scifi
    print(f"Data type: {args.type}, scifi_count_threshold:{scifi_count_threshold}")
    
    
    event_dict = {}
    
    
    base = os.path.basename(args.digi_path)          # e.g. "run00123.root"
    base = os.path.splitext(base)[0]   
    out_pdf = f'./plot_event_display/{base}.pdf' 
    os.makedirs('./plot_event_display', exist_ok=True)
    with PdfPages("multipage_plots.pdf") as pdf:
        for i_event, event in tqdm(enumerate(raw_tree), total=raw_tree.GetEntries()):
            #if i_event % 10000 == 0:
            #    print(f"processed {i_event} events")
            
            
            event_dict["eventIndex"] = i_event
            event_dict["runId"] = event.EventHeader.GetRunId()

            if ('MC' in  args.type):
                event_dict["isMC"] = 1
                #print(dir(event.EventHeader))
                
                try:
                    event_dict["eventId"] = event.EventHeader.GetEventNumber()
                except Exception:
                    event_dict["eventId"] = event.EventHeader.GetMCEntryNumber()
                # Particle codes and initial position
                
                event_pdg0 = raw_tree.MCTrack[0].GetPdgCode()
                event_pdg1 = raw_tree.MCTrack[1].GetPdgCode()
                
                event_dict["energy"] = raw_tree.MCTrack[0].GetEnergy()
                

                neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
                if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
                    event_dict["pdgCode"] = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
                else:
                    event_dict["pdgCode"] = event_pdg0


            elif('real' in  args.type):
                event_dict["isMC"] = 0
                event_dict["pdgCode"] = 0
                event_dict["eventId"] = event.EventHeader.GetEventNumber()
            print(f'-------{event_dict["pdgCode"]}---------')
            process_hits(raw_tree, snd_geo, event_dict, det_layout, args, pdf)
            
            if i_event>2:
                break
        

    print("finished")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default='RECREATE')
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)
    parser.add_argument("-b", "--beam", dest='beam', help='testbeam or TI18', default="TI18")
    parser.add_argument("-s", "--n_scifi", dest='n_scifi', help='scifi count threshold', default=1)

    args = parser.parse_args()

    main(args)
    
#    python event_display.py -d /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/1/sndLHC.Genie-TGeant4_20240126_digCPP.root -g /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/1/geofile_full.Genie-TGeant4.root -o ./test_data/vetoTagged_shower_feature.root -t MC_neutrino
