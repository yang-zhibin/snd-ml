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

pdg_to_name = {
    11: 'e-', -11: 'e+',
    211: 'pi+', -211: 'pi-',
    12: 've', -12: 've',
    14: 'vm', -14: 'vm',
    16: 'vt', -16: 'vt',
    112: 'NC', -112: 'NC', 114: 'NC', -114: 'NC', 116: 'NC', -116: 'NC',
    130: 'kaon', 310: 'kaon',
    2112: 'neutron',
    13: 'muon', -13: 'muon' ,
    0: 'data'
}

name_to_pdg = {
    'e-': [11],
    'e+': [-11],
    'pi+': [211],
    'pi-': [-211],
    've': [12, -12],
    'vm': [14, -14],
    'vt': [16, -16],
    'NC': [112, -112, 114, -114, 116, -116],
    'kaon': [130, 310],
    'neutron': [2112],
    'muon': [13, -13],
    'data': [0]
}


# ROOT.gInterpreter.Declare(r"""
# #include <vector>

# struct scifi_point {
#   double pdg = -999;
#   double energy_loss = -999;
# };

# struct VetoHit {
#   double hit_time = -999;
#   double energy_loss = -999;
#   int    veto_plane = -999;
#   double qdc = -999;
#   std::vector<scifi_point> scifiPoints; // nested, variable length
# };

# #ifdef __CLING__
# #pragma link C++ class ScifiPoint+;
# #pragma link C++ class VetoHit+;
# #pragma link C++ class std::vector<ScifiPoint>+;
# #pragma link C++ class std::vector<VetoHit>+;
# #endif
# """)



def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo

def open_root_file(file_path, tree_name='cbmsim', mode='READ'):
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        print(f"⚠️  Skipping corrupted or missing file: {file_path}")
        return None, None
    file = ROOT.TFile(file_path, mode)
    if not file or file.IsZombie():
        print(f"[Warning] Could not open ROOT file: {file_path}")
        return None, None

    tree = file.Get(tree_name)
    if not tree or not isinstance(tree, ROOT.TTree):
        print(f"[Warning] TTree '{tree_name}' not found in {file_path}")
        file.Close()
        return file, None

    return file, tree


def create_output_file(path, mode):
    """Create and return a new ROOT file and a new tree for output."""
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    out_file = ROOT.TFile(path, mode)
    new_tree = ROOT.TTree('sndData', 'converted SND hits tree')
    return out_file, new_tree

def process_counts(all_hits, branch_vars):
    # Reset counters to 0
    for key in [
        "count_scifi1", "count_scifi2", "count_scifi3", "count_scifi4", "count_scifi",
    ]:
        branch_vars[key][0] = 0

    for hit in all_hits:
        detType = hit["detType"]
        station = hit["station"]

        if detType == 0:  # SciFi
            if 1 <= station <= 5:
                branch_vars[f"count_scifi{station}"][0] += 1
            branch_vars["count_scifi"][0] += 1

        elif detType == 1:  # Veto
            if station in [1, 2, 3]:
                branch_vars[f"count_veto{station}"][0] += 1
            branch_vars["count_veto"][0] += 1

        elif detType == 2:  # Upstream
            if 1 <= station <= 5:
                branch_vars[f"count_us{station}"][0] += 1
            branch_vars["count_us"][0] += 1

        elif detType == 3:  # Downstream
            if 1 <= station <= 4:
                branch_vars[f"count_ds{station}"][0] += 1
            branch_vars["count_ds"][0] += 1

def process_avgPos(all_hits, branch_vars):
    #veto_{1-2}_y, 
    #veto_3_x, 
    
    #scifi_{1-5}_{x,y}, 
    #US_{1-5}_y, 
    
    #DS_{1-4}_{x,y}
    
    #y means hotrizontal, and cal the avgPos of y
    #x means vertival, and cal the avgPos of y
    # Create temporary storage for sums and counts
    sums = defaultdict(float)
    counts = defaultdict(int)

    for hit in all_hits:
        detType = hit["detType"]
        station = hit["station"]
        isVertical = hit["isVertical"]

        if detType == 1:  # Veto
            if station in [1, 2] and not isVertical:
                key = f"avg_veto{station}_y"
                sums[key] += hit["y"]
                counts[key] += 1
                
                sums["avg_veto_y"] += hit["y"]
                counts["avg_veto_y"] += 1
                
            elif station == 3 and isVertical:
                key = "avg_veto3_x"
                sums[key] += hit["x"]
                counts[key] += 1
                sums["avg_veto_x"] += hit["x"]
                counts["avg_veto_x"] += 1

        elif detType == 0:  # SciFi
            if 1 <= station <= 5:
                if isVertical:
                    key = f"avg_scifi{station}_x"
                    sums[key] += hit["x"]
                    counts[key] += 1
                    
                    sums["avg_scifi_x"] += hit["x"]
                    counts["avg_scifi_x"] += 1
                else:
                    key = f"avg_scifi{station}_y"
                    sums[key] += hit["y"]
                    counts[key] += 1
                    
                    sums["avg_scifi_y"] += hit["y"]
                    counts["avg_scifi_y"] += 1

        elif detType == 2:  # Upstream (horizontal only)
            if 1 <= station <= 5 and not isVertical:
                key = f"avg_us{station}_y"
                sums[key] += hit["y"]
                counts[key] += 1
                
                sums["avg_us_y"] += hit["y"]
                counts["avg_us_y"] += 1
                

        elif detType == 3:  # Downstream
            if 1 <= station <= 4:
                if isVertical:
                    key = f"avg_ds{station}_x"
                    sums[key] += hit["x"]
                    counts[key] += 1
                    
                    sums["avg_ds_x"] += hit["x"]
                    counts["avg_ds_x"] += 1
                    
                else:
                    key = f"avg_ds{station}_y"
                    sums[key] += hit["y"]
                    counts[key] += 1
                    
                    sums["avg_ds_y"] += hit["y"]
                    counts["avg_ds_y"] += 1
                    
                    

    # Write averages to branch_vars
    for key in sums:
        avg = sums[key] / counts[key] if counts[key] > 0 else -999
        branch_vars[key][0] = avg



def process_centroid(all_hits, branch_vars):
    # Store sum(QDC * pos) and sum(QDC) per plane
    weighted_sums = defaultdict(float)
    total_qdc = defaultdict(float)

    for hit in all_hits:
        detType = hit["detType"]
        station = hit["station"]
        isVertical = hit["isVertical"]
        qdc = hit.get("qdc", 0)

        if qdc <= 0:
            continue

        if detType == 1:  # Veto
            if station in [1, 2] and not isVertical:
                key = f"centroid_veto{station}_y"
                weighted_sums[key] += qdc * hit["y"]
                total_qdc[key] += qdc
                weighted_sums["centroid_veto_y"] += qdc * hit["y"]
                total_qdc["centroid_veto_y"] += qdc
            elif station == 3 and isVertical:
                key = "centroid_veto3_x"
                weighted_sums[key] += qdc * hit["x"]
                total_qdc[key] += qdc
                
                weighted_sums["centroid_veto_x"] += qdc * hit["x"]
                total_qdc["centroid_veto_x"] += qdc

        elif detType == 0:  # SciFi
            if 1 <= station <= 5:
                if isVertical:
                    key = f"centroid_scifi{station}_x"
                    weighted_sums[key] += qdc * hit["x"]
                    total_qdc[key] += qdc
                    
                    weighted_sums["centroid_scifi_x"] += qdc * hit["x"]
                    total_qdc["centroid_scifi_x"] += qdc
                    
                else:
                    key = f"centroid_scifi{station}_y"
                    weighted_sums[key] += qdc * hit["y"]
                    total_qdc[key] += qdc
                    
                    weighted_sums["centroid_scifi_y"] += qdc * hit["y"]
                    total_qdc["centroid_scifi_y"] += qdc


        elif detType == 2:  # Upstream
            if 1 <= station <= 5 and not isVertical:
                key = f"centroid_us{station}_y"
                weighted_sums[key] += qdc * hit["y"]
                total_qdc[key] += qdc
                
                weighted_sums["centroid_us_y"] += qdc * hit["y"]
                total_qdc["centroid_us_y"] += qdc

        elif detType == 3:  # Downstream
            if 1 <= station <= 4:
                if isVertical:
                    key = f"centroid_ds{station}_x"
                    weighted_sums[key] += qdc * hit["x"]
                    total_qdc[key] += qdc
                    
                    weighted_sums["centroid_ds_x"] += qdc * hit["x"]
                    total_qdc["centroid_ds_x"] += qdc
                else:
                    key = f"centroid_ds{station}_y"
                    weighted_sums[key] += qdc * hit["y"]
                    total_qdc[key] += qdc
                    
                    weighted_sums["centroid_ds_y"] += qdc * hit["y"]
                    total_qdc["centroid_ds_y"] += qdc

    for key in weighted_sums:
        centroid = weighted_sums[key] / total_qdc[key] if total_qdc[key] > 0 else -999
        branch_vars[key][0] = centroid

    
def sum_valid_densities(branch_vars, group_keys, target_key):
    total = 0
    for key in group_keys:
        value = branch_vars.get(key, [0])[0]
        if value <=0:
            total += value
    branch_vars[target_key][0] = total
         

def process_hit_density(all_hits, branch_vars):
    plane_hits = defaultdict(list)

    # Group hits by plane
    for hit in all_hits:
        detType = hit["detType"]
        station = hit["station"]
        isVertical = hit["isVertical"]

        if detType == 0 and 1 <= station <= 5:
            key = f"density_scifi{station}"
        elif detType == 1 and station in [1, 2, 3]:
            key = f"density_veto{station}"
        elif detType == 2 and 1 <= station <= 5:
            key = f"density_us{station}"
        elif detType == 3 and 1 <= station <= 4:
            key = f"density_ds{station}"
        else:
            continue

        coord = hit["x"] if isVertical else hit["y"]
        plane_hits[key].append(coord)

    # Compute density sum for each plane
    for key, coords in plane_hits.items():
        N = len(coords)
        if N < 2:
            branch_vars[key][0] = 0
            continue

        density_sum = 0
        for i, xi in enumerate(coords):
            wi = sum(
                1 for j, xj in enumerate(coords)
                if i != j and abs(xj - xi) < 1.0  # ±1 cm window
            )
            density_sum += wi

        branch_vars[key][0] = density_sum
    
    
    # calculate the below, exlude value == -999 in them
    # density_veto = density_veto1+density_veto2+density_veto3
    # density_scifi1 ...
    # density_us ...
    # density_ds ...
    # desity_total ...
    sum_valid_densities(
        branch_vars,
        ["density_veto1", "density_veto2", "density_veto3"],
        "density_veto"
    )
    sum_valid_densities(
        branch_vars,
        [f"density_scifi{i}" for i in range(1, 6)],
        "density_scifi"
    )
    sum_valid_densities(
        branch_vars,
        [f"density_us{i}" for i in range(1, 6)],
        "density_us"
    )
    sum_valid_densities(
        branch_vars,
        [f"density_ds{i}" for i in range(1, 5)],
        "density_ds"
    )

    # Total density
    sum_valid_densities(
        branch_vars,
        ["density_veto", "density_scifi", "density_us", "density_ds"],
        "density_total"
    )
    

    
    
def process_showerTagged(all_hits, branch_vars, window_cm=3.3, threshold=36):
    #A sliding window of length d (33mm) checks for at least H (set to 36) hits within one SciFi station (X and Y).
    #The most upstream station satisfying this requirement marks the start of the shower
    # save the result to branch

    # Group hits by station and orientation
    scifi_hits = defaultdict(lambda: {"x": [], "y": [], "z": []})

    for hit in all_hits:
        if hit["detType"] != 0:  # Only SciFi
            continue
        station = hit["station"]
        if 1 <= station <= 5:
            if hit["isVertical"]:
                scifi_hits[station]["x"].append(hit["x"])
            else:
                scifi_hits[station]["y"].append(hit["y"])
            scifi_hits[station]["z"].append(hit["z"])

    # Sliding window check
    showerTagged = 0
    showerStartStation = -1

    for station in sorted(scifi_hits.keys()):
        for orientation in ["x", "y"]:
            positions = sorted(scifi_hits[station][orientation])
            N = len(positions)
            if N < threshold:
                continue

            i = 0
            j = 0
            while i < N:
                while j < N and positions[j] - positions[i] < window_cm:
                    j += 1
                if (j - i) >= threshold:
                    showerTagged = 1
                    showerStartStation = station
                    break
                i += 1

            if showerTagged:
                break
        if showerTagged:
            break

    
    branch_vars["showerTagged"][0] = showerTagged
    branch_vars["showerStartStation"][0] = showerStartStation

    # calculate the avg z position of hits in showerStartStation
    if showerTagged == 1:
        avgShowerZ = sum(scifi_hits[showerStartStation]['z'])/len(scifi_hits[showerStartStation]['z'])
        branch_vars["showerStart_z"][0] = avgShowerZ
        
        branch_vars["showerStart_centroid_x"][0] = branch_vars[f"centroid_scifi{showerStartStation}_x"][0]
        branch_vars["showerStart_centroid_y"][0] = branch_vars[f"centroid_scifi{showerStartStation}_y"][0]
        branch_vars["showerStart_avg_x"][0] = branch_vars[f"avg_scifi{showerStartStation}_x"][0]
        branch_vars["showerStart_avg_y"][0] = branch_vars[f"avg_scifi{showerStartStation}_y"][0]
        
        
    hitStartStation = -1
    for station in range(1, 6):
        if station in scifi_hits:
            total_hits = len(scifi_hits[station]["x"]) + len(scifi_hits[station]["y"])
            if total_hits >= 2:
                hitStartStation = station
                break
    
    branch_vars["hitStartStation"][0] = hitStartStation
    avgHitZ = sum(scifi_hits[hitStartStation]['z'])/len(scifi_hits[hitStartStation]['z'])
    branch_vars["hitStart_z"][0] = avgHitZ
    
    branch_vars["hitStart_centroid_x"][0] = branch_vars[f"centroid_scifi{hitStartStation}_x"][0]
    branch_vars["hitStart_centroid_y"][0] = branch_vars[f"centroid_scifi{hitStartStation}_y"][0]
    branch_vars["hitStart_avg_x"][0] = branch_vars[f"avg_scifi{hitStartStation}_x"][0]
    branch_vars["hitStart_avg_y"][0] = branch_vars[f"avg_scifi{hitStartStation}_y"][0]
    
    


def process_slope(all_hits, branch_vars):
    #cal avgPos_slope_{x,y}, and centroid_slope{x,y}, only cal SciFi planes
    #   get the start plane,
    #   calculate the slope with the starting plane and the following planes
    
    # Check if event is shower tagged
    if branch_vars["showerTagged"][0] != 1:
        branch_vars["avgPos_slope_x"][0] = -999
        branch_vars["avgPos_slope_y"][0] = -999
        branch_vars["centroid_slope_x"][0] = -999
        branch_vars["centroid_slope_y"][0] = -999
        return

    # Collect (z, value) pairs for slope computation
    avg_x_points = []
    avg_y_points = []
    centroid_x_points = []
    centroid_y_points = []
    
    start_station = branch_vars["showerStartStation"][0]

    for station in range(start_station, 6):  # 1 to 5 inclusive
        # Find avg_x, avg_y
        avg_x = branch_vars.get(f"avg_scifi{station}_x", [None])[0]
        avg_y = branch_vars.get(f"avg_scifi{station}_y", [None])[0]
        cx = branch_vars.get(f"centroid_scifi{station}_x", [None])[0]
        cy = branch_vars.get(f"centroid_scifi{station}_y", [None])[0]

        # Find corresponding z coordinate from hits
        z_vals = [hit["z"] for hit in all_hits if hit["detType"] == 0 and hit["station"] == station]
        if not z_vals:
            continue
        z = sum(z_vals) / len(z_vals)
        #print(f"start station {station}, z position: {z}")
        if avg_x is not None and avg_x > -998:
            avg_x_points.append((z, avg_x))
        if avg_y is not None and avg_y > -998:
            avg_y_points.append((z, avg_y))
        if cx is not None and cx > -998:
            centroid_x_points.append((z, cx))
        if cy is not None and cy > -998:
            centroid_y_points.append((z, cy))

    def compute_slope(points):
        if len(points) < 2:
            return -999
        z0, v0 = points[0]
        for z1, v1 in points[1:]:
            dz = z1 - z0
            if abs(dz) > 1e-5:
                return (v1 - v0) / dz
        return -999

    branch_vars["avgPos_slope_x"][0] = compute_slope(avg_x_points)
    branch_vars["avgPos_slope_y"][0] = compute_slope(avg_y_points)
    branch_vars["centroid_slope_x"][0] = compute_slope(centroid_x_points)
    branch_vars["centroid_slope_y"][0] = compute_slope(centroid_y_points)
    
    

def print_hits_summary(all_hits):
    summary = defaultdict(list)

    for hit in all_hits:
        detType = hit["detType"]
        station = hit["station"]
        isVertical = hit["isVertical"]
        qdc = hit["qdc"]

        # Grouping key: (detType, station, orientation)
        key = (detType, station, "V" if isVertical else "H")
        summary[key].append(qdc)

    detType_names = {
        0: "SciFi",
        1: "Veto",
        2: "Upstream",
        3: "Downstream"
    }

    print("\n--- Event Hit Summary ---")
    for (detType, station, orientation), qdcs in sorted(summary.items()):
        name = detType_names.get(detType, f"Unknown({detType})")
        count = len(qdcs)
        avg_qdc = sum(qdcs) / count if count > 0 else 0
        print(f"{name:<10} Station {station:<2} {orientation} - Hits: {count:3} | Avg QDC: {avg_qdc:.1f}")
    print("--------------------------\n")
    

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

    

def plot_clustering(all_hits, det_layout, event_ID, pdg, run_ID, args):
    group_color = {
        "wall":      "tab:gray",
        "scifi":       "tab:blue",
        # "veto_bar":  "tab:red",
        # "us_bar":    "tab:orange",
        # "ds_bar":    "tab:purple",
        # "block":     "tab:green",
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
        # if group == 'mat':
        #     continue
        color = group_color.get(group, "k")
        for it in items:
            x, y, z = it["x"], it["y"], it["z"]
            dx, dy, dz = float(it["dx"]), float(it["dy"]), float(it["dz"])
            ver, hor   = int(it["ver"]), int(it["hor"])
            fill = (group != 'scifi')

            if ver == 1:
                add_rect(ax_xz_qdc, z, x, dz, dx, color, fill=fill, alpha=0.4 if fill else 0.8)
                add_rect(ax_xz_shw, z, x, dz, dx, color, fill=fill, alpha=0.4 if fill else 0.8)
            if hor == 1:
                add_rect(ax_yz_qdc, z, y, dz, dy, color, fill=fill, alpha=0.4 if fill else 0.8)
                add_rect(ax_yz_shw, z, y, dz, dy, color, fill=fill, alpha=0.4 if fill else 0.8)

    # ---- bar dimensions per orientation (veto/us/ds) ------------------------

    # dim_lookup = {}
    # for group in ["veto_bar", "us_bar", "ds_bar"]:
    #     dim_lookup[group] = {"ver": None, "hor": None}
    #     for it in det_layout.get(group, []):
    #         if it["ver"] == 1 and dim_lookup[group]["ver"] is None:
    #             dim_lookup[group]["ver"] = (it["dx"], it["dy"], it["dz"])
    #         if it["hor"] == 1 and dim_lookup[group]["hor"] is None:
    #             dim_lookup[group]["hor"] = (it["dx"], it["dy"], it["dz"])

    # ---- QDC colormaps & scales (SciFi vs bars) -----------------------------

    scifi_qdcs = [h["qdc"] for h in all_hits if h["detType"] == 0 and h["qdc"] > 0]
    # bar_qdcs   = [h["qdc"] for h in all_hits if h["detType"] in [1,2,3] and h["qdc"] > 0]
    sci_min, sci_max = (min(scifi_qdcs), max(scifi_qdcs)) if scifi_qdcs else (0, 1)
    # bar_min, bar_max = (min(bar_qdcs),   max(bar_qdcs))   if bar_qdcs   else (0, 1)
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

    run_id = run_ID
    evt_id = event_ID
    pdg    = pdg # if available in your tree
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
    is_nc_event = str(event_level_pdg) in {"112", "114", "116"} or str(event_level_pdg).startswith("11") and len(str(event_level_pdg)) == 3

    if apdg in (11, 13, 15):
        return int(math.copysign(apdg, pdg))  # ±11/±13/±15
    if apdg in (12, 14, 16):
        return int(math.copysign(100 + apdg, pdg)) if is_nc_event else int(math.copysign(apdg, pdg))
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


def process_hits(event, snd_geo, event_ID, pdg, run_ID, det_layout, args):
    """Process all hits in the event and update hits array and averages."""
    MC = args.type
    Scifi = snd_geo.modules['Scifi']
    # MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()
    
    eventId = event_ID
    pdgCode = pdg
    
    #read plane positions, vertical->top_x->A.x, horizontal->right_y->A.y


    # pdgCode, ve:12/-12, vm:14/-14, vt:16/-16, NC:112/-112/114/-114/116/-116
    # need a dictionary: trackId -> showerId (e:11/-11, muon:13/-13, tau:15/-15, NC:12/-12/13/-13/15/-15, hadron: 0)
    # 1.ind all the track which motherId=0
    # 2. assign showerId to these track
    # 3. assign the same showerId to the sub-track of these track
    
    # --- Build hierarchy and shower mapping ---
    m2d = build_mother_to_daughters(event)

    # Print the full tree starting from primaries (mother == -1)
    for root_idx in m2d.get(-1, []):
        # build shower map once (outside the loop is fine too)
        pass

    shower_map = _assign_shower_ids(event, m2d, pdgCode)

    # Now actually print the tree(s)
    primary_tracks = m2d.get(-1, [])
    # for idx in primary_tracks:
    #     print_track_tree(event, m2d, idx, shower_map, indent=0)
    
    #print(shower_map)

    # Temporary storage for all hits with positions
    all_hits = []
    # SciFi hits
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        station = detID // 1000000
        if station not in range(1, 5):
            continue 

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
            n_scifiPoint = event.ScifiPoint.GetEntries()
            shower_id_list = []
            for mc_point_i, weight in linksToMCPoints:  
                if mc_point_i >= n_scifiPoint: #to prevent segmentation fault
                    continue 
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
    
    
    # process_counts(all_hits, branch_vars)
    # process_avgPos(all_hits, branch_vars)
    # process_centroid(all_hits, branch_vars)
    # process_hit_density(all_hits, branch_vars)
    # process_showerTagged(all_hits, branch_vars)
    # process_slope(all_hits, branch_vars)
    plot_clustering(all_hits, det_layout, event_ID, pdg, run_ID, args)
    
    #print_hits_summary(all_hits)
    return 


def plot_detector(det_layout):
    # for group, vols in det_layout.items():
    #     print(f"\n=== {group.upper()} (n={len(vols)}) ===")
    #     for v in vols:  
    #         print(f"{v['name']:25s} @ ({v['x']:.2f}, {v['y']:.2f}, {v['z']:.2f}) "
    #             f"dx={v['dx']:.2f}, dy={v['dy']:.2f}, dz={v['dz']:.2f} "
    #             f"ver={v['ver']}, hor={v['hor']}")
    # --- color map per group ---
    group_color = {
        "wall":      "tab:gray",
        "mat":       "tab:blue",
        "veto_bar":  "tab:red",
        "us_bar":    "tab:orange",
        "ds_bar":    "tab:purple",
        "block":     "tab:green",
    }

    fig, (ax_xz, ax_yz) = plt.subplots(2, 1, figsize=(15, 12), facecolor="white")

    # --- XZ view ---
    ax_xz.set_title("XZ View")
    ax_xz.set_xlabel("Z [cm]")
    ax_xz.set_ylabel("X [cm]")
    ax_xz.set_aspect("equal")
    ax_xz.set_facecolor("white")
    ax_xz.tick_params(direction="in")
    ax_xz.spines[:].set_visible(True)
    ax_xz.grid(False)

    # --- YZ view ---
    ax_yz.set_title("YZ View")
    ax_yz.set_xlabel("Z [cm]")
    ax_yz.set_ylabel("Y [cm]")
    ax_yz.set_aspect("equal")
    ax_yz.set_facecolor("white")
    ax_yz.tick_params(direction="in")
    ax_yz.spines[:].set_visible(True)
    ax_yz.grid(False)

    def add_rect(ax, z, c, half_w, half_h, color, fill=False, lw=0.6, alpha=0.9):
        ax.add_patch(Rectangle(
            (z - half_w, c - half_h),
            2 * half_w, 2 * half_h,
            fill=fill,
            linewidth=lw,
            edgecolor=color,
            facecolor=color if fill else "none",
            alpha=alpha,
        ))

    # --- draw volumes ---
    for group, items in det_layout.items():
        color = group_color.get(group, "k")
        for it in items:
            x, y, z = it["x"], it["y"], it["z"]
            dx, dy, dz = float(it["dx"]), float(it["dy"]), float(it["dz"])
            ver, hor = int(it["ver"]), int(it["hor"])

            # determine if we fill the box (only for wall and block)
            fill = group in {"wall", "block"}

            if ver == 1:
                add_rect(ax_xz, z, x, dz, dx, color, fill=fill,  alpha=0.6 if fill else 0.8)
            if hor == 1:
                add_rect(ax_yz, z, y, dz, dy, color, fill=fill,  alpha=0.6 if fill else 0.8)

    # autoscale axes to data
    for ax in (ax_xz, ax_yz):
        ax.relim()
        ax.autoscale_view()

    os.makedirs("./plots", exist_ok=True)
    output_path = "./plots/detector_layout.pdf"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved detector layout to {output_path}")
    
    
import ROOT, re
from collections import defaultdict


def get_detector_layout(geo_file):
    """
    Parcourt le geofile ROOT et extrait les positions globales
    des plans SciFi et des murs (quel que soit leur nom).
    """

    # --- Regex patterns pour identifier les volumes d'intérêt ---
    SCIFI_RE = re.compile(r"ScifiVolume\d+", re.IGNORECASE)
    WALL_RE  = re.compile(r"^(Wall(_\d+)?|volWallborder(_\d+)?|volFeTarget\d+)$", re.IGNORECASE)

    IS_TARGET   = lambda path: any(n.startswith("volTarget") for n in path)

    def get_global_xyz(mat):
        """Retourne la position (x, y, z) globale d'une matrice TGeo."""
        t = mat.GetTranslation()
        return float(t[0]), float(t[1]), float(t[2])

    # --- Ouverture du fichier ROOT ---
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
    stack = [(top, ROOT.TGeoHMatrix(), [top.GetVolume().GetName()])]

    # --- Parcours récursif de la hiérarchie géométrique ---
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
            M_child = ROOT.TGeoHMatrix(M)
            M_child.Multiply(ch.GetMatrix())
            path_child = path + [ch_name]

            x, y, z = get_global_xyz(M_child)
            shp = ch_vol.GetShape()
            dx = getattr(shp, "GetDX", lambda: 0)()
            dy = getattr(shp, "GetDY", lambda: 0)()
            dz = getattr(shp, "GetDZ", lambda: 0)()

            ver, hor = 0, 0

            if WALL_RE.match(ch_name):
                ver, hor = 1, 1
            else:
                if "ver" in ch_name.lower() or any("ver" in p.lower() for p in path_child):
                    ver, hor = 1, 0
                elif "hor" in ch_name.lower() or any("hor" in p.lower() for p in path_child):
                    ver, hor = 0, 1
            # Identifier SciFi ou Wall
            if SCIFI_RE.match(ch_name) and IS_TARGET(path_child):
                out["scifi"].append(dict(
                    name=ch_name, x=x, y=y, z=z,
                    dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
                ))

            elif WALL_RE.match(ch_name) and IS_TARGET(path_child):
                out["wall"].append(dict(
                    name=ch_name, x=x, y=y, z=z,
                    dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
                ))

            # Poursuivre la descente récursive
            stack.append((ch, M_child, path_child))

    f.Close()
    return dict(out)


# def get_detector_layout(geo_file):
#     """
#     Extract detector component geometry from an SND@LHC/FairRoot geofile.

#     Returns
#     -------
#     dict[str, list[dict]]:
#         Keys: wall, mat, veto_bar, us_bar, ds_bar, block
#         Each entry has: name, x, y, z, dx, dy, dz, ver, hor, path
#     """

#     # ---- Regex patterns for relevant subdetectors ----
#     WALL_RE      = re.compile(r"^volWallborder(_\d+)?$", re.IGNORECASE)
#     MAT_RE       = re.compile(r"^(Hor|Vert)?MatVolume(_\d+)?$", re.IGNORECASE)
#     # VETO_BAR_RE  = re.compile(r"^volVetoBar(_ver)?(_\d+)?$", re.IGNORECASE)
#     # US_BAR_RE    = re.compile(r"^volMuUpstreamBar(_(hor|ver))?(_\d+)?$", re.IGNORECASE)
#     # DS_BAR_RE    = re.compile(r"^volMuDownstreamBar(_(hor|ver))?(_\d+)?$", re.IGNORECASE)
#     # BLOCK_RE     = re.compile(r"^volFeBlock(_\d+)?$", re.IGNORECASE)

#     # ---- Ancestor name hints ----
#     IS_TARGET   = lambda path: any(n.startswith("volTarget") for n in path)
#     # IS_VETO     = lambda path: any(n.startswith("volVeto") for n in path)
#     # IS_VETO_PL  = lambda name: name.startswith("volVetoPlane")
#     # IS_MUFILT   = lambda path: any(n.startswith("volMuFilter") for n in path)
#     # IS_US_DET   = lambda name: name.startswith("volMuUpstreamDet")
#     # IS_DS_DET   = lambda name: name.startswith("volMuDownstreamDet")

#     def get_global_xyz(mat):
#         t = mat.GetTranslation()
#         return float(t[0]), float(t[1]), float(t[2])

#     # ---- open geometry ----
#     f = ROOT.TFile.Open(geo_file)
#     if not f or f.IsZombie():
#         raise RuntimeError(f"Could not open geometry file: {geo_file}")
#     geom = f.Get("FAIRGeom")
#     if not geom:
#         raise RuntimeError("TGeoManager 'FAIRGeom' not found in file")
#     top = geom.GetTopNode()
#     if not top:
#         raise RuntimeError("No top node in geometry")

#     out = defaultdict(list)

#     # ---- DFS traversal ----
#     stack = [(top, ROOT.TGeoHMatrix(), [top.GetVolume().GetName()])]
#     while stack:
#         node, M, path = stack.pop()
#         vol = node.GetVolume()
#         if not vol:
#             continue

#         for i in range(node.GetNdaughters()):
#             ch = node.GetDaughter(i)
#             ch_vol = ch.GetVolume()
#             if not ch_vol:
#                 continue
#             ch_name = ch_vol.GetName()

#             # compose global transform
#             M_child = ROOT.TGeoHMatrix(M)
#             M_child.Multiply(ch.GetMatrix())
#             path_child = path + [ch_name]
#             x, y, z = get_global_xyz(M_child)

#             # get bounding box dimensions
#             shp = ch_vol.GetShape()
#             dx = getattr(shp, "GetDX", lambda: 0)()
#             dy = getattr(shp, "GetDY", lambda: 0)()
#             dz = getattr(shp, "GetDZ", lambda: 0)()

#             # --- Orientation logic ---
#             ver, hor = 0, 0

#             if WALL_RE.match(ch_name):
#                 ver, hor = 1, 1
#             # elif US_BAR_RE.match(ch_name):
#             #     ver, hor = 0, 1
#             # elif VETO_BAR_RE.match(ch_name):
#             #     if "ver" in ch_name.lower():
#             #         ver, hor = 1, 0
#             #     else:
#             #         ver, hor = 0, 1
#             # elif BLOCK_RE.match(ch_name):
#             #     ver, hor = 1, 1
#             else:
#                 if "ver" in ch_name.lower() or any("ver" in p.lower() for p in path_child):
#                     ver, hor = 1, 0
#                 elif "hor" in ch_name.lower() or any("hor" in p.lower() for p in path_child):
#                     ver, hor = 0, 1

#             # ---- Classification ----
#             # 1) Wallborder under Target
#             if WALL_RE.match(ch_name) and IS_TARGET(path_child):
#                 out["wall"].append(dict(
#                     name=ch_name, x=x, y=y, z=z,
#                     dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
#                 ))

#             # 2) SciFi material layers under Target
#             if MAT_RE.match(ch_name) and IS_TARGET(path_child):
#                 out["mat"].append(dict(
#                     name=ch_name, x=x, y=y, z=z,
#                     dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
#                 ))

#             # # 3) Veto bars
#             # if VETO_BAR_RE.match(ch_name) and IS_VETO(path_child):
#             #     if any(IS_VETO_PL(n) for n in path_child):
#             #         out["veto_bar"].append(dict(
#             #             name=ch_name, x=x, y=y, z=z,
#             #             dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
#             #         ))

#             # # 4) MuFilter upstream bars
#             # if US_BAR_RE.match(ch_name) and IS_MUFILT(path_child):
#             #     if any(IS_US_DET(n) for n in path_child):
#             #         out["us_bar"].append(dict(
#             #             name=ch_name, x=x, y=y, z=z,
#             #             dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
#             #         ))

#             # # 5) MuFilter downstream bars
#             # if DS_BAR_RE.match(ch_name) and IS_MUFILT(path_child):
#             #     if any(IS_DS_DET(n) for n in path_child):
#             #         out["ds_bar"].append(dict(
#             #             name=ch_name, x=x, y=y, z=z,
#             #             dx=dx, dy=dy, dz=dz, ver=ver, hor=hor, path=tuple(path_child)
#             #         ))

#             # # 6) MuFilter Fe blocks
#             # if BLOCK_RE.match(ch_name) and IS_MUFILT(path_child):
#             #     out["block"].append(dict(
#             #         name=ch_name, x=x, y=y, z=z,
#             #         dx=dx, dy=dy, dz=dz, ver=1, hor=1, path=tuple(path_child)
#             #     ))

#             # continue recursion
#             stack.append((ch, M_child, path_child))

#     return dict(out)

def main(args):
    print("start processing digi to features")
    
    print("getting geo layout")
    det_layout = get_detector_layout(args.geo_path)    
    #plot_detector(det_layout)


    print("setting snd geo interface")
    snd_geo = setup_geometry(args.geo_path)
    raw_data, raw_tree = open_root_file(args.digi_path)
   
    pdg = name_to_pdg.get(args.pdg, [])
    
    for i in range(raw_tree.GetEntries()):
        if i % 10000 == 0:
            print(f"processed {i} events")
            
        raw_tree.GetEntry(i)
            
        run_ID = raw_tree.EventHeader.GetRunId()
        
        if ('MC' in  args.type):
            try:
                event_ID = raw_tree.EventHeader.GetEventNumber()
            except Exception:
                event_ID = raw_tree.EventHeader.GetMCEntryNumber()
        elif('real' in  args.type):
            event_ID = raw_tree.EventHeader.GetEventNumber()
    
        process_hits(raw_tree, snd_geo, event_ID, pdg, run_ID, det_layout, args)
        # if i>2:
        #   break


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("-p", "--preSelectPath", dest="preSelect_path", help="pre selection data file path", required=True)
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)
    parser.add_argument("-pdg", "--pdg", dest='pdg', help='type of particle', required=True)


    args = parser.parse_args()

    main(args)
    
# python digi_2_features_shower.py -p /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/1/preSelect_MC_neutrino_volTarget_100fb-1_1.root -d /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/1/sndLHC.Genie-TGeant4_20240126_digCPP.root -g /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/1/geofile_full.Genie-TGeant4.root -o ./test_data/vetoTagged_shower_feature.root -t MC_neutrino