import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import array
from collections import defaultdict
import math
from tqdm import tqdm
import bisect
import pandas as pd


def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo

def open_root_file(file_path, tree_name='cbmsim', mode='read'):
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
            if 1 <= station <= 4:
                branch_vars[f"count_scifi{station}"][0] += 1
            branch_vars["count_scifi"][0] += 1

def process_avgPos(all_hits, branch_vars):
    """
    Computes:
      - Per-station averages: avg_scifi{1-4}_{x,y}
          vertical planes -> x, horizontal planes -> y
      - Global averages: avg_scifi_x, avg_scifi_y
          vertical planes -> x, horizontal planes -> y
    """

    # Per-station accumulators
    sums = defaultdict(float)
    counts = defaultdict(int)

    # Global accumulators
    sum_x = 0.0
    sum_y = 0.0
    cnt_x = 0
    cnt_y = 0

    for hit in all_hits:
        if hit["detType"] != 0:  # SciFi only
            continue

        station = hit["station"]
        isVertical = hit["isVertical"]

        # Global averages (all stations)
        if isVertical:
            sum_x += hit["x"]
            cnt_x += 1
        else:
            sum_y += hit["y"]
            cnt_y += 1

        # Per-station averages (stations 1-4, as in your original code)
        if 1 <= station <= 4:
            if isVertical:
                key = f"avg_scifi{station}_x"
                sums[key] += hit["x"]
                counts[key] += 1
            else:
                key = f"avg_scifi{station}_y"
                sums[key] += hit["y"]
                counts[key] += 1

    DEFAULT = -999

    # ---- Fill per-station branches (computed ones) ----
    for key, total in sums.items():
        if key in branch_vars:
            branch_vars[key][0] = total / counts[key] if counts[key] > 0 else DEFAULT

    # ---- Fill missing per-station branches with DEFAULT ----
    for key in [
        "avg_scifi1_x", "avg_scifi1_y", "avg_scifi2_x", "avg_scifi2_y",
        "avg_scifi3_x", "avg_scifi3_y", "avg_scifi4_x", "avg_scifi4_y",
    ]:
        if key in branch_vars and key not in counts:
            branch_vars[key][0] = DEFAULT

    # ---- Fill global branches ----
    if "avg_scifi_x" in branch_vars:
        branch_vars["avg_scifi_x"][0] = (sum_x / cnt_x) if cnt_x > 0 else DEFAULT

    if "avg_scifi_y" in branch_vars:
        branch_vars["avg_scifi_y"][0] = (sum_y / cnt_y) if cnt_y > 0 else DEFAULT


def process_centroid(all_hits, branch_vars):
    """
    Computes:
      - Per-station QDC-weighted centroids:
          centroid_scifi{1-4}_{x,y}
      - Global QDC-weighted centroids:
          centroid_scifi_x, centroid_scifi_y

    Convention:
      - Vertical planes   -> x
      - Horizontal planes -> y
    """

    # Per-station accumulators
    weighted_sums = defaultdict(float)
    total_qdc = defaultdict(float)

    # Global accumulators
    global_weighted_x = 0.0
    global_weighted_y = 0.0
    global_qdc_x = 0.0
    global_qdc_y = 0.0

    for hit in all_hits:
        if hit["detType"] != 0:  # SciFi only
            continue

        qdc = hit.get("qdc", 0)
        if qdc <= 0:
            continue

        station = hit["station"]
        isVertical = hit["isVertical"]

        # ---- Global centroids ----
        if isVertical:
            global_weighted_x += qdc * hit["x"]
            global_qdc_x += qdc
        else:
            global_weighted_y += qdc * hit["y"]
            global_qdc_y += qdc

        # ---- Per-station centroids (1–4, same as original) ----
        if 1 <= station <= 4:
            if isVertical:
                key = f"centroid_scifi{station}_x"
                weighted_sums[key] += qdc * hit["x"]
                total_qdc[key] += qdc
            else:
                key = f"centroid_scifi{station}_y"
                weighted_sums[key] += qdc * hit["y"]
                total_qdc[key] += qdc

    DEFAULT = -999

    # ---- Per-station finalization ----
    per_station_keys = [
        "centroid_scifi1_x", "centroid_scifi1_y",
        "centroid_scifi2_x", "centroid_scifi2_y",
        "centroid_scifi3_x", "centroid_scifi3_y",
        "centroid_scifi4_x", "centroid_scifi4_y",
    ]

    for key in per_station_keys:
        if key in branch_vars and total_qdc[key] > 0:
            branch_vars[key][0] = weighted_sums[key] / total_qdc[key]
        elif key in branch_vars:
            branch_vars[key][0] = DEFAULT

    # ---- Global finalization ----
    if "centroid_scifi_x" in branch_vars:
        branch_vars["centroid_scifi_x"][0] = (
            global_weighted_x / global_qdc_x if global_qdc_x > 0 else DEFAULT
        )

    if "centroid_scifi_y" in branch_vars:
        branch_vars["centroid_scifi_y"][0] = (
            global_weighted_y / global_qdc_y if global_qdc_y > 0 else DEFAULT
        )

def process_hit_density(all_hits, branch_vars):
    plane_hits = defaultdict(list)

    # Group hits by plane
    for hit in all_hits:
        detType = hit["detType"]
        station = hit["station"]
        isVertical = hit["isVertical"]

        if detType == 0 and 1 <= station <= 4:
            key = f"density_scifi{station}"
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
    
    sum_valid_densities(
        branch_vars,
        [f"density_scifi{i}" for i in range(1, 5)],
        "density_scifi"
    )
    
    
def sum_valid_densities(branch_vars, group_keys, target_key):
    total = 0
    for key in group_keys:
        value = branch_vars[key][0]
        if value >=0:
            total += value
    branch_vars[target_key][0] = total
         
    
def process_showerTagged(all_hits, branch_vars, window_cm=3.3, threshold=36):
    #A sliding window of length d (33mm) checks for at least H (set to 36) hits within one SciFi station (X and Y).
    #The most upstream station satisfying this requirement marks the start of the shower
    # save the result to branch

    # Group hits by station and orientation
    scifi_hits = defaultdict(lambda: {"x": [], "y": []})

    for hit in all_hits:
        if hit["detType"] != 0:  # Only SciFi
            continue
        station = hit["station"]
        if 1 <= station <= 4:
            if hit["isVertical"]:
                scifi_hits[station]["x"].append(hit["x"])
            else:
                scifi_hits[station]["y"].append(hit["y"])

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

    for station in range(start_station, 5):  # 1 to 4 inclusive
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

def process_qdc(all_hits, branch_vars):
    """
    Process QDC values by detector type and station.
    """
    # Reset all QDC sums to 0
    for key in [
        "qdc_scifi1", "qdc_scifi2", "qdc_scifi3", "qdc_scifi4","qdc_scifi",
    ]:
        branch_vars[key][0] = 0
    for hit in all_hits:
        detType = hit["detType"]
        station = hit["station"]
        qdc_value = hit.get("qdc")  # Use get with default 0.0
        if qdc_value<0:
            continue
        
        if detType == 0:  # SciFi
            if 1 <= station <= 4:
                branch_vars[f"qdc_scifi{station}"][0] += qdc_value
            branch_vars["qdc_scifi"][0] += qdc_value


def density_sum_1d_fast(coords, dx=1.0):
    n = len(coords)
    if n < 2:
        return 0
    xs = sorted(coords)
    density_sum = 0
    left = 0
    right = 0
    for i in range(n):
        xi = xs[i]
        while xi - xs[left] >= dx:
            left += 1
        if right < i + 1:
            right = i + 1
        while right < n and xs[right] - xi < dx:
            right += 1
        density_sum += (right - left - 1)
    return density_sum


def process_qdc_with_clockCycle(all_hits, branch_vars, nbins=100,
                                clock_cycle_windows=(4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08),
                                det_type_scifi=0,
                                density_dx=1.0):
    # Collect SciFi hits
    scifi_hits = [h for h in all_hits
                  if h.get("detType") == det_type_scifi and "hitTime_clockCycle" in h]
    if not scifi_hits:
        return

    cc_all = [h["hitTime_clockCycle"] for h in scifi_hits]

    # --- MPV via histogram mode (same as your original) ---
    cmin, cmax = min(cc_all), max(cc_all)
    if cmin == cmax:
        mpv = cmin
    else:
        pad = 0.05 * (cmax - cmin)
        htmp = ROOT.TH1F("h_cc_tmp_qdc", "", nbins, cmin - pad, cmax + pad)
        for x in cc_all:
            htmp.Fill(x)
        mpv = htmp.GetBinCenter(htmp.GetMaximumBin())

    # --- Sort once by clock cycle for fast window slicing ---
    scifi_hits.sort(key=lambda h: h["hitTime_clockCycle"])
    cc_sorted = [h["hitTime_clockCycle"] for h in scifi_hits]

    for cy in clock_cycle_windows:
        lo, hi = mpv - cy, mpv + cy

        # Find hits in [lo, hi] using bisect (O(log N))
        i0 = bisect.bisect_left(cc_sorted, lo)
        i1 = bisect.bisect_right(cc_sorted, hi)
        hits_in_window = scifi_hits[i0:i1]
        N = len(hits_in_window)

        
        sum_qdc = 0.0
        # Group coordinates by station (or station+something)
        coords_by_group = defaultdict(list)

        for hit in hits_in_window:
            q = hit.get("qdc", 0.0)
            if q is not None and math.isfinite(q):
                sum_qdc += float(q)

            isVertical = hit.get("isVertical", False)
            coord = hit.get("x", 0.0) if isVertical else hit.get("y", 0.0)

            station = hit.get("station", None)   # <-- MUST exist in hit dict (adapt key name)
            # Optional: also split by detector plane/layer if needed:
            # layer = hit.get("layer", None)
            # group_key = (station, layer, isVertical)

            group_key = (station, isVertical)    # minimal safe version
            coords_by_group[group_key].append(coord)

        density_sum = 0
        for group_key, coords in coords_by_group.items():
            density_sum += density_sum_1d_fast(coords, dx=density_dx)

        # Store
        kq = f"qdc_scifi_{cy}cy"
        if kq in branch_vars:
            branch_vars[kq][0] = sum_qdc

        kc = f"count_scifi_{cy}cy"
        if kc in branch_vars:
            branch_vars[kc][0] = N

        kd = f"density_scifi_{cy}cy"
        if kd in branch_vars:
            branch_vars[kd][0] = density_sum


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
        0: "SciFi"
    }

    print("\n--- Event Hit Summary ---")
    for (detType, station, orientation), qdcs in sorted(summary.items()):
        name = detType_names.get(detType, f"Unknown({detType})")
        count = len(qdcs)
        avg_qdc = sum(qdcs) / count if count > 0 else 0
        print(f"{name:<10} Station {station:<2} {orientation} - Hits: {count:3} | Avg QDC: {avg_qdc:.1f}")
    print("--------------------------\n")





def process_hitTime_and_QDC(all_hits, branch_vars, nbins=100):
    """
    - Find MPV of hitTime_clockCycle (ROOT.TH1F mode)
    - Keep hits within [MPV-0.5, MPV+2.3] (filters all_hits in place)
    - Compute min, max, avg, mpv, std of hitTime_clockCycle and qdc after cut
    - Store in branch_vars:
        qdc_min, qdc_max, qdc_avg, qdc_mpv, qdc_std,
        hitTime_min, hitTime_max, hitTime_avg, hitTime_mpv, hitTime_std
    """

    # --- Collect clock cycles (for MPV) ---
    cc_all = [h["hitTime_clockCycle"] for h in all_hits if "hitTime_clockCycle" in h]
    n_before = len(cc_all)

    # Helper to set outputs to a default
    def _set_defaults():
        for k in ("qdc_min","qdc_max","qdc_avg","qdc_mpv","qdc_std",
                  "hitTime_min","hitTime_max","hitTime_avg","hitTime_mpv","hitTime_std"):
            if k in branch_vars:
                branch_vars[k][0] = -999.0

    if n_before == 0:
        return

    # --- MPV of hitTime_clockCycle via histogram mode (same as your process_hitTime) ---
    cmin, cmax = min(cc_all), max(cc_all)
    if cmin == cmax:
        mpv_cc = cmin
    else:
        pad = 0.05 * (cmax - cmin)
        h = ROOT.TH1F("h_cc_tmp_hitTimeQDC", "", nbins, cmin - pad, cmax + pad)
        for x in cc_all:
            h.Fill(x)
        mpv_cc = h.GetBinCenter(h.GetMaximumBin())

    # --- Apply MPV window cut, filter hits in place ---
    lo, hi = mpv_cc - 0.5, mpv_cc + 2.3
    all_hits[:] = [h for h in all_hits
                   if ("hitTime_clockCycle" in h) and (lo <= h["hitTime_clockCycle"] <= hi)]

    # After cut: collect hitTime and QDC
    cc_sel  = [h["hitTime_clockCycle"] for h in all_hits if "hitTime_clockCycle" in h]
    qdc_sel = [h["qdc"] for h in all_hits if "qdc" in h]

    if len(cc_sel) == 0 or len(qdc_sel) == 0:
        _set_defaults()
        return

    # --- Stats helpers ---
    def _mean_std(vals):
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            var = sum((x - mean)**2 for x in vals) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        return mean, std

    def _mpv_hist(vals, name):
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax:
            return float(vmin)
        pad = 0.05 * (vmax - vmin)
        hh = ROOT.TH1F(name, "", nbins, vmin - pad, vmax + pad)
        for x in vals:
            hh.Fill(x)
        return float(hh.GetBinCenter(hh.GetMaximumBin()))

    # --- hitTime stats (after cut) ---
    hit_min = float(min(cc_sel))
    hit_max = float(max(cc_sel))
    hit_avg, hit_std = _mean_std(cc_sel)
    hit_mpv = float(mpv_cc)  # MPV used for the window (consistent with selection)

    # --- QDC stats (after cut) ---
    qdc_min = float(min(qdc_sel))
    qdc_max = float(max(qdc_sel))
    qdc_avg, qdc_std = _mean_std(qdc_sel)
    qdc_mpv = _mpv_hist(qdc_sel, "h_qdc_tmp_hitTimeQDC")
    #print(f"[QDC] min={qdc_min:.3f} max={qdc_max:.3f} mean={qdc_avg:.3f} std={qdc_std:.3f} mpv={qdc_mpv:.3f}")

    # --- Store to branches ---
    if "hitTime_min" in branch_vars: branch_vars["hitTime_min"][0] = hit_min
    if "hitTime_max" in branch_vars: branch_vars["hitTime_max"][0] = hit_max
    if "hitTime_avg" in branch_vars: branch_vars["hitTime_avg"][0] = float(hit_avg)
    if "hitTime_std" in branch_vars: branch_vars["hitTime_std"][0] = float(hit_std)
    if "hitTime_mpv" in branch_vars: branch_vars["hitTime_mpv"][0] = hit_mpv

    if "qdc_min" in branch_vars: branch_vars["qdc_min"][0] = qdc_min
    if "qdc_max" in branch_vars: branch_vars["qdc_max"][0] = qdc_max
    if "qdc_avg" in branch_vars: branch_vars["qdc_avg"][0] = float(qdc_avg)
    if "qdc_std" in branch_vars: branch_vars["qdc_std"][0] = float(qdc_std)
    if "qdc_mpv" in branch_vars: branch_vars["qdc_mpv"][0] = float(qdc_mpv)
    
  
    
def process_hits(event, snd_geo, branch_vars, offset_df):
    """Process all hits in the event and update hits array and averages."""
    Scifi = snd_geo.modules['Scifi']
    A, B = ROOT.TVector3(), ROOT.TVector3()
    
    
    
    
    #read plane positions, vertical->top_x->A.x, horizontal->right_y->A.y
    
    # Temporary storage for all hits with positions
    all_hits = []

    # SciFi hits
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        hitTime = aHit.GetTime()
        clock_cycle = hitTime/6.25
        
        detID = aHit.GetDetectorID()

        st  = detID // 1000000
        ori = int(aHit.isVertical())           # 0/1
        mat = (detID % 100000) // 10000
        local_ch  = detID % 1000
        tofpet_id = (detID % 10000) // 1000

        ch = tofpet_id*128 + local_ch

        Scifi.GetSiPMPosition(detID, A, B)

        qdc = aHit.GetSignal(0)
        if st == 2 and mat == 0 and ori==0 and ('real' in  args.type):
            mpv_value = offset_df.loc[ch, "mpv_data_avg"]
            qdc = qdc-mpv_value
        if qdc<0:
            qdc = 0
        
        all_hits.append({
            "detType": 0,
            "station": st,
            "isVertical": aHit.isVertical(),
            "x":A.x(),
            "y":A.y(),
            "z":A.z(),
            "qdc":qdc,
            "hitTime_ns": hitTime,
            "hitTime_clockCycle": clock_cycle,
        })


    process_hitTime_and_QDC(all_hits, branch_vars)
    process_showerTagged(all_hits, branch_vars)
    process_counts(all_hits, branch_vars)
    process_avgPos(all_hits, branch_vars)
    process_centroid(all_hits, branch_vars)
    process_hit_density(all_hits, branch_vars)
    process_slope(all_hits, branch_vars)
    process_qdc(all_hits, branch_vars)
    process_qdc_with_clockCycle(all_hits, branch_vars)
    


    #print_hits_summary(all_hits)
    return 


def main(args):
    print("start processing digi to features")
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)
    beam_type = args.pdg
    
    out_file, new_tree = create_output_file(args.out_path, args.mode)

    branches = [
        ("runId", 'i'), ("eventId", 'i'), ("pdgCode", 'i'), ("isMC", 'i'), ("eventIndex", 'i'),("pdgCode1", 'i'), ("eventTime", 'd'), ("previous_event_time_gap", 'd'),
        ("px", 'f'), ("py", 'f'), ("pz", 'f'),  # Floats
        ("x", 'f'), ("y", 'f'), ("z", 'f'),    # Floats
        
        
        ("count_scifi1", 'i'), ("count_scifi2", 'i'), ("count_scifi3", 'i'),("count_scifi4", 'i'), ("count_scifi", 'i'),
        ("count_scifi_0.5cy", 'd'),
        ("count_scifi_0.4cy", 'd'),
        ("count_scifi_0.3cy", 'd'),
        ("count_scifi_0.2cy", 'd'),
        ("count_scifi_0.1cy", 'd'),
        ("count_scifi_4cy", 'd'),
        
        ("qdc_scifi1", 'd'), ("qdc_scifi2", 'd'), ("qdc_scifi3", 'd'), ("qdc_scifi4", 'd'), ("qdc_scifi", 'd'),
        ("qdc_scifi_0.5cy", 'd'),
        ("qdc_scifi_0.4cy", 'd'),
        ("qdc_scifi_0.3cy", 'd'),
        ("qdc_scifi_0.2cy", 'd'),
        ("qdc_scifi_0.1cy", 'd'),
        ("qdc_scifi_4cy", 'd'),
        ("qdc_min", 'd'), ("qdc_max", 'd'), ("qdc_avg", 'd'), ("qdc_mpv", 'd'), ("qdc_std", 'd'), 
        ("hitTime_min", 'd'), ("hitTime_max", 'd'), ("hitTime_avg", 'd'), ("hitTime_mpv", 'd'), ("hitTime_std", 'd'), 
  
        ("avg_scifi_x", 'd'), ("avg_scifi_y", 'd'),
        ("avg_scifi1_x", 'd'), ("avg_scifi1_y", 'd'),
        ("avg_scifi2_x", 'd'), ("avg_scifi2_y", 'd'),
        ("avg_scifi3_x", 'd'), ("avg_scifi3_y", 'd'),
        ("avg_scifi4_x", 'd'), ("avg_scifi4_y", 'd'),
        
        ("centroid_scifi_x", 'd'), ("centroid_scifi_y", 'd'),
        ("centroid_scifi1_x", 'd'), ("centroid_scifi1_y", 'd'),
        ("centroid_scifi2_x", 'd'), ("centroid_scifi2_y", 'd'),
        ("centroid_scifi3_x", 'd'), ("centroid_scifi3_y", 'd'),
        ("centroid_scifi4_x", 'd'), ("centroid_scifi4_y", 'd'),


        # Hit density sums per plane
        ("density_scifi1", 'd'), ("density_scifi2", 'd'), ("density_scifi3", 'd'), ("density_scifi4", 'd'), ("density_scifi", 'd'),
        ("density_scifi_0.5cy", 'd'),
        ("density_scifi_0.4cy", 'd'),
        ("density_scifi_0.3cy", 'd'),
        ("density_scifi_0.2cy", 'd'),
        ("density_scifi_0.1cy", 'd'),
        ("density_scifi_4cy", 'd'),

        ("showerTagged", 'i'),
        ("showerStartStation", 'i'),
        
        ("avgPos_slope_x", 'd'), ("avgPos_slope_y", 'd'),
        ("centroid_slope_x", 'd'), ("centroid_slope_y", 'd'),

        ("start_z", 'd'),
        #energy, 
    ]
    
    offset_df = pd.read_csv("/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/evaluation/QDC_offset/QDC_offset_st2_mat0_ori0.csv")
    offset_df = offset_df.set_index("channel")

    # Dictionary to hold branch variables
    branch_vars = {}

    # Create branches dynamically
    for name, dtype in branches:
        branch_vars[name] = array.array(dtype, [-999])  # Initialize the array
        new_tree.Branch(name, branch_vars[name], f"{name}/{dtype.upper()}")
        
    # Process each event

    n = raw_tree.GetEntries()
    if (n>args.max_event) and ('real' in  args.type):
        n = args.max_event
    for i in tqdm(range(n), total=n, desc="Processing events"):
        if i % 10000 == 0:
            print(f"processed {i} events")

        # Reset all branch variables before filling them
        for key in branch_vars:
            branch_vars[key][0] = -999

        entry_number = i
        raw_tree.GetEntry(entry_number)
        
        branch_vars["eventIndex"][0] = entry_number
        branch_vars["eventTime"][0] = raw_tree.EventHeader.GetEventTime()
        if i==0:
            last_event_time = raw_tree.EventHeader.GetEventTime()
            branch_vars["previous_event_time_gap"][0] = 0
        else:
            branch_vars["previous_event_time_gap"][0] = raw_tree.EventHeader.GetEventTime() - last_event_time
            last_event_time = raw_tree.EventHeader.GetEventTime()

        branch_vars["runId"][0] = raw_tree.EventHeader.GetRunId()
        
        
        if ('MC' in  args.type):
            branch_vars["isMC"][0] = 1
            try:
                branch_vars["eventId"][0] = raw_tree.EventHeader.GetEventNumber()
            except Exception:
                branch_vars["eventId"][0] = raw_tree.EventHeader.GetMCEntryNumber()
                
            event_pdg0 = raw_tree.MCTrack[0].GetPdgCode()
            event_pdg1 = raw_tree.MCTrack[1].GetPdgCode()
            
            branch_vars["pdgCode1"][0] = event_pdg1

            neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
            if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
                branch_vars["pdgCode"][0] = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
            else:
                branch_vars["pdgCode"][0] = event_pdg0
                
            branch_vars["x"][0]= raw_tree.MCTrack[1].GetStartX()
            branch_vars["y"][0]= raw_tree.MCTrack[1].GetStartY()
            branch_vars["z"][0]= raw_tree.MCTrack[1].GetStartZ()

            branch_vars["px"][0] = raw_tree.MCTrack[0].GetPx()
            branch_vars["py"][0] = raw_tree.MCTrack[0].GetPy()
            branch_vars["pz"][0] = raw_tree.MCTrack[0].GetPz()
            
            start_z = (
                raw_tree.MCTrack[1].GetStartZ()
                if raw_tree.MCTrack.GetEntries() > 1
                else -999
                )
            branch_vars["start_z"][0] = start_z
            
        elif('real' in  args.type):
            if beam_type != 'no type':
                if 'pi+' in beam_type:
                    branch_vars["pdgCode"][0] = 211
                elif 'pi-' in beam_type:
                    branch_vars["pdgCode"][0] = -211
                elif 'mu-' in beam_type:
                    branch_vars["pdgCode"][0] = 13
                elif 'e-' in beam_type:
                    branch_vars["pdgCode"][0] = 11
                elif 'e+' in beam_type:
                    branch_vars["pdgCode"][0] = -11
                else:
                    branch_vars["pdgCode"][0] = 0
            else:
                branch_vars["pdgCode"][0] = 0
            branch_vars["isMC"][0] = 0
            branch_vars["eventId"][0] = raw_tree.EventHeader.GetEventNumber()

        process_hits(raw_tree, snd_geo, branch_vars, offset_df)
        #if i>2:
        #    break
        new_tree.Fill()
    # Finalize the output file
    new_tree.Write()
    out_file.Close()
    print("finish processing digi to feature")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default='RECREATE')
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)
    parser.add_argument("-pdg", "--pdg", dest="pdg", help="PDG code", required=False, default='no type')
    parser.add_argument("-m", "--max_event", dest="max_event", help="max processed events", required=False, default=2000)
    args = parser.parse_args()

    main(args)
    
# python digi_2_features.py -p /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/preSelect_MC_neutrino_volTarget_100fb-1_0.root -d /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_20240126_digCPP.root -g /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/geofile_full.Genie-TGeant4.root -o /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/vetoTagged_feature_MC_neutrino_volTarget_100fb-1_0.root -t MC_neutrino