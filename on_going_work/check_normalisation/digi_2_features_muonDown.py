import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import array
import collections
from collections import defaultdict
import math
from tqdm import tqdm
import numpy as np


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

def open_root_file(file_path, tree_name='cbmsim', mode='read'):
    file = ROOT.TFile(file_path, mode)
    tree = file.Get(tree_name)
    if not tree or not isinstance(tree, ROOT.TTree):
        raise RuntimeError(f"TTree '{tree_name}' not found in {file_path}")
    return file, tree


def create_output_file(path, mode):
    """Create and return a new ROOT file and a new tree for output."""
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    out_file = ROOT.TFile(path, mode)
    new_tree = ROOT.TTree('sndData', 'converted SND hits tree')
    new_tree.SetDirectory(out_file)
    return out_file, new_tree

def process_counts(all_hits, branch_vars):
    # Reset counters to 0
    for key in [
        "count_veto1", "count_veto2", "count_veto3", "count_veto",
        "count_scifi1", "count_scifi2", "count_scifi3", "count_scifi4", "count_scifi5", "count_scifi",
        "count_us1", "count_us2", "count_us3", "count_us4", "count_us5", "count_us",
        "count_ds1", "count_ds2", "count_ds3", "count_ds4", "count_ds", 
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
        
        
    # hitStartStation = -1
    # for station in range(1, 6):
    #     if station in scifi_hits:
    #         total_hits = len(scifi_hits[station]["x"]) + len(scifi_hits[station]["y"])
    #         if total_hits >= 1:
    #             hitStartStation = station
    #             break
    
    # branch_vars["hitStartStation"][0] = hitStartStation

    # avgHitZ = sum(scifi_hits[hitStartStation]['z'])/len(scifi_hits[hitStartStation]['z']) if scifi_hits[hitStartStation]['z'] else float("-999")
    # branch_vars["hitStart_z"][0] = avgHitZ
    
    # branch_vars["hitStart_centroid_x"][0] = branch_vars[f"centroid_scifi{hitStartStation}_x"][0]
    # branch_vars["hitStart_centroid_y"][0] = branch_vars[f"centroid_scifi{hitStartStation}_y"][0]
    # branch_vars["hitStart_avg_x"][0] = branch_vars[f"avg_scifi{hitStartStation}_x"][0]
    # branch_vars["hitStart_avg_y"][0] = branch_vars[f"avg_scifi{hitStartStation}_y"][0]
    
    


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
   
def process_vetoHitTime(all_hits, branch_vars):
    # Filter veto hits (detType == 1)
    veto_hits = [h for h in all_hits if h["detType"] == 1]

    # Compute earliest and latest per station
    per_station = {}
    for s in (1, 2, 3):
        times = [h["hit_time"] for h in veto_hits if h["station"] == s]
        per_station[s] = {
            "earliest": min(times) if times else -1,  # use -1 or 0 if no hit
            "latest":   max(times) if times else -1,
        }

    # Compute overall earliest/latest
    all_times = [h["hit_time"] for h in veto_hits]
    overall_earliest = min(all_times) if all_times else -1
    overall_latest   = max(all_times) if all_times else -1

    # Fill the branch variables
    branch_vars["vetoHitTime_earlist"][0] = overall_earliest
    branch_vars["vetoHitTime_latest"][0]  = overall_latest

    for s in (1, 2, 3):
        branch_vars[f"vetoHitTime_earlist_veto{s}"][0] = per_station[s]["earliest"]
        branch_vars[f"vetoHitTime_latest_veto{s}"][0]  = per_station[s]["latest"]
    
    
    
    
    

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

def process_hits(event, vetoHits, snd_geo, new_tree, branch_vars, eventId, args):
    """Process all hits in the event and update hits array and averages."""
    MC = args.type
    Scifi = snd_geo.modules['Scifi']
    MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()
    
    #read plane positions, vertical->top_x->A.x, horizontal->right_y->A.y
    
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
        
        all_hits.append({
            "detType": 0,
            "station": station,
            "isVertical": aHit.isVertical(),
            "x":A.x(),
            "y":A.y(),
            "z":A.z(),
            "qdc":this_qdc,
            "hit_time": hit_time
        })

    # MuFilter hits

    
    hit2MC = event.Digi_MuFilterHits2MCPoints[0]
    EvtScifiPoint = event.ScifiPoint
    n_veto_hit = 0
    # print(f"\n================ Event {eventId} ================")
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
            linksToMCPoints = hit2MC.wList(detID)
            start_z = (
                event.MCTrack[1].GetStartZ()
                if event.MCTrack.GetEntries() > 1
                else -999
                )
            branch_vars["start_z"][0] = start_z
            
            if aHit.GetSystem() == 1:
                vh = vetoHits.ConstructedAt(n_veto_hit)
                n_veto_hit += 1
                n_scifiPoint = event.ScifiPoint.GetEntries()
                # print(f"\n------ Veto hit #{n_veto_hit} ------")
                # print(f"Veto Plane: {station}, DetID: {detID}, QDC: {this_qdc}, Time: {hit_time:.2f} ns, StartZ (MCTrack[1]): {start_z:.2f} cm")
                
                # print(f"{'mc_point_i':<12} {'PDG':<8} {'Energy loss [MeV]':<20} {'Position (x, y, z) [cm]'}")
                # print("-" * 70)
                
                total_energy_loss = 0
                vh.scifiPoints.clear()
                for mc_point_i, _ in linksToMCPoints:   
                    if mc_point_i >= n_scifiPoint: #to prevent segmentation fault
                        continue
                    
                    scifi_point = event.ScifiPoint[mc_point_i]
                    pdg = int(scifi_point.PdgCode())
                    el  = float(scifi_point.GetEnergyLoss())
                    x   = float(scifi_point.GetX())
                    y   = float(scifi_point.GetY())
                    z   = float(scifi_point.GetZ())

                    total_energy_loss += el

                    # Construct ScifiMiniPoint in-place, then fill its fields
                    vh.scifiPoints.emplace_back()
                    p = vh.scifiPoints.back()
                    p.pdg = pdg
                    p.energy_loss = el
                    p.x, p.y, p.z = x, y, z


                    #print(f"{mc_point_i:<12} {pdg:<8} {el*1000:<20.4f} ({x:7.2f}, {y:7.2f}, {z:7.2f})")
                    
                # fill veto fields (note: you probably want station+1)
                vh.hit_time   = hit_time
                vh.veto_plane = int(station + 1)
                vh.energy_loss = float(total_energy_loss)
                vh.qdc = float(this_qdc)
        
        all_hits.append({
            "detType": detType,
            "station": station+1,
            "isVertical": aHit.isVertical(),
            "x":A.x(),
            "y":A.y(),
            "z":A.z(),
            "qdc":this_qdc,
            "hit_time": hit_time
        })

    
            
            
        
    process_counts(all_hits, branch_vars)
    process_avgPos(all_hits, branch_vars)
    process_centroid(all_hits, branch_vars)
    process_hit_density(all_hits, branch_vars)
    process_showerTagged(all_hits, branch_vars)
    process_slope(all_hits, branch_vars)
    process_vetoHitTime(all_hits, branch_vars)

    #print_hits_summary(all_hits)
    return 

def get_veto_pos(event, snd_geo):
    MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()

    veto_hits = []
    n_veto_hit = 0
    for aHit in event.Digi_MuFilterHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        detType = aHit.GetSystem()
        station = (detID // 1000) % 10

        MuFilter.GetPosition(detID, A, B)
        if detType == 1:  # veto hits
            x = 0.5 * (A.x() + B.x())
            y = 0.5 * (A.y() + B.y())
            z = 0.5 * (A.z() + B.z())
            signal = aHit.GetSignal() if hasattr(aHit, 'GetSignal') else 0
            veto_hits.append({
                "x": x,
                "y": y,
                "z": z,
                "station": station,
                "signal": signal,
                "detID": detID
            })
            n_veto_hit += 1
            
    #print(veto_hits)
    return(veto_hits)

def fill_track_fields(branch_vars, tagLR, angle_xz, angle_yz,slope_xz, slope_yz, chi2ndf, x0, y0, z0, vetoDy, converged_flag):
    # tagLR is like ("HT","DS") or ("ST","Scifi")
    tag = f"{tagLR[0]}_{tagLR[1]}"
    branch_vars[f"{tag}_angle_xz"][0] = angle_xz
    branch_vars[f"{tag}_angle_yz"][0] = angle_yz
    branch_vars[f"{tag}_slope_xz"][0] = slope_xz
    branch_vars[f"{tag}_slope_yz"][0] = slope_yz
    branch_vars[f"{tag}_Chi2Ndf"][0]  = chi2ndf
    branch_vars[f"{tag}_startX"][0]   = x0
    branch_vars[f"{tag}_startY"][0]   = y0
    branch_vars[f"{tag}_startZ"][0]   = z0
    branch_vars[f"{tag}_vetoDy"][0]   = vetoDy
    branch_vars[f"{tag}_flag"][0] = converged_flag
    
def safe_set(branch_vars, name, value, dtype):
    try:
        branch_vars[name][0] = value
    except Exception:
        # Recreate the array if it was overwritten or invalid
        branch_vars[name] = array.array(dtype, [value])

def main(args):
    print("start processing digi to features")
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)
    preSelect_data, preSelect_tree = open_root_file(args.preSelect_path, tree_name='sndData')
    
    out_file, new_tree = create_output_file(args.out_path, args.mode)
    
    elist_name = "elist"


    if "vetoFree" in args.out_path:
        selection = "preSelect_vetoFree==1"
    elif "vetoTagged" in args.out_path:
        selection = "preSelect_vetoTagged==1"
    else:
        selection = ""

    if selection:
        print(f"Applying selection: {selection}")
        n_match = preSelect_tree.GetEntries(selection)
        print(f"Entries matching selection: {n_match}")
        if n_match == 0:
            new_tree.Write()
            out_file.Close()
            print("No entries matched the selection condition, save empty file")
            return 0

        preSelect_tree.Draw(f">>{elist_name}", selection, "entrylist")
        elist = ROOT.gDirectory.Get(elist_name)

        if not elist or not isinstance(elist, ROOT.TEntryList):
            raise RuntimeError("Failed to create or retrieve TEntryList")
        
        preSelect_tree.SetEntryList(elist)
    else:
        raise ValueError("No selection condition determined from output file name.")

    branches = [
        ("runId", 'i'), ("eventId", 'i'), ("pdgCode", 'i'), ("isMC", 'i'), ("eventIndex", 'i'),
        ("px", 'f'), ("py", 'f'), ("pz", 'f'),  # Floats
        ("x", 'f'), ("y", 'f'), ("z", 'f'),    # Floats
        
        # ("count_veto1", 'i'), ("count_veto2", 'i'), ("count_veto3", 'i'),  ("count_veto", 'i'), 
        # ("count_scifi1", 'i'), ("count_scifi2", 'i'), ("count_scifi3", 'i'),("count_scifi4", 'i'), ("count_scifi5", 'i'), ("count_scifi", 'i'),
        # ("count_us1", 'i'), ("count_us2", 'i'), ("count_us3", 'i'),("count_us4", 'i'), ("count_us5", 'i'), ("count_us", 'i'),
        # ("count_ds1", 'i'), ("count_ds2", 'i'), ("count_ds3", 'i'), ("count_ds4", 'i'),("count_ds", 'i'),
        
        # ("avg_veto1_y", 'd'), ("avg_veto2_y", 'd'), ("avg_veto3_x", 'd'), ("avg_veto_x", 'd'), ("avg_veto_y", 'd'),
        # ("avg_scifi1_x", 'd'), ("avg_scifi1_y", 'd'),
        # ("avg_scifi2_x", 'd'), ("avg_scifi2_y", 'd'),
        # ("avg_scifi3_x", 'd'), ("avg_scifi3_y", 'd'),
        # ("avg_scifi4_x", 'd'), ("avg_scifi4_y", 'd'),
        # ("avg_scifi5_x", 'd'), ("avg_scifi5_y", 'd'), ("avg_scifi_y", 'd'), ("avg_scifi_x", 'd'),
        # ("avg_us1_y", 'd'), ("avg_us2_y", 'd'), ("avg_us3_y", 'd'), ("avg_us4_y", 'd'), ("avg_us5_y", 'd'), ("avg_us_y", 'd'),
        # ("avg_ds1_x", 'd'), ("avg_ds1_y", 'd'),
        # ("avg_ds2_x", 'd'), ("avg_ds2_y", 'd'),
        # ("avg_ds3_x", 'd'), ("avg_ds3_y", 'd'),
        # ("avg_ds4_x", 'd'), ("avg_ds4_y", 'd'), ("avg_ds_x", 'd'), ("avg_ds_y", 'd'),
        
        # ("centroid_veto1_y", 'd'), ("centroid_veto2_y", 'd'), ("centroid_veto3_x", 'd'), ("centroid_veto_y", 'd'), ("centroid_veto_x", 'd'), 
        # ("centroid_scifi1_x", 'd'), ("centroid_scifi1_y", 'd'),
        # ("centroid_scifi2_x", 'd'), ("centroid_scifi2_y", 'd'),
        # ("centroid_scifi3_x", 'd'), ("centroid_scifi3_y", 'd'),
        # ("centroid_scifi4_x", 'd'), ("centroid_scifi4_y", 'd'),
        # ("centroid_scifi5_x", 'd'), ("centroid_scifi5_y", 'd'), ("centroid_scifi_x", 'd'), ("centroid_scifi_y", 'd'),
        # ("centroid_us1_y", 'd'), ("centroid_us2_y", 'd'),("centroid_us3_y", 'd'), ("centroid_us4_y", 'd'), ("centroid_us5_y", 'd'), ("centroid_us_y", 'd'),
        # ("centroid_ds1_x", 'd'), ("centroid_ds1_y", 'd'),
        # ("centroid_ds2_x", 'd'), ("centroid_ds2_y", 'd'),
        # ("centroid_ds3_x", 'd'), ("centroid_ds3_y", 'd'),
        # ("centroid_ds4_x", 'd'), ("centroid_ds4_y", 'd'), ("centroid_ds_x", 'd'), ("centroid_ds_y", 'd'),

        # # Hit density sums per plane
        # ("density_veto1", 'd'), ("density_veto2", 'd'), ("density_veto3", 'd'), ("density_veto", 'd'),
        # ("density_scifi1", 'd'), ("density_scifi2", 'd'), ("density_scifi3", 'd'), ("density_scifi4", 'd'), ("density_scifi5", 'd'), ("density_scifi", 'd'),
        # ("density_us1", 'd'), ("density_us2", 'd'), ("density_us3", 'd'), ("density_us4", 'd'), ("density_us5", 'd'), ("density_us", 'd'),
        # ("density_ds1", 'd'), ("density_ds2", 'd'), ("density_ds3", 'd'), ("density_ds4", 'd'), ("density_ds", 'd'),
        # ("density_total", 'd'),

        # ("showerTagged", 'i'),
        # ("showerStartStation", 'i'),
        # ("showerStart_z", 'd'),
        # ("showerStart_centroid_x", 'd'), ("showerStart_centroid_y", 'd'),
        # ("showerStart_avg_x", 'd'), ("showerStart_avg_y", 'd'),
        
        # ("hitStartStation",'i'),
        # ("hitStart_z", 'd'),
        # ("hitStart_centroid_x", 'd'), ("hitStart_centroid_y", 'd'),
        # ("hitStart_avg_x", 'd'), ("hitStart_avg_y", 'd'),

        # ("avgPos_slope_x", 'd'), ("avgPos_slope_y", 'd'),
        # ("centroid_slope_x", 'd'), ("centroid_slope_y", 'd'),
        # ("vetoHitTime_earlist", 'd'), ("vetoHitTime_latest", 'd'),
        # ("vetoHitTime_earlist_veto1", 'd'), ("vetoHitTime_latest_veto1", 'd'),
        # ("vetoHitTime_earlist_veto2", 'd'), ("vetoHitTime_latest_veto2", 'd'),
        # ("vetoHitTime_earlist_veto3", 'd'), ("vetoHitTime_latest_veto3", 'd'),
        
        ("start_z", 'd'),
        ("fluka_weight", 'd'),
        
        ("HT_track", 'i'),
        ("ST_track", 'i'),
        
        ("HT_DS_angle_xz", 'd'), ("HT_DS_angle_yz", 'd'), ("HT_DS_slope_xz", 'd'), ("HT_DS_slope_yz", 'd'), ("HT_DS_Chi2Ndf", 'd'), ("HT_DS_startX", 'd'),  ("HT_DS_startY", 'd'),  ("HT_DS_startZ", 'd'),  ("HT_DS_vetoDy", 'd'),  ("HT_DS_flag", 'i'),
        ("HT_Scifi_angle_xz", 'd'), ("HT_Scifi_angle_yz", 'd'), ("HT_Scifi_slope_xz", 'd'), ("HT_Scifi_slope_yz", 'd'),("HT_Scifi_Chi2Ndf", 'd'), ("HT_Scifi_startX", 'd'),  ("HT_Scifi_startY", 'd'),  ("HT_Scifi_startZ", 'd'),  ("HT_Scifi_vetoDy", 'd'),  ("HT_Scifi_flag", 'i'),
        ("HT_DS_to_Scifi_x", 'd'), ("HT_DS_to_Scifi_y", 'd'),
        ("ST_DS_angle_xz", 'd'), ("ST_DS_angle_yz", 'd'), ("ST_DS_slope_xz", 'd'), ("ST_DS_slope_yz", 'd'), ("ST_DS_Chi2Ndf", 'd'), ("ST_DS_startX", 'd'),  ("ST_DS_startY", 'd'),  ("ST_DS_startZ", 'd'),  ("ST_DS_vetoDy", 'd'),  ("ST_DS_flag", 'i'),
        ("ST_Scifi_angle_xz", 'd'), ("ST_Scifi_angle_yz", 'd'), ("ST_Scifi_slope_xz", 'd'), ("ST_Scifi_slope_yz", 'd'),  ("ST_Scifi_Chi2Ndf", 'd'), ("ST_Scifi_startX", 'd'),  ("ST_Scifi_startY", 'd'),  ("ST_Scifi_startZ", 'd'),  ("ST_Scifi_vetoDy", 'd'),  ("ST_Scifi_flag", 'i'),
        ("ST_DS_to_Scifi_x", 'd'),("ST_DS_to_Scifi_y", 'd'),
    ]

    # Dictionary to hold branch variables
    branch_vars = {}

    # Create branches dynamically
    for name, dtype in branches:
        branch_vars[name] = array.array(dtype, [-999])  # Initialize the array
        new_tree.Branch(name, branch_vars[name], f"{name}/{dtype.upper()}")
        
    #vetoHits = ROOT.std.vector('VetoHit')()
    
    
    ROOT.gROOT.ProcessLine(".L /afs/cern.ch/user/z/zhibin/work/snd-ml/convertData/EventClass.h+")
    vetoHits = ROOT.TClonesArray("VetoHit")

    # Branch on the vector; ROOT will serialize the container each entry
    new_tree.Branch("vetoHits", vetoHits)
   
    total_tracks_per_event = []
    track_type_map = {
        1:  ("ST", "Scifi"),   # ST SciFi
        11: ("HT", "Scifi"),   # HT SciFi
        3:  ("ST", "DS"),      # ST DS
        13: ("HT", "DS"),      # HT DS
    }
    collected = {
        ("HT","DS"):     None,
        ("HT","Scifi"):  None,
        ("ST","DS"):     None,
        ("ST","Scifi"):  None,
    }
    # Process each event
    for i in tqdm(range(raw_tree.GetEntries()), desc="Processing events", unit="event"):
        vetoHits.Clear()
        for key_name, dtype in branches:
            safe_set(branch_vars, key_name, -999, dtype)

        entry_number = i
        raw_tree.GetEntry(entry_number)
        preSelect_tree.GetEntry(entry_number)
        
        branch_vars["eventIndex"][0] = entry_number
        branch_vars["runId"][0] = raw_tree.EventHeader.GetRunId()
        
        
        if ('MC' in  args.type):
            branch_vars["isMC"][0] = 1
            try:
                branch_vars["eventId"][0] = raw_tree.EventHeader.GetEventNumber()
            except Exception:
                branch_vars["eventId"][0] = raw_tree.EventHeader.GetMCEntryNumber()
                
            event_pdg0 = raw_tree.MCTrack[0].GetPdgCode()
            event_pdg1 = raw_tree.MCTrack[1].GetPdgCode()

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
            
            
        
        elif('real' in  args.type):
            branch_vars["isMC"][0] = 0
            branch_vars["pdgCode"][0] = 0
            branch_vars["eventId"][0] = raw_tree.EventHeader.GetEventNumber()
        #process_hits(raw_tree,vetoHits, snd_geo, new_tree, branch_vars, branch_vars["eventId"][0],  args)
        
        
        n_tracks = len(raw_tree.Reco_MuonTracks)
        total_tracks_per_event.append(n_tracks)
        
        
        # creat branch "passingMuon"
        
        ## Good DS track 
        # 1 converged fit (1 track for 1 tracking method)
        # 2 slopes in both projections below 80 mrad ('getSlopeXZ', 'getSlopeYZ')
        # 3 χ2/ndf < 5 (getChi2Ndf)
        # 4 extrapolated DS track at the Veto planes is within 3 cm of a fired Veto bar
        # 
        
        ## Good Scifi track
        # 1 converged fit (1 track for 1 tracking method) ('getTrackType()')
        # 2 χ2/ndf < 20
        
        #Good tracks
        # If SciFi track’s and DS track’s projections on the reference plane are within 3 cm distance ('extrapolateToPlaneAtZ')
        
        veto_hits = get_veto_pos(raw_tree, snd_geo)
        
        track_counts = {name: 0 for name in track_type_map.values()}
        
        #if raw_tree.Reco_MuonTracks.GetEntries() <4:
            #print(raw_tree.Reco_MuonTracks.GetEntries())
            #continue
        for mctrack in raw_tree.MCTrack :
            #primary muon
            if mctrack.GetMotherId()==-1:
                branch_vars['fluka_weight'][0] = mctrack.GetWeight()
        
        for i_track, muon_track in enumerate(raw_tree.Reco_MuonTracks):
            ttype = muon_track.getTrackType()
            #print(dir(muon_track))
            converged_flag = muon_track.getTrackFlag()
            if ttype not in track_type_map:
                continue

            slope_xz = muon_track.getSlopeXZ()
            slope_yz = muon_track.getSlopeYZ()
            angle_xz = muon_track.getAngleXZ() * 1e3  # mrad
            angle_yz = muon_track.getAngleYZ() * 1e3  # mrad
            start_pos = muon_track.getStart()
            x0, y0, z0 = start_pos.x(), start_pos.y(), start_pos.z()
            chi2ndf = muon_track.getChi2Ndf()

            # compute dy to each veto plane at its z; choose BEST (max |dy|)
            max_abs_dy = np.nan
            if veto_hits:
                dy_list = []
                for vetoHit in veto_hits:
                    veto_z = vetoHit['z']
                    veto_y = vetoHit['y']
                    extrapolated_y = y0 + slope_yz * (veto_z - z0)
                    dy = veto_y - extrapolated_y
                    dy_list.append(dy)
                if dy_list:
                    max_abs_dy = min(dy_list, key=lambda v: abs(v))
                
            tagLR = track_type_map[ttype]  # ("HT","DS") etc.
            fill_track_fields(branch_vars, tagLR, angle_xz, angle_yz,slope_xz, slope_yz, chi2ndf, x0, y0, z0, max_abs_dy, converged_flag)

            # keep a copy for DS↔Scifi comparison later
            collected[tagLR] = dict(
                angle_xz=angle_xz, angle_yz=angle_yz,
                slope_xz=slope_xz, slope_yz=slope_yz,
                x0=x0, y0=y0, z0=z0
            )
                        
        z_ref = 490.0  # make sure units match your geometry (cm if your inputs are cm)

        for tier in ("HT", "ST"):
            ds = collected[(tier, "DS")]
            sc = collected[(tier, "Scifi")]
            if ds is not None and sc is not None:
                # extrapolate x(z), y(z) = x0 + slope_xz*(z - z0), y0 + slope_yz*(z - z0)
                x_ds = ds["x0"] + ds["slope_xz"] * (z_ref - ds["z0"])
                y_ds = ds["y0"] + ds["slope_yz"] * (z_ref - ds["z0"])

                x_sc = sc["x0"] + sc["slope_xz"] * (z_ref - sc["z0"])
                y_sc = sc["y0"] + sc["slope_yz"] * (z_ref - sc["z0"])

                dx = abs(x_ds - x_sc)  # use abs for distance; drop abs for signed delta
                dy = abs(y_ds - y_sc)

                branch_vars[f"{tier}_DS_to_Scifi_x"][0] = dx
                branch_vars[f"{tier}_DS_to_Scifi_y"][0] = dy
                
                
        # if i>20000:
        #    break
        
        new_tree.Fill()
    
    # summarize how many events have N tracks
    hist = collections.Counter(total_tracks_per_event)

    print("\n=== Number of events vs. total tracks ===")
    for n_tracks, n_events in sorted(hist.items()):
        print(f"{n_tracks} tracks → {n_events} events")
    # Finalize the output file
    new_tree.Write()
    out_file.Close()
    print("finish processing digi to feature")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--preSelectPath", dest="preSelect_path", help="pre selection data file path", required=True)
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default='RECREATE')
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)

    args = parser.parse_args()

    main(args)

# python digi_2_features_muonDown.py -p /eos/experiment/sndlhc/users/zhibin/MC_muon/down/scoring_1.8_Bfield_4xstat/preSelect_MC_muon_down_scoring_1.8_Bfield_4xstat_3.root -d /eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_1.8_Bfield_4xstat/sndLHC.Ntuple-TGeant4-160urad_magfield_2022TCL6_muons_rock_2e8pr_Trks.root -g /eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_1.8_Bfield_4xstat/geofile_full.Ntuple-TGeant4.root -o /eos/experiment/sndlhc/users/zhibin/MC_muon/down/scoring_1.8_Bfield_4xstat/vetoTagged_feature_MC_muon_down_scoring_1.8_Bfield_4xstat_3.root -t MC_muon