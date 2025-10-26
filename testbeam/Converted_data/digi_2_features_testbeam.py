import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import array
from collections import defaultdict
import math

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


        if detType == 0:  # SciFi
            if 1 <= station <= 4:
                if isVertical:
                    key = f"avg_scifi{station}_x"
                    sums[key] += hit["x"]
                    counts[key] += 1
                else:
                    key = f"avg_scifi{station}_y"
                    sums[key] += hit["y"]
                    counts[key] += 1

    # Write averages to branch_vars
    for key in sums:
        avg = sums[key] / counts[key] if counts[key] > 0 else -999
        branch_vars[key][0] = avg

    # Fill missing branches with default -999
    for key in [
        "avg_scifi1_x", "avg_scifi1_y", "avg_scifi2_x", "avg_scifi2_y",
        "avg_scifi3_x", "avg_scifi3_y", "avg_scifi4_x", "avg_scifi4_y",
    ]:
        if key not in branch_vars:
            continue  # skip if branch not defined
        if key not in counts:
            branch_vars[key][0] = -999


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


        if detType == 0:  # SciFi
            if 1 <= station <= 4:
                if isVertical:
                    key = f"centroid_scifi{station}_x"
                    weighted_sums[key] += qdc * hit["x"]
                    total_qdc[key] += qdc
                else:
                    key = f"centroid_scifi{station}_y"
                    weighted_sums[key] += qdc * hit["y"]
                    total_qdc[key] += qdc

    # Finalize
    all_keys = [
        "centroid_scifi1_x", "centroid_scifi1_y", "centroid_scifi2_x", "centroid_scifi2_y",
        "centroid_scifi3_x", "centroid_scifi3_y", "centroid_scifi4_x", "centroid_scifi4_y",
    ]

    for key in all_keys:
        if total_qdc[key] > 0:
            branch_vars[key][0] = weighted_sums[key] / total_qdc[key]
        else:
            branch_vars[key][0] = -999
            

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

    # Set default value for missing planes
    for key in [
        "density_scifi1", "density_scifi2", "density_scifi3", "density_scifi4",
    ]:
        if key not in branch_vars:
            continue
        if key not in plane_hits:
            branch_vars[key][0] = -999
    
    
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

def process_hits(event, snd_geo, branch_vars):
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
        
        all_hits.append({
            "detType": 0,
            "station": station,
            "isVertical": aHit.isVertical(),
            "x":A.x(),
            "y":A.y(),
            "z":A.z(),
            "qdc":this_qdc
        })

    process_showerTagged(all_hits, branch_vars)
    process_counts(all_hits, branch_vars)
    process_avgPos(all_hits, branch_vars)
    process_centroid(all_hits, branch_vars)
    process_hit_density(all_hits, branch_vars)
    process_slope(all_hits, branch_vars)
    


    #print_hits_summary(all_hits)
    return 


def main(args):
    print("start processing digi to features")
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)
    beam_type = args.pdg
    
    out_file, new_tree = create_output_file(args.out_path, args.mode)

    branches = [
        ("runId", 'i'), ("eventId", 'i'), ("pdgCode", 'i'), ("isMC", 'i'), ("eventIndex", 'i'),
        ("px", 'f'), ("py", 'f'), ("pz", 'f'),  # Floats
        ("x", 'f'), ("y", 'f'), ("z", 'f'),    # Floats
        
        ("count_scifi1", 'i'), ("count_scifi2", 'i'), ("count_scifi3", 'i'),("count_scifi4", 'i'), ("count_scifi", 'i'),
        
        ("avg_scifi1_x", 'd'), ("avg_scifi1_y", 'd'),
        ("avg_scifi2_x", 'd'), ("avg_scifi2_y", 'd'),
        ("avg_scifi3_x", 'd'), ("avg_scifi3_y", 'd'),
        ("avg_scifi4_x", 'd'), ("avg_scifi4_y", 'd'),
        
        ("centroid_scifi1_x", 'd'), ("centroid_scifi1_y", 'd'),
        ("centroid_scifi2_x", 'd'), ("centroid_scifi2_y", 'd'),
        ("centroid_scifi3_x", 'd'), ("centroid_scifi3_y", 'd'),
        ("centroid_scifi4_x", 'd'), ("centroid_scifi4_y", 'd'),


        # Hit density sums per plane
        ("density_scifi1", 'd'), ("density_scifi2", 'd'), ("density_scifi3", 'd'), ("density_scifi4", 'd'),

        ("showerTagged", 'i'),
        ("showerStartStation", 'i'),
        
        ("avgPos_slope_x", 'd'), ("avgPos_slope_y", 'd'),
        ("centroid_slope_x", 'd'), ("centroid_slope_y", 'd'),

        #energy, 
    ]

    # Dictionary to hold branch variables
    branch_vars = {}

    # Create branches dynamically
    for name, dtype in branches:
        branch_vars[name] = array.array(dtype, [-999])  # Initialize the array
        new_tree.Branch(name, branch_vars[name], f"{name}/{dtype.upper()}")
        
    # Process each event

    for i in range(raw_tree.GetEntries()):
        if i % 10000 == 0:
            print(f"processed {i} events")
        # Reset all branch variables before filling them
        for key in branch_vars:
            branch_vars[key][0] = -999

        entry_number = i
        raw_tree.GetEntry(entry_number)
        
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

        process_hits(raw_tree, snd_geo, branch_vars)
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

    args = parser.parse_args()

    main(args)
    
# python digi_2_features.py -p /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/preSelect_MC_neutrino_volTarget_100fb-1_0.root -d /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_20240126_digCPP.root -g /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/geofile_full.Genie-TGeant4.root -o /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/vetoTagged_feature_MC_neutrino_volTarget_100fb-1_0.root -t MC_neutrino