import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import array
from collections import defaultdict
import math
from analysis.analyses.snd_analysis_2024_0mu.sciFiTools import selectHits, getSumDensity
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


def process_count_and_qdc(SciFi_hits, MuFilter_hits, branch_vars):
    # ---------- helpers ----------
    def qdc_value(x):
        # x can be float/int or dict-like (e.g. GetAllSignals())
        if isinstance(x, dict):
            return float(sum(x.values()))
        try:
            return float(x)
        except Exception:
            return 0.0

    # ---------- init counters ----------
    scifi_counts = [0] * 5
    veto_counts = [0] * 3
    us_counts = [0] * 5
    ds_counts = [0] * 4

    scifi_qdc = [0.0] * 5
    veto_qdc = [0.0] * 3
    us_qdc = [0.0] * 5
    ds_qdc = [0.0] * 4

    # for SciFi event-ID logic (need both views info)
    scifi_hor = [0] * 5
    scifi_ver = [0] * 5

    # ---------- fill SciFi ----------
    for h in SciFi_hits:
        st = int(h.get("station", 0))
        if 1 <= st <= 5:
            i = st - 1
            scifi_counts[i] += 1
            scifi_qdc[i] += qdc_value(h.get("qdc", 0.0))
            if h.get("isVertical", False):
                scifi_ver[i] += 1
            else:
                scifi_hor[i] += 1

    # ---------- fill MuFilter ----------
    for h in MuFilter_hits:
        det = int(h.get("detType", -1))   # 1=veto, 2=US, 3=DS
        st = int(h.get("station", 0))     # expected 1-based
        q = qdc_value(h.get("qdc", 0.0))

        if det == 1 and 1 <= st <= 3:
            veto_counts[st - 1] += 1
            veto_qdc[st - 1] += q
        elif det == 2 and 1 <= st <= 5:
            us_counts[st - 1] += 1
            us_qdc[st - 1] += q
        elif det == 3 and 1 <= st <= 4:
            ds_counts[st - 1] += 1
            ds_qdc[st - 1] += q

    # ---------- write count branches ----------
    for i in range(3):
        branch_vars[f"count_veto{i+1}"][0] = veto_counts[i]
    branch_vars["count_veto"][0] = sum(veto_counts)

    for i in range(5):
        branch_vars[f"count_scifi{i+1}"][0] = scifi_counts[i]
    branch_vars["count_scifi"][0] = sum(scifi_counts)

    for i in range(5):
        branch_vars[f"count_us{i+1}"][0] = us_counts[i]
    branch_vars["count_us"][0] = sum(us_counts)

    for i in range(4):
        branch_vars[f"count_ds{i+1}"][0] = ds_counts[i]
    branch_vars["count_ds"][0] = sum(ds_counts)

    # ---------- write qdc branches ----------
    for i in range(3):
        branch_vars[f"qdc_veto{i+1}"][0] = veto_qdc[i]
    branch_vars["qdc_veto"][0] = sum(veto_qdc)

    for i in range(5):
        branch_vars[f"qdc_scifi{i+1}"][0] = scifi_qdc[i]
    branch_vars["qdc_scifi"][0] = sum(scifi_qdc)

    for i in range(5):
        branch_vars[f"qdc_us{i+1}"][0] = us_qdc[i]
    branch_vars["qdc_us"][0] = sum(us_qdc)

    for i in range(4):
        branch_vars[f"qdc_ds{i+1}"][0] = ds_qdc[i]
    branch_vars["qdc_ds"][0] = sum(ds_qdc)
    

 

def process_vetoHitTime(MuFilter_hits, branch_vars):
    # Filter veto hits (detType == 1)
    veto_hits = [h for h in MuFilter_hits if h["detType"] == 1]

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
    
    
    

def process_avgPos(SciFi_hits, MuFilter_hits, branch_vars):
    """
    Compute average hit positions for:
      - veto_{1-2}_y
      - veto_3_x
      - scifi_{1-5}_{x,y}
      - us_{1-5}_y
      - ds_{1-4}_{x,y}

    Convention:
      - horizontal planes -> average y
      - vertical planes   -> average x
    """
    sums = defaultdict(float)
    counts = defaultdict(int)

    # combine both hit collections
    all_hits = SciFi_hits + MuFilter_hits

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

        elif detType == 2:  # Upstream
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

    # Fill all requested branch_vars safely
    for key in branch_vars:
        if key.startswith("avg_"):
            branch_vars[key][0] = sums[key] / counts[key] if counts[key] > 0 else -999.

    

def filter_SciFiHits(SciFi_hits, lower_time_threshold=0.5, upper_time_threshold=2.3, bin_width=0.25):
    """
    Filter SciFi hits around MPV of hit time (per station + orientation).

    Inputs:
      SciFi_hits: list of dicts, each with at least:
        - station (int)
        - isVertical (bool)
        - hitTimeCY (float)  # in clock cycles
      lower_time_threshold, upper_time_threshold: in clock cycles

    Returns:
      filtered_hits, peak_by_group
    """
    if not SciFi_hits:
        return [], {}

    groups = defaultdict(list)
    for h in SciFi_hits:
        groups[(h["station"], h["isVertical"])].append(h)

    filtered = []
    peak_by_group = {}

    MAX_BINS = 2000
    for key, hits in groups.items():
        times = np.array([h["hitTimeCY"] for h in hits], dtype=float)

        # Robust MPV estimate via histogram mode
        tmin, tmax = float(times.min()), float(times.max())
        if tmax <= tmin:
            peak = tmin
        else:
            width = tmax - tmin
            nbins = int(np.ceil(width / bin_width))
            nbins = max(10, min(nbins, MAX_BINS))
            
            hist, edges = np.histogram(times, bins=nbins, range=(tmin, tmax))
            i_max = int(np.argmax(hist))
            peak = 0.5 * (edges[i_max] + edges[i_max + 1])

        peak_by_group[key] = peak

        lo = peak - lower_time_threshold
        hi = peak + upper_time_threshold
        for h in hits:
            t = h["hitTimeCY"]
            if lo <= t <= hi:
                filtered.append(h)

    return filtered, peak_by_group


def fill_mycode_density(branch_vars):
    station_x = [float(branch_vars[f"density_scifi{i}_x"][0]) for i in range(1, 6)]
    station_y = [float(branch_vars[f"density_scifi{i}_y"][0]) for i in range(1, 6)]

    # Match getSumDensity station ranking: choose by total density x+y
    station_sum = [station_x[i] + station_y[i] for i in range(5)]
    station_min = [min(station_x[i], station_y[i]) for i in range(5)]

    best_idx = 0
    best_sum = station_sum[0]

    second_idx = -1
    second_sum = 0.0

    for i in range(5):
        s = station_sum[i]

        # Match getSumDensity: strict '>' for best, so first max wins ties
        if s > best_sum:
            second_idx = best_idx
            second_sum = best_sum
            best_idx = i
            best_sum = s
        # Match getSumDensity: second-best must be strictly below best and strictly above current second
        elif s < best_sum and s > second_sum:
            second_idx = i
            second_sum = s

    branch_vars["density_mycode_scifi"][0] = station_min[best_idx]
    branch_vars["density_mycodescifi_second"][0] = station_min[second_idx] if second_idx >= 0 else 0.0
    branch_vars["density_mycode_scifi_hor"][0] = station_y[best_idx]
    branch_vars["density_mycode_scifi_ver"][0] = station_x[best_idx]

def hitWeightDensity(SciFi_hits, branch_vars):
    # Python equivalent of sndSciFiTools.cxx hitWeightComputation (width = 1 cm)
    def sum_hit_weights_1d(positions, width=1.0):
        if not positions:
            return 0.0

        pos = sorted(float(p) for p in positions)
        n = len(pos)
        left = 0
        right = 0
        total = 0.0

        for i in range(n):
            x = pos[i]
            while right < n and pos[right] <= x + width:
                right += 1
            while left < n and pos[left] < x - width:
                left += 1

            neighbors = (right - left) - 1  # exclude self
            if neighbors < 0:
                neighbors = 0
            total += neighbors

        return total

    # collect positions per station and orientation
    # vertical -> x density, horizontal -> y density
    st_x = {i: [] for i in range(1, 6)}
    st_y = {i: [] for i in range(1, 6)}

    for h in SciFi_hits:
        st = int(h.get("station", 0))
        if st < 1 or st > 5:
            continue

        if h.get("isVertical", False):
            st_x[st].append(h.get("x", 0.0))
        else:
            st_y[st].append(h.get("y", 0.0))

    total_x = 0.0
    total_y = 0.0

    for st in range(1, 6):
        dx = sum_hit_weights_1d(st_x[st], width=1.0)
        dy = sum_hit_weights_1d(st_y[st], width=1.0)
        d = dx + dy

        branch_vars[f"density_scifi{st}_x"][0] = dx
        branch_vars[f"density_scifi{st}_y"][0] = dy
        branch_vars[f"density_scifi{st}"][0] = d

        total_x += dx
        total_y += dy

    branch_vars["density_scifi_x"][0] = total_x
    branch_vars["density_scifi_y"][0] = total_y
    branch_vars["density_scifi"][0] = total_x + total_y

    return branch_vars

def process_hits(args, event, vetoHits, snd_geo, branch_vars):
    """Process all hits in the event and update hits array and averages."""
    eventId = branch_vars["eventId"][0]
    MC = args.type
    Scifi = snd_geo.modules['Scifi']
    MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()
    

    sel_hits = selectHits(event, MC = (True if "MC" in args.type else False))  
    dens, dens2, dver, dhor = getSumDensity(sel_hits, return_2ndhighest=True, return_hv=True)
    branch_vars["density_sndsw_scifi"][0] = dens
    branch_vars["density_sndsw_scifi_second"][0] = dens2
    branch_vars["density_sndsw_scifi_ver"][0] = dver
    branch_vars["density_sndsw_scifi_hor"][0] = dhor
    

        
    
    SciFi_hits = []
    # Process SciFi hits
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        station = detID // 1000000
        hitTime = aHit.GetTime()
        clock_cycle = hitTime/6.25
        qdc = aHit.GetSignal(0)
        mat = aHit.GetMat()
        sipm = aHit.GetSiPM()
        channel = aHit.GetSiPMChan()
        layer_channel = channel + sipm*128 + mat*4*128
        
        Scifi.GetSiPMPosition(detID, A, B)
        
        SciFi_hits.append({
            "detType": 0,
            "station": station,
            "isVertical": aHit.isVertical(),
            "layer_channel": layer_channel,
            "qdc":qdc,
            "x":A.x(),
            "y":A.y(),
            "z":A.z(),
            "hitTimeCY": clock_cycle
        })

    # Process MuFilter hits
    MuFilter_hits = []
    n_veto_hit = 0
    for aHit in event.Digi_MuFilterHits:
        if not aHit.isValid():
            continue
        
        # process veto hits
        if aHit.GetSystem() == 1:
            vh = vetoHits.ConstructedAt(n_veto_hit)
            n_veto_hit += 1
            
            total_energy_loss = 0
            vh.mcPoints.clear()
            for mc_point_i, mc_point_weight in linksToMCPoints:   
                mc_point = event.MuFilterPoint[mc_point_i]
                pdg = int(mc_point.PdgCode())
                el  = float(mc_point.GetEnergyLoss())
                x   = float(mc_point.GetX())
                y   = float(mc_point.GetY())
                z   = float(mc_point.GetZ())

                total_energy_loss += el

                # Construct ScifiMiniPoint in-place, then fill its fields
                vh.mcPoints.emplace_back()
                p = vh.mcPoints.back()
                p.pdg = pdg
                p.energy_loss = el
                p.x, p.y, p.z = x, y, z
                p.weight = mc_point_weight
                
            # fill veto fields (note: you probably want station+1)
            vh.hit_time   = hit_time
            vh.veto_plane = int(station + 1)
            vh.energy_loss = float(total_energy_loss)
            vh.qdc = float(this_qdc)
                
        MuFilter.GetPosition(detID, A, B)
        detID = aHit.GetDetectorID()
        detType = aHit.GetSystem()
        station = (detID // 1000) % 10
        hitTime = aHit.GetTime()
        clock_cycle = hitTime/6.25
        barIndex = detID%100
        
        
        qdc = 0.0
        for key, value in aHit.GetAllSignals():
            qdc += value
        
        
        MuFilter_hits.append({
            "detType": detType,
            "station": station+1,
            "isVertical": aHit.isVertical(),
            "qdc":qdc,
            "barIndex":barIndex,
            "hitTimeCY": clock_cycle,
            "x":A.x(),
            "y":A.y(),
            "z":A.z(),
        })
    
    #filter SciFi hits for real data
    SciFi_hits, peak_by_group = filter_SciFiHits(
            SciFi_hits,
            lower_time_threshold=0.5,
            upper_time_threshold=1.2
        )
    
    hitWeightDensity(SciFi_hits, branch_vars)
    
    process_count_and_qdc(SciFi_hits, MuFilter_hits, branch_vars)
    
    process_avgPos(SciFi_hits, MuFilter_hits, branch_vars)
    
    fill_mycode_density(branch_vars)
    
    process_vetoHitTime(MuFilter_hits, branch_vars)

    return 





def main(args):
    print("start processing digi to features")
    
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)
    preSelect_data, preSelect_tree = open_root_file(args.preSelect_path, tree_name='cutFlowSummary')
    preSelect_tree.SetAlias("EventDeltat_m1_100", "EventDeltat_-1_100")
    
    out_file, new_tree = create_output_file(args.out_path, args.mode)
    
    elist_name = "elist"


    if "veto_" in args.out_path:
        selection = "AvgSFChan == 1 && NoVetoHits == 0"
    else:
        selection = "AvgSFChan == 1 && NoVetoHits == 1"

    if "MC" not in args.type:
        selection = selection + "&& StableBeams==1 && IP1 == 1 && EventDeltat_1_100 == 1"
    if selection:
        print(f"Applying selection: {selection}")
        n_match = preSelect_tree.GetEntries(selection)
        print(f"Entries matching selection: {n_match}")
        if n_match == 0:
            out_file.cd()
            new_tree.Write()
            cutflow_selected = preSelect_tree.CloneTree(0)
            cutflow_selected.SetName("cutFlowSummary")
            cutflow_selected.Write()
            out_file.Close()
            print("No entries matched the selection condition, saved empty sndData and cutFlowSummary trees")
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
        
        ("count_veto1", 'i'), ("count_veto2", 'i'), ("count_veto3", 'i'),  ("count_veto", 'i'), 
        ("count_scifi1", 'i'), ("count_scifi2", 'i'), ("count_scifi3", 'i'),("count_scifi4", 'i'), ("count_scifi5", 'i'), ("count_scifi", 'i'),
        ("count_us1", 'i'), ("count_us2", 'i'), ("count_us3", 'i'),("count_us4", 'i'), ("count_us5", 'i'), ("count_us", 'i'),
        ("count_ds1", 'i'), ("count_ds2", 'i'), ("count_ds3", 'i'), ("count_ds4", 'i'),("count_ds", 'i'),
        
        ("qdc_veto1", 'd'), ("qdc_veto2", 'd'), ("qdc_veto3", 'd'),  ("qdc_veto", 'd'), 
        ("qdc_scifi1", 'd'), ("qdc_scifi2", 'd'), ("qdc_scifi3", 'd'),("qdc_scifi4", 'd'), ("qdc_scifi5", 'd'), ("qdc_scifi", 'd'),
        ("qdc_us1", 'd'), ("qdc_us2", 'd'), ("qdc_us3", 'd'),("qdc_us4", 'd'), ("qdc_us5", 'd'), ("qdc_us", 'd'),
        ("qdc_ds1", 'd'), ("qdc_ds2", 'd'), ("qdc_ds3", 'd'), ("qdc_ds4", 'd'),("qdc_ds", 'd'),
        
        ("avg_veto1_y", 'd'), ("avg_veto2_y", 'd'), ("avg_veto3_x", 'd'), ("avg_veto_x", 'd'), ("avg_veto_y", 'd'),
        ("avg_scifi1_x", 'd'), ("avg_scifi1_y", 'd'),
        ("avg_scifi2_x", 'd'), ("avg_scifi2_y", 'd'),
        ("avg_scifi3_x", 'd'), ("avg_scifi3_y", 'd'),
        ("avg_scifi4_x", 'd'), ("avg_scifi4_y", 'd'),
        ("avg_scifi5_x", 'd'), ("avg_scifi5_y", 'd'), ("avg_scifi_y", 'd'), ("avg_scifi_x", 'd'),
        ("avg_us1_y", 'd'), ("avg_us2_y", 'd'), ("avg_us3_y", 'd'), ("avg_us4_y", 'd'), ("avg_us5_y", 'd'), ("avg_us_y", 'd'),
        ("avg_ds1_x", 'd'), ("avg_ds1_y", 'd'),
        ("avg_ds2_x", 'd'), ("avg_ds2_y", 'd'),
        ("avg_ds3_x", 'd'), ("avg_ds3_y", 'd'),
        ("avg_ds4_x", 'd'), ("avg_ds4_y", 'd'), ("avg_ds_x", 'd'), ("avg_ds_y", 'd'),
        
        # Hit density sums per plane
        ("density_scifi1", 'd'), ("density_scifi2", 'd'), ("density_scifi3", 'd'), ("density_scifi4", 'd'), ("density_scifi5", 'd'), ("density_scifi", 'd'),
        ("density_scifi1_x", 'd'), ("density_scifi2_x", 'd'), ("density_scifi3_x", 'd'), ("density_scifi4_x", 'd'), ("density_scifi5_x", 'd'), ("density_scifi_x", 'd'),
        ("density_scifi1_y", 'd'), ("density_scifi2_y", 'd'), ("density_scifi3_y", 'd'), ("density_scifi4_y", 'd'), ("density_scifi5_y", 'd'), ("density_scifi_y", 'd'),
        
        ("density_mycode_scifi", 'd'), ("density_mycodescifi_second", 'd'), ("density_mycode_scifi_hor", 'd'), ("density_mycode_scifi_ver", 'd'), 
        
        ("density_sndsw_scifi", 'd'), ("density_sndsw_scifi_second", 'd'), ("density_sndsw_scifi_hor", 'd'), ("density_sndsw_scifi_ver", 'd'), 
        

        ("vetoHitTime_earlist", 'd'), ("vetoHitTime_latest", 'd'),
        ("vetoHitTime_earlist_veto1", 'd'), ("vetoHitTime_latest_veto1", 'd'),
        ("vetoHitTime_earlist_veto2", 'd'), ("vetoHitTime_latest_veto2", 'd'),
        ("vetoHitTime_earlist_veto3", 'd'), ("vetoHitTime_latest_veto3", 'd'),
        
        ("start_z", 'd'),
    ]

    # Dictionary to hold branch variables
    branch_vars = {}

    # Create branches dynamically
    for name, dtype in branches:
        branch_vars[name] = array.array(dtype, [-999])  # Initialize the array
        new_tree.Branch(name, branch_vars[name], f"{name}/{dtype.upper()}")
        
    #vetoHits = ROOT.std.vector('VetoHit')()
    
    
    if not ROOT.TClass.GetClass("VetoHit"):
        ROOT.gROOT.ProcessLine('.L /afs/cern.ch/user/z/zhibin/work/snd-ml/convertData/EventClass.h+')

    vetoHits = ROOT.TClonesArray("VetoHit")
    new_tree.Branch("vetoHits", vetoHits)

    
    for i in range(elist.GetN()):
        vetoHits.Clear()
        if i % 10000 == 0:
            print(f"processed {i} events")
        # Reset all branch variables before filling them
        for key in branch_vars:
            branch_vars[key][0] = -999

        entry_number = elist.GetEntry(i)
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
        process_hits(args, raw_tree, vetoHits, snd_geo, branch_vars)
        #if i>2:
        #    break
        new_tree.Fill()
    # Finalize the output file
    new_tree.Write()
    cutflow_selected = preSelect_tree.CopyTree(selection)
    cutflow_selected.SetName("cutFlowSummary")
    cutflow_selected.Write()
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
    
# python digi_2_features.py -p /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/preSelect_MC_neutrino_volTarget_100fb-1_0.root -d /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_20240126_digCPP.root -g /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/geofile_full.Genie-TGeant4.root -o /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/vetoTagged_feature_MC_neutrino_volTarget_100fb-1_0.root -t MC_neutrino

# python digi_2_features.py -p /eos/experiment/sndlhc/users/zhibin/MC_muon/down/scoring_1.8_Bfield_4xstat/preSelect_MC_muon_down_scoring_1.8_Bfield_4xstat_3.root -d /eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_1.8_Bfield_4xstat/sndLHC.Ntuple-TGeant4-160urad_magfield_2022TCL6_muons_rock_2e8pr_Trks.root -g /eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_1.8_Bfield_4xstat/geofile_full.Ntuple-TGeant4.root -o /eos/experiment/sndlhc/users/zhibin/MC_muon/down/scoring_1.8_Bfield_4xstat/vetoTagged_feature_MC_muon_down_scoring_1.8_Bfield_4xstat_3.root -t MC_muon
# python digi_2_features.py -p /eos/experiment/sndlhc/users/zhibin/MC_neutrino/2024_vm/10/nueAnalysisFilter_MC_neutrino_2024_vm_10.root -d /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2024/nu14/volume_volTarget/10/sndLHC.Genie-TGeant4_dig.root -g /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2024/nu14/volume_volTarget/10/geofile_full.Genie-TGeant4.root -o /eos/experiment/sndlhc/users/zhibin/MC_neutrino/2024_vm/10/feature_MC_neutrino_2024_vm_10.root -t MC_ve