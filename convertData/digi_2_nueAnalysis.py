import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import array
from tqdm import tqdm
from collections import defaultdict
import numpy as np


def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo

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

    for key, hits in groups.items():
        times = np.array([h["hitTimeCY"] for h in hits], dtype=float)

        # Robust MPV estimate via histogram mode
        tmin, tmax = float(times.min()), float(times.max())
        if tmax <= tmin:
            peak = tmin
        else:
            nbins = max(10, int(np.ceil((tmax - tmin) / bin_width)))
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

def Fiducial_Selection(SciFi_hits, MuFilter_hits, branch_vars):
    # ---------- SciFi averages ----------
    ver_channels = [h["layer_channel"] for h in SciFi_hits if h.get("isVertical", False)]
    hor_channels = [h["layer_channel"] for h in SciFi_hits if not h.get("isVertical", False)]

    avg_ver = sum(ver_channels) / len(ver_channels) if ver_channels else -1.0
    avg_hor = sum(hor_channels) / len(hor_channels) if hor_channels else -1.0

    branch_vars["avgScifiVer"][0] = avg_ver
    branch_vars["avgScifiHor"][0] = avg_hor

    # Avg SciFi Vertical [200,1200] and Horizontal [300,1336]
    pass_scifi_fid = (
        len(ver_channels) > 0 and
        len(hor_channels) > 0 and
        200 <= avg_ver <= 1200 and
        300 <= avg_hor <= 1336
    )
    branch_vars["avgScifiFiducial"][0] = int(pass_scifi_fid)

    # ---------- US bars veto ----------
    # MuFilter hits are expected as detType=2 (US), station=1..5
    us1_bars = [h["barIndex"] for h in MuFilter_hits if h.get("detType") == 2 and h.get("station") == 1]
    us2_bars = [h["barIndex"] for h in MuFilter_hits if h.get("detType") == 2 and h.get("station") == 2]

    us1_avg = sum(us1_bars) / len(us1_bars) if us1_bars else -1.0
    us2_avg = sum(us2_bars) / len(us2_bars) if us2_bars else -1.0

    branch_vars["US1avg"][0] = us1_avg
    branch_vars["US2avg"][0] = us2_avg

    # user-requested rule: avg bar > 2 and <= 8 in both US1 and US2
    pass_us_bars = (
        (len(us1_bars) > 0 and len(us2_bars) > 0 and
        us1_avg > 2 and us1_avg <= 8 and
        us2_avg > 2 and us2_avg <= 8)
    )
    branch_vars["USBarsVeto"][0] = int(pass_us_bars)

    # ---------- Veto-hit requirement ----------
    has_veto_hit = any(h.get("detType") == 1 for h in MuFilter_hits)
    branch_vars["noVetoHit"][0] = int(not has_veto_hit)

    return branch_vars

def nueEventIdentification(SciFi_hits, MuFilter_hits, branch_vars):
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

    # ---------- nue event identification ----------
    both_views = [(scifi_hor[i] > 0 and scifi_ver[i] > 0) for i in range(5)]

    # at least 2 consecutive SciFi stations with both views hit
    consecutive = any(both_views[i] and both_views[i + 1] for i in range(4))
    branch_vars["consecutiveSciFiHits"][0] = int(consecutive)

    # SciFi continuity: after first station with both views, all downstream must also have both views
    first = next((i for i, ok in enumerate(both_views) if ok), None)
    if first is None:
        continuity = 1
    else:
        continuity = int(all(both_views[i] for i in range(first, 5)))
    branch_vars["SciFiContinuity"][0] = continuity

    # US planes 0 and 1 hit => stations 1 and 2 in this 1-based storage
    branch_vars["USPlaneHit_0_1"][0] = int(us_counts[0] > 0 and us_counts[1] > 0)

    # total SciFi hits > 35
    branch_vars["SciFiHit35"][0] = int(sum(scifi_counts) > 35)

    # total US QDC > 600
    branch_vars["USQDC600"][0] = int(sum(us_qdc) > 600.0)

    # no hit in last DS plane (ds4)
    branch_vars["NoHitLastDS"][0] = int(ds_counts[3] == 0)

    return branch_vars

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

def process_hits(args, event, snd_geo, branch_vars):
    """Process all hits in the event and update hits array and averages."""
    Scifi = snd_geo.modules['Scifi']
    # MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()

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
    for aHit in event.Digi_MuFilterHits:
        if not aHit.isValid():
            continue
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
        })
        
    # process_vetoHitTime(all_hits, branch_vars)
    
    #filter SciFi hits for real data
    if "real" in args.type:
        SciFi_hits, peak_by_group = filter_SciFiHits(
            SciFi_hits,
            lower_time_threshold=0.5,
            upper_time_threshold=2.3
        )

        
    
    #process Fiducial Selection
    Fiducial_Selection(SciFi_hits,MuFilter_hits, branch_vars)
    
    if not (branch_vars["noVetoHit"][0] and branch_vars["USBarsVeto"][0] and branch_vars["avgScifiFiducial"][0]):
        return False

    branch_vars["FiducialSelection"][0] = 1
    nueEventIdentification(SciFi_hits, MuFilter_hits, branch_vars)
    
    if (branch_vars["consecutiveSciFiHits"][0] and branch_vars["SciFiContinuity"][0] and branch_vars["USPlaneHit_0_1"][0] and branch_vars["USQDC600"][0] and branch_vars["NoHitLastDS"][0]):
        branch_vars["nueEventIdentification"][0] =1
    hitWeightDensity(SciFi_hits, branch_vars)

    return True



def reset_branches(branch_vars, default=-999):
    for a in branch_vars.values():
        a[0] = default



def main(args):
    print("start processing digi to preSelection")
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)
    
    out_file, new_tree = create_output_file(args.out_path, args.mode)

    branches = [
        # ID info
        ("runId", 'i'), ("eventId", 'i'), ("isMC", 'i'), ("eventIndex", 'i'),
        # MC info
        ("pdgCode", 'i'), ("energy", 'd'),
        
        # real data info
        ("stableBeam", 'i'), ("IP1BunchCrossing", 'i'), ("PreEvtClockCycle100", 'i'),("PreEvtClockCycle", 'd'),
        
        
        ("dataQuality", 'i'),
        
        # features
        ## Fiducial cuts
        ("avgScifiFiducial", 'i'), #  Avg SciFi Vertical channel [200,1200] and Horizontal [300, 1336]
        ("avgScifiHor", 'd'),("avgScifiVer", 'd'),
        ("USBarsVeto", 'i'), # Avg US bar (sum of bar index / number of hits ),  larger than 2 and small_equal than 8 for US plane 0 and 1
        ("US1avg", 'd'),("US2avg", 'd'),
        
        ("FiducialSelection", 'i'),
        
        ("noVetoHit", 'i'),
        
        # nue Event Identification
        ("consecutiveSciFiHits",'i'), # at least 2 consecutive SciFi planes
        ("SciFiContinuity",'i'), # if a SciFi station has hits (both ver and hor), the following SciFi planes must have hits
        ("USPlaneHit_0_1",'i'), # require valid hit for US plane 0, and 1
        ("SciFiHit35", 'i'), # Number of SciFi hits larger than 35
        ("USQDC600", 'i'), # total US QDC larger than 600
        ("NoHitLastDS", 'i'), # no hit in the last planes
        ("nueEventIdentification", 'i'),
        
        ("count_veto1", 'i'), ("count_veto2", 'i'), ("count_veto3", 'i'),  ("count_veto", 'i'), 
        ("count_scifi1", 'i'), ("count_scifi2", 'i'), ("count_scifi3", 'i'),("count_scifi4", 'i'), ("count_scifi5", 'i'), ("count_scifi", 'i'),
        ("count_us1", 'i'), ("count_us2", 'i'), ("count_us3", 'i'),("count_us4", 'i'), ("count_us5", 'i'), ("count_us", 'i'),
        ("count_ds1", 'i'), ("count_ds2", 'i'), ("count_ds3", 'i'), ("count_ds4", 'i'),("count_ds", 'i'),
        
        ("qdc_veto1", 'd'), ("qdc_veto2", 'd'), ("qdc_veto3", 'd'),  ("qdc_veto", 'd'), 
        ("qdc_scifi1", 'd'), ("qdc_scifi2", 'd'), ("qdc_scifi3", 'd'),("qdc_scifi4", 'd'), ("qdc_scifi5", 'd'), ("qdc_scifi", 'd'),
        ("qdc_us1", 'd'), ("qdc_us2", 'd'), ("qdc_us3", 'd'),("qdc_us4", 'd'), ("qdc_us5", 'd'), ("qdc_us", 'd'),
        ("qdc_ds1", 'd'), ("qdc_ds2", 'd'), ("qdc_ds3", 'd'), ("qdc_ds4", 'd'),("qdc_ds", 'd'),
        
        # Sum of SciFi hit density weight
        ("density_scifi1", 'd'), ("density_scifi2", 'd'), ("density_scifi3", 'd'), ("density_scifi4", 'd'), ("density_scifi5", 'd'), ("density_scifi", 'd'),
        ("density_scifi1_x", 'd'), ("density_scifi2_x", 'd'), ("density_scifi3_x", 'd'), ("density_scifi4_x", 'd'), ("density_scifi5_x", 'd'), ("density_scifi_x", 'd'),
        ("density_scifi1_y", 'd'), ("density_scifi2_y", 'd'), ("density_scifi3_y", 'd'), ("density_scifi4_y", 'd'), ("density_scifi5_y", 'd'), ("density_scifi_y", 'd'),
    ]

    # Dictionary to hold branch variables
    branch_vars = {}

    # Create branches dynamically
    for name, dtype in branches:
        branch_vars[name] = array.array(dtype, [-999])  # Initialize the array
        new_tree.Branch(name, branch_vars[name], f"{name}/{dtype.upper()}")
    
    prev_evt_time = None

    for i_event, event in tqdm(enumerate(raw_tree), total=raw_tree.GetEntries()):
        reset_branches(branch_vars)
        
        h = event.EventHeader

        branch_vars["eventIndex"][0] = i_event
        branch_vars["runId"][0] = h.GetRunId()

        if "MC" in args.type:
            branch_vars["isMC"][0] = 1
            try:
                branch_vars["eventId"][0] = h.GetEventNumber()
            except Exception:
                branch_vars["eventId"][0] = h.GetMCEntryNumber()

            event_pdg0 = raw_tree.MCTrack[0].GetPdgCode()
            event_pdg1 = raw_tree.MCTrack[1].GetPdgCode()

            branch_vars["energy"][0] = raw_tree.MCTrack[0].GetEnergy()

            neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
            if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
                branch_vars["pdgCode"][0] = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
            else:
                branch_vars["pdgCode"][0] = event_pdg0

            # Not meaningful for MC
            branch_vars["stableBeam"][0] = -1
            branch_vars["IP1BunchCrossing"][0] = -1
            branch_vars["PreEvtClockCycle100"][0] = -1

        elif "real" in args.type:
            branch_vars["isMC"][0] = 0
            branch_vars["pdgCode"][0] = 0
            branch_vars["eventId"][0] = h.GetEventNumber()

            # 1) Stable beams
            branch_vars["stableBeam"][0] = int(
                h.GetBeamMode() == int(ROOT.LhcBeamMode.StableBeams)
            )

            # 2) IP1 bunch crossing
            branch_vars["IP1BunchCrossing"][0] = int(h.isIP1())

            # 3) Previous event > 100 clock cycles away
            if prev_evt_time is None:
                branch_vars["PreEvtClockCycle100"][0] = 0
                branch_vars["PreEvtClockCycle"][0] = -1.0
            else:
                dt = h.GetEventTime() - prev_evt_time
                dt_cy = dt / 6.25
                branch_vars["PreEvtClockCycle"][0] = dt_cy
                branch_vars["PreEvtClockCycle100"][0] = int(dt_cy > 100)

            prev_evt_time = h.GetEventTime()

        if (("real" in args.type) and branch_vars["PreEvtClockCycle100"][0] and branch_vars["IP1BunchCrossing"][0] and branch_vars["stableBeam"][0]) or ("MC" in args.type):
            branch_vars["dataQuality"][0] = 1
            process_hits(args, event, snd_geo, branch_vars)
        new_tree.Fill()
        # if i_event>1e4:
        #     break

    # Finalize the output file
    new_tree.Write()
    out_file.Close()
    print("finish processing digi to nueAnalysis")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default='RECREATE')
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)

    args = parser.parse_args()

    main(args)