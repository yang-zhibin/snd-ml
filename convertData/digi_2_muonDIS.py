import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import array
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import array
from analysis.analyses.snd_analysis_2024_0mu.sciFiTools import selectHits, getSumDensity


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

def reset_scalar_branches(branch_vars, default=-999):
    """Reset scalar branches for a new event."""
    for v in branch_vars.values():
        v[0] = default


def reset_vector_branches(vector_branches):
    """Clear all vector branches for a new event."""
    for v in vector_branches.values():
        v.clear()


def make_track_vector_branches():
    """Create vector branches for primary and secondary tracks."""
    return {
        # primary tracks
        "primary_pdg": ROOT.std.vector("int")(),
        "primary_energy": ROOT.std.vector("double")(),
        "primary_startX": ROOT.std.vector("double")(),
        "primary_startY": ROOT.std.vector("double")(),
        "primary_startZ": ROOT.std.vector("double")(),
        "primary_px": ROOT.std.vector("double")(),
        "primary_py": ROOT.std.vector("double")(),
        "primary_pz": ROOT.std.vector("double")(),

        # secondary tracks
        "secondary_pdg": ROOT.std.vector("int")(),
        "secondary_energy": ROOT.std.vector("double")(),
        "secondary_startX": ROOT.std.vector("double")(),
        "secondary_startY": ROOT.std.vector("double")(),
        "secondary_startZ": ROOT.std.vector("double")(),
        "secondary_px": ROOT.std.vector("double")(),
        "secondary_py": ROOT.std.vector("double")(),
        "secondary_pz": ROOT.std.vector("double")(),
    }


def branch_vector_vars(tree, vector_branches):
    """Attach vector branches to the output tree."""
    for name, vec in vector_branches.items():
        tree.Branch(name, vec)


def fill_track_info(prefix, tr, vector_branches):
    """Append one track's information into the chosen vector group."""
    vector_branches[f"{prefix}_pdg"].push_back(int(tr.GetPdgCode()))
    vector_branches[f"{prefix}_energy"].push_back(float(tr.GetEnergy()))
    vector_branches[f"{prefix}_startX"].push_back(float(tr.GetStartX()))
    vector_branches[f"{prefix}_startY"].push_back(float(tr.GetStartY()))
    vector_branches[f"{prefix}_startZ"].push_back(float(tr.GetStartZ()))
    vector_branches[f"{prefix}_px"].push_back(float(tr.GetPx()))
    vector_branches[f"{prefix}_py"].push_back(float(tr.GetPy()))
    vector_branches[f"{prefix}_pz"].push_back(float(tr.GetPz()))

def process_hits(events):
    # loop over MCPoints
    # 
     

def process_tracks(event, track_vector_branches):
    """
    Save tracks event by event.

    Definition used here:
      - primary   : MCTrack[0]
      - secondary : tracks with mother_id == 0
    """
    mc_tracks = event.MCTrack
    if mc_tracks is None:
        return 0, 0

    n_primary = 0
    n_secondary = 0

    n_tracks = int(mc_tracks.GetEntriesFast())
    for i in range(n_tracks):
        tr = mc_tracks.At(i)
        if tr is None:
            continue

        mother_id = int(tr.GetMotherId())

        if i == 0:
            fill_track_info("primary", tr, track_vector_branches)
            n_primary += 1

        elif mother_id == 0:
            fill_track_info("secondary", tr, track_vector_branches)
            n_secondary += 1

    return n_primary, n_secondary


def process_event_info(event, snd_geo, branch_vars):
    """
    Compute event-level summary information for SciFi and MuFilter hits.

    Position convention:
      - vertical hit   -> use x
      - horizontal hit -> use y

    Subsystems:
      - SciFi: stations 1..5
      - Veto : stations 1..3   (MuFilter system == 1)
      - US   : stations 1..5   (MuFilter system == 2)
      - DS   : stations 1..4   (MuFilter system == 3)

    Written branches:
      SciFi:
        count_scifi, qdc_scifi
        count_scifi1..5, qdc_scifi1..5
        avg_scifi_ver_x, avg_scifi_hor_y
        avg_scifi1_ver_x..avg_scifi5_ver_x
        avg_scifi1_hor_y..avg_scifi5_hor_y

      Veto:
        count_veto, qdc_veto
        count_veto1..3, qdc_veto1..3
        avg_veto_ver_x, avg_veto_hor_y
        avg_veto1_ver_x..avg_veto3_ver_x
        avg_veto1_hor_y..avg_veto3_hor_y

      US:
        count_us, qdc_us
        count_us1..5, qdc_us1..5
        avg_us_ver_x, avg_us_hor_y
        avg_us1_ver_x..avg_us5_ver_x
        avg_us1_hor_y..avg_us5_hor_y

      DS:
        count_ds, qdc_ds
        count_ds1..4, qdc_ds1..4
        avg_ds_ver_x, avg_ds_hor_y
        avg_ds1_ver_x..avg_ds4_ver_x
        avg_ds1_hor_y..avg_ds4_hor_y
    """

    def safe_set(name, value):
        """Write branch only if it exists in branch_vars."""
        if name in branch_vars:
            branch_vars[name][0] = value

    def avg_or_default(sum_val, count_val, default=-999.0):
        return (sum_val / count_val) if count_val > 0 else default

    def init_subsystem(nstations):
        return {
            "total_count": 0,
            "total_qdc": 0.0,
            "total_ver_count": 0,
            "total_hor_count": 0,
            "sum_ver_x": 0.0,
            "sum_hor_y": 0.0,
            "counts": [0] * nstations,
            "qdc": [0.0] * nstations,
            "ver_counts": [0] * nstations,
            "hor_counts": [0] * nstations,
            "sum_ver_x_st": [0.0] * nstations,
            "sum_hor_y_st": [0.0] * nstations,
        }

    def write_subsystem(prefix, data):
        nstations = len(data["counts"])

        # totals
        safe_set(f"count_{prefix}", data["total_count"])
        safe_set(f"qdc_{prefix}", data["total_qdc"])

        # overall averages
        safe_set(f"avg_{prefix}_ver_x",
                 avg_or_default(data["sum_ver_x"], data["total_ver_count"]))
        safe_set(f"avg_{prefix}_hor_y",
                 avg_or_default(data["sum_hor_y"], data["total_hor_count"]))

        # per-station
        for i in range(nstations):
            st = i + 1
            safe_set(f"count_{prefix}{st}", data["counts"][i])
            safe_set(f"qdc_{prefix}{st}", data["qdc"][i])

            safe_set(
                f"avg_{prefix}{st}_ver_x",
                avg_or_default(data["sum_ver_x_st"][i], data["ver_counts"][i])
            )
            safe_set(
                f"avg_{prefix}{st}_hor_y",
                avg_or_default(data["sum_hor_y_st"][i], data["hor_counts"][i])
            )

    # Geometry modules
    Scifi = snd_geo.modules["Scifi"]
    MuFilter = snd_geo.modules["MuFilter"]

    A = ROOT.TVector3()
    B = ROOT.TVector3()

    # Initialize containers
    scifi = init_subsystem(5)
    veto = init_subsystem(3)
    us = init_subsystem(5)
    ds = init_subsystem(4)

    # -------------------------
    # Process SciFi hits
    # -------------------------
    if hasattr(event, "Digi_ScifiHits"):
        for aHit in event.Digi_ScifiHits:
            if not aHit.isValid():
                continue

            detID = int(aHit.GetDetectorID())
            station = detID // 1000000
            if not (1 <= station <= 5):
                continue

            qdc = float(aHit.GetSignal(0))
            is_vertical = bool(aHit.isVertical())

            Scifi.GetSiPMPosition(detID, A, B)
            x = float(A.x())
            y = float(A.y())

            i = station - 1

            scifi["total_count"] += 1
            scifi["total_qdc"] += qdc
            scifi["counts"][i] += 1
            scifi["qdc"][i] += qdc

            if is_vertical:
                scifi["total_ver_count"] += 1
                scifi["sum_ver_x"] += x
                scifi["ver_counts"][i] += 1
                scifi["sum_ver_x_st"][i] += x
            else:
                scifi["total_hor_count"] += 1
                scifi["sum_hor_y"] += y
                scifi["hor_counts"][i] += 1
                scifi["sum_hor_y_st"][i] += y

    # -------------------------
    # Process MuFilter hits
    # -------------------------
    if hasattr(event, "Digi_MuFilterHits"):
        for aHit in event.Digi_MuFilterHits:
            if not aHit.isValid():
                continue

            detID = int(aHit.GetDetectorID())
            detType = int(aHit.GetSystem())     # 1=veto, 2=US, 3=DS
            station0 = (detID // 1000) % 10     # raw station index
            station = station0 + 1              # convert to 1-based
            is_vertical = bool(aHit.isVertical())

            qdc = 0.0
            for key, value in aHit.GetAllSignals():
                qdc += float(value)

            MuFilter.GetPosition(detID, A, B)
            x = float(A.x())
            y = float(A.y())

            # Choose subsystem and valid station range
            target = None
            max_station = 0

            if detType == 1:
                target = veto
                max_station = 3
            elif detType == 2:
                target = us
                max_station = 5
            elif detType == 3:
                target = ds
                max_station = 4
            else:
                continue

            if not (1 <= station <= max_station):
                continue

            i = station - 1

            target["total_count"] += 1
            target["total_qdc"] += qdc
            target["counts"][i] += 1
            target["qdc"][i] += qdc

            if is_vertical:
                target["total_ver_count"] += 1
                target["sum_ver_x"] += x
                target["ver_counts"][i] += 1
                target["sum_ver_x_st"][i] += x
            else:
                target["total_hor_count"] += 1
                target["sum_hor_y"] += y
                target["hor_counts"][i] += 1
                target["sum_hor_y_st"][i] += y

    # -------------------------
    # Write outputs
    # -------------------------
    write_subsystem("scifi", scifi)
    write_subsystem("veto", veto)
    write_subsystem("us", us)
    write_subsystem("ds", ds)

    
def main(args):
    print("start processing digi to preSelection")

    snd_geo = setup_geometry(args.geo_path)
    raw_data, raw_tree = open_root_file(args.digi_path)
    out_file, new_tree = create_output_file(args.out_path, args.mode)

    # scalar event-level branches
    scalar_branches = [
        ("runId", "i"),
        ("eventId", "i"),
        ("isMC", "i"),
        ("eventIndex", "i"),
        ("nPrimary", "i"),
        ("nSecondary", "i"),

        # SciFi event-level summary
        ("count_scifi", "i"),
        ("count_scifi1", "i"),
        ("count_scifi2", "i"),
        ("count_scifi3", "i"),
        ("count_scifi4", "i"),
        ("count_scifi5", "i"),

        ("qdc_scifi", "d"),
        ("qdc_scifi1", "d"),
        ("qdc_scifi2", "d"),
        ("qdc_scifi3", "d"),
        ("qdc_scifi4", "d"),
        ("qdc_scifi5", "d"),

        ("avg_scifi_ver_x", "d"),
        ("avg_scifi_hor_y", "d"),

        ("avg_scifi1_ver_x", "d"),
        ("avg_scifi1_hor_y", "d"),

        ("avg_scifi2_ver_x", "d"),
        ("avg_scifi2_hor_y", "d"),

        ("avg_scifi3_ver_x", "d"),
        ("avg_scifi3_hor_y", "d"),

        ("avg_scifi4_ver_x", "d"),
        ("avg_scifi4_hor_y", "d"),

        ("avg_scifi5_ver_x", "d"),
        ("avg_scifi5_hor_y", "d"),

        # Veto event-level summary
        ("count_veto", "i"),
        ("count_veto1", "i"),
        ("count_veto2", "i"),
        ("count_veto3", "i"),

        ("qdc_veto", "d"),
        ("qdc_veto1", "d"),
        ("qdc_veto2", "d"),
        ("qdc_veto3", "d"),

        ("avg_veto_ver_x", "d"),
        ("avg_veto_hor_y", "d"),

        ("avg_veto1_ver_x", "d"),
        ("avg_veto1_hor_y", "d"),

        ("avg_veto2_ver_x", "d"),
        ("avg_veto2_hor_y", "d"),

        ("avg_veto3_ver_x", "d"),
        ("avg_veto3_hor_y", "d"),

        # US event-level summary
        ("count_us", "i"),
        ("count_us1", "i"),
        ("count_us2", "i"),
        ("count_us3", "i"),
        ("count_us4", "i"),
        ("count_us5", "i"),

        ("qdc_us", "d"),
        ("qdc_us1", "d"),
        ("qdc_us2", "d"),
        ("qdc_us3", "d"),
        ("qdc_us4", "d"),
        ("qdc_us5", "d"),

        ("avg_us_ver_x", "d"),
        ("avg_us_hor_y", "d"),

        ("avg_us1_ver_x", "d"),
        ("avg_us1_hor_y", "d"),

        ("avg_us2_ver_x", "d"),
        ("avg_us2_hor_y", "d"),

        ("avg_us3_ver_x", "d"),
        ("avg_us3_hor_y", "d"),

        ("avg_us4_ver_x", "d"),
        ("avg_us4_hor_y", "d"),

        ("avg_us5_ver_x", "d"),
        ("avg_us5_hor_y", "d"),

        # DS event-level summary
        ("count_ds", "i"),
        ("count_ds1", "i"),
        ("count_ds2", "i"),
        ("count_ds3", "i"),
        ("count_ds4", "i"),

        ("qdc_ds", "d"),
        ("qdc_ds1", "d"),
        ("qdc_ds2", "d"),
        ("qdc_ds3", "d"),
        ("qdc_ds4", "d"),

        ("avg_ds_ver_x", "d"),
        ("avg_ds_hor_y", "d"),

        ("avg_ds1_ver_x", "d"),
        ("avg_ds1_hor_y", "d"),

        ("avg_ds2_ver_x", "d"),
        ("avg_ds2_hor_y", "d"),

        ("avg_ds3_ver_x", "d"),
        ("avg_ds3_hor_y", "d"),

        ("avg_ds4_ver_x", "d"),
        ("avg_ds4_hor_y", "d"),

        ("density_sndsw_scifi", "d"),
        ("density_sndsw_scifi_second", "d"),
        ("density_sndsw_scifi_hor", "d"),
        ("density_sndsw_scifi_ver", "d"),
    ]

    branch_vars = {}
    for name, dtype in scalar_branches:
        branch_vars[name] = array.array(dtype, [-999])
        new_tree.Branch(name, branch_vars[name], f"{name}/{dtype.upper()}")

    # vector track-level branches
    track_vector_branches = make_track_vector_branches()
    branch_vector_vars(new_tree, track_vector_branches)

    for i_event, event in tqdm(enumerate(raw_tree), total=raw_tree.GetEntries()):
        reset_scalar_branches(branch_vars)
        reset_vector_branches(track_vector_branches)

        mc_tracks = event.MCTrack
        if mc_tracks is None:
            continue

        n_tracks = int(mc_tracks.GetEntriesFast())
        if n_tracks < 1:
            continue

        h = event.EventHeader
        branch_vars["eventIndex"][0] = int(i_event)
        branch_vars["runId"][0] = int(h.GetRunId())

        if "MC" in args.type:
            branch_vars["isMC"][0] = 1
            try:
                branch_vars["eventId"][0] = int(h.GetEventNumber())
            except Exception:
                branch_vars["eventId"][0] = int(h.GetMCEntryNumber())
        else:
            raise ValueError("only process MC muonDIS")

        n_primary, n_secondary = process_tracks(event, track_vector_branches)
        branch_vars["nPrimary"][0] = n_primary
        branch_vars["nSecondary"][0] = n_secondary
        

        # Skip empty events
        if n_primary == 0 and n_secondary == 0:
            continue

        # fill event-level SciFi summaries
        process_event_info(event, snd_geo, branch_vars)
        
        # sel_hits = selectHits(event, MC = (True if "MC" in args.type else False))  
        dens, dens2, dver, dhor = getSumDensity(event.Digi_ScifiHits, return_2ndhighest=True, return_hv=True)
        branch_vars["density_sndsw_scifi"][0] = dens
        branch_vars["density_sndsw_scifi_second"][0] = dens2
        branch_vars["density_sndsw_scifi_ver"][0] = dver
        branch_vars["density_sndsw_scifi_hor"][0] = dhor

        new_tree.Fill()

        # if i_event > 2000:
        #     break

    new_tree.Write()
    out_file.Close()
    print("finish processing digi to muonDIS")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default='RECREATE')
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)

    args = parser.parse_args()

    main(args)