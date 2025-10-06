import ROOT
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import SndlhcGeo
import numpy as np
import re
import math
pdg_db = ROOT.TDatabasePDG.Instance()
pdg_label_map = {
        12: "ve",
        14: "vm",
        16: "vt",
        112: "NC",
        114: "NC",
        116: "NC",
    }

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

def load_chains_from_neutrinoMC(metadata_csv: str, tree_name: str = "cbmsim", max_rows: int = 1):
    """
    Reads a metadata CSV and loads digi and newDigi ROOT files into TChains,
    skipping missing files and stopping after a limited number of valid rows.

    Args:
        metadata_csv (str): Path to the metadata CSV file.
        tree_name (str): Name of the TTree to load from each file.
        max_rows (int): Maximum number of valid rows to process.

    Returns:
        tuple: (chain_digi, chain_newDigi) as ROOT.TChain objects
    """
    metadata = pd.read_csv(metadata_csv)
    chain_digi = ROOT.TChain(tree_name)
    chain_newDigi = ROOT.TChain(tree_name)

    valid_count = 0

    for index, row in metadata.iterrows():
        digi_path = row.get("digi_path", "")
        new_digi_path = row.get("newDigi_path", "")

        # Skip row if any path is missing or file does not exist
        if not (isinstance(digi_path, str) and os.path.exists(digi_path)):
            continue
        if not (isinstance(new_digi_path, str) and os.path.exists(new_digi_path)):
            continue
        try:
            digi_file = ROOT.TFile.Open(digi_path)
            digi_tree = digi_file.Get("cbmsim")
            digi_entries = digi_tree.GetEntries() if digi_tree else -1
        except:
            digi_entries = -1
        finally:
            digi_file.Close()

        try:
            new_digi_file = ROOT.TFile.Open(new_digi_path)
            new_digi_tree = new_digi_file.Get("cbmsim")
            new_digi_entries = new_digi_tree.GetEntries() if new_digi_tree else -1
        except:
            new_digi_entries = -1
        finally:
            new_digi_file.Close()
        if (index == 3):
            continue
        print(f'----{index}---')
        print(f"    -> Entries in digi_path: {digi_entries} {digi_path}")
        print(f"    -> Entries in newDigi_path: {new_digi_entries} {new_digi_path}")
            
        if (new_digi_entries==-1):
            # Remove the corrupted or unusable newDigi file
            if os.path.exists(row['newDigi_path']):
                print(f"    -> Removing {row['newDigi_path']}")
                os.remove(row['newDigi_path'])
            else:
                print(f"    -> File {row['newDigi_path']} does not exist.")

            # Also remove the corresponding raw file, if needed
            if 'newRaw_path' in row and os.path.exists(row['newRaw_path']):
                print(f"    -> Removing {row['newRaw_path']}")
                os.remove(row['newRaw_path'])
            elif 'newRaw_path' in row:
                print(f"    -> File {row['newRaw_path']} does not exist.")
            continue
        
        chain_digi.Add(digi_path)
        chain_newDigi.Add(new_digi_path)

        valid_count += 1
        if valid_count >= max_rows:
            break

    return chain_digi, chain_newDigi



def load_chains_from_metadata(metadata_csv: str, tree_name: str = "cbmsim", max_rows: int = 1):

    metadata = pd.read_csv(metadata_csv)
    chain_digi = ROOT.TChain(tree_name)

    valid_count = 0

    for index, row in metadata.iterrows():
        digi_path = row.get("digi_path", "")

        # Skip row if any path is missing or file does not exist
        if not (isinstance(digi_path, str) and os.path.exists(digi_path)):
            continue
        
        try:
            digi_file = ROOT.TFile.Open(digi_path)
            digi_tree = digi_file.Get("cbmsim")
            digi_entries = digi_tree.GetEntries() if digi_tree else -1
        except:
            digi_entries = -1
        finally:
            digi_file.Close()

        
        print(f'----{index}---')
        print(f"    -> Entries in digi_path: {digi_entries} {digi_path}")
            
        
        chain_digi.Add(digi_path)

        valid_count += 1
        if valid_count >= max_rows:
            break

    return chain_digi


def fill_vetoCount_vetoTime(
    digi_chain,
    snd_geo,
    data_type,
    count_scifi_threshold=200,
    out_dir="hist_data",
    prefix="veto",
):
    """
    Fill histograms and save PDF plots with data_type in filename.

    Args:
        digi_chain: TChain with Digi_MuFilterHits and Digi_ScifiHits branches
        snd_geo: SND geometry object (with modules['MuFilter'])
        data_type (str): data tag for output filenames (e.g. 'MC2022')
        count_scifi_threshold (int): minimum SciFi hits required to keep the event
        out_dir (str): directory to save plots
        prefix (str): filename prefix for saved plots
    Returns:
        dict of histograms
    """

    os.makedirs(out_dir, exist_ok=True)
    
    import ROOT

    # ---------------- Configurable binning/ranges ----------------
    # Start position (e.g., z in cm). Adjust to your data.
    START_MIN, START_MAX, START_NBINS = 280.0, 360.0, 80

    # Veto counts
    COUNT_ALL_NBINS, COUNT_ALL_MIN, COUNT_ALL_MAX = 36, 0, 36   # global count
    COUNT_S_NBINS,   COUNT_S_MIN,   COUNT_S_MAX   = 12, 0, 12   # per-station count

    # Times (ns)
    TIME_NBINS, TIME_MIN, TIME_MAX = 50, 0.0, 50.0

    # Stations (1,2,3)
    STATIONS = [1, 2, 3]
    STATION_MIN, STATION_MAX = 0.5, 3.5   # for station-index axes

    def _cfg(h):
        h.SetDirectory(0)
        h.Sumw2()
        return h

    # ---------------- 1D histograms ----------------
    veto_count_all = _cfg(ROOT.TH1F("veto_count_all", "Veto Hits per Event (All);veto count;events",
                                    COUNT_ALL_NBINS, COUNT_ALL_MIN, COUNT_ALL_MAX))

    veto_count_s1 = _cfg(ROOT.TH1F("veto_count_s1", "Veto Hits per Event (Station 1);veto count;events",
                                    COUNT_S_NBINS, COUNT_S_MIN, COUNT_S_MAX))
    veto_count_s2 = _cfg(ROOT.TH1F("veto_count_s2", "Veto Hits per Event (Station 2);veto count;events",
                                    COUNT_S_NBINS, COUNT_S_MIN, COUNT_S_MAX))
    veto_count_s3 = _cfg(ROOT.TH1F("veto_count_s3", "Veto Hits per Event (Station 3);veto count;events",
                                    COUNT_S_NBINS, COUNT_S_MIN, COUNT_S_MAX))

    h_hitTime     = _cfg(ROOT.TH1F("h_hitTime", "All Veto Hit Times;time [ns];entries",
                                    TIME_NBINS, TIME_MIN, TIME_MAX))

    # Note: name kept as requested: "earlist_*"
    earlist_hitTime_all = _cfg(ROOT.TH1F("earlist_hitTime_all", "Earliest Veto Hit Time per Event (All);time [ns];events",
                                        TIME_NBINS, TIME_MIN, TIME_MAX))

    earliest_hitTime_s1 = _cfg(ROOT.TH1F("earliest_hitTime_s1", "Earliest Veto Hit Time per Event (Station 1);time [ns];events",
                                        TIME_NBINS, TIME_MIN, TIME_MAX))
    earliest_hitTime_s2 = _cfg(ROOT.TH1F("earliest_hitTime_s2", "Earliest Veto Hit Time per Event (Station 2);time [ns];events",
                                        TIME_NBINS, TIME_MIN, TIME_MAX))
    earliest_hitTime_s3 = _cfg(ROOT.TH1F("earliest_hitTime_s3", "Earliest Veto Hit Time per Event (Station 3);time [ns];events",
                                        TIME_NBINS, TIME_MIN, TIME_MAX))

    latest_hitTime_all = _cfg(ROOT.TH1F("latest_hitTime_all", "Latest Veto Hit Time per Event (All);time [ns];events",
                                        TIME_NBINS, TIME_MIN, TIME_MAX))

    latest_hitTime_s1 = _cfg(ROOT.TH1F("latest_hitTime_s1", "Latest Veto Hit Time per Event (Station 1);time [ns];events",
                                    TIME_NBINS, TIME_MIN, TIME_MAX))
    latest_hitTime_s2 = _cfg(ROOT.TH1F("latest_hitTime_s2", "Latest Veto Hit Time per Event (Station 2);time [ns];events",
                                    TIME_NBINS, TIME_MIN, TIME_MAX))
    latest_hitTime_s3 = _cfg(ROOT.TH1F("latest_hitTime_s3", "Latest Veto Hit Time per Event (Station 3);time [ns];events",
                                    TIME_NBINS, TIME_MIN, TIME_MAX))

    # ---------------- 2D histograms ----------------
    # start_position vs earliest time
    start_vs_earlist_hitTime_all = _cfg(ROOT.TH2F("start_vs_earlist_hitTime_all",
        "Start position vs earliest time (All);start position [cm];earliest time [ns]",
        START_NBINS, START_MIN, START_MAX, TIME_NBINS, TIME_MIN, TIME_MAX))

    start_vs_earlist_hitTime_s1 = _cfg(ROOT.TH2F("start_vs_earlist_hitTime_s1",
        "Start position vs earliest time (Station 1);start position [cm];earliest time [ns]",
        START_NBINS, START_MIN, START_MAX, TIME_NBINS, TIME_MIN, TIME_MAX))
    start_vs_earlist_hitTime_s2 = _cfg(ROOT.TH2F("start_vs_earlist_hitTime_s2",
        "Start position vs earliest time (Station 2);start position [cm];earliest time [ns]",
        START_NBINS, START_MIN, START_MAX, TIME_NBINS, TIME_MIN, TIME_MAX))
    start_vs_earlist_hitTime_s3 = _cfg(ROOT.TH2F("start_vs_earlist_hitTime_s3",
        "Start position vs earliest time (Station 3);start position [cm];earliest time [ns]",
        START_NBINS, START_MIN, START_MAX, TIME_NBINS, TIME_MIN, TIME_MAX))

    # start_position vs veto counts
    start_vs_veto_count_all = _cfg(ROOT.TH2F("start_vs_veto_count_all",
        "Start position vs veto count (All);start position [cm];veto count",
        START_NBINS, START_MIN, START_MAX, COUNT_ALL_NBINS, COUNT_ALL_MIN, COUNT_ALL_MAX))

    start_vs_veto_count_s1 = _cfg(ROOT.TH2F("start_vs_veto_count_s1",
        "Start position vs veto count (Station 1);start position [cm];veto count",
        START_NBINS, START_MIN, START_MAX, COUNT_S_NBINS, COUNT_S_MIN, COUNT_S_MAX))
    start_vs_veto_count_s2 = _cfg(ROOT.TH2F("start_vs_veto_count_s2",
        "Start position vs veto count (Station 2);start position [cm];veto count",
        START_NBINS, START_MIN, START_MAX, COUNT_S_NBINS, COUNT_S_MIN, COUNT_S_MAX))
    start_vs_veto_count_s3 = _cfg(ROOT.TH2F("start_vs_veto_count_s3",
        "Start position vs veto count (Station 3);start position [cm];veto count",
        START_NBINS, START_MIN, START_MAX, COUNT_S_NBINS, COUNT_S_MIN, COUNT_S_MAX))

    # hitTime vs veto counts (x=time, y=count)
    hitTime_vs_veto_count_all = _cfg(ROOT.TH2F("hitTime_vs_veto_count_all",
        "Hit time vs veto count (All);time [ns];veto count",
        TIME_NBINS, TIME_MIN, TIME_MAX, COUNT_ALL_NBINS, COUNT_ALL_MIN, COUNT_ALL_MAX))

    hitTime_vs_veto_count_s1 = _cfg(ROOT.TH2F("hitTime_vs_veto_count_s1",
        "Hit time vs veto count (Station 1);time [ns];veto count",
        TIME_NBINS, TIME_MIN, TIME_MAX, COUNT_S_NBINS, COUNT_S_MIN, COUNT_S_MAX))
    hitTime_vs_veto_count_s2 = _cfg(ROOT.TH2F("hitTime_vs_veto_count_s2",
        "Hit time vs veto count (Station 2);time [ns];veto count",
        TIME_NBINS, TIME_MIN, TIME_MAX, COUNT_S_NBINS, COUNT_S_MIN, COUNT_S_MAX))
    hitTime_vs_veto_count_s3 = _cfg(ROOT.TH2F("hitTime_vs_veto_count_s3",
        "Hit time vs veto count (Station 3);time [ns];veto count",
        TIME_NBINS, TIME_MIN, TIME_MAX, COUNT_S_NBINS, COUNT_S_MIN, COUNT_S_MAX))

    # hitTime vs vetoStation (x=station index, y=time)
    hitTime_vs_vetoStation = _cfg(ROOT.TH2F("hitTime_vs_vetoStation",
        "Hit time vs veto station;veto station;time [ns]",
        len(STATIONS), STATION_MIN, STATION_MAX, TIME_NBINS, TIME_MIN, TIME_MAX))
    for i, s in enumerate(STATIONS, start=1):
        hitTime_vs_vetoStation.GetXaxis().SetBinLabel(i, f"{s}")

    # count vs vetoStation (x=station, y=count)
    count_vs_vetoStation = _cfg(ROOT.TH2F("count_vs_vetoStation",
        "Veto count vs station;veto station;veto count",
        len(STATIONS), STATION_MIN, STATION_MAX, COUNT_ALL_NBINS, COUNT_ALL_MIN, COUNT_ALL_MAX))
    for i, s in enumerate(STATIONS, start=1):
        count_vs_vetoStation.GetXaxis().SetBinLabel(i, f"{s}")

    # earliest_hitTime vs vetoStation (x=station, y=earliest time)
    earlist_hitTime_vs_vetoStation = _cfg(ROOT.TH2F("earlist_hitTime_vs_vetoStation",
        "Earliest time vs veto station;veto station;earliest time [ns]",
        len(STATIONS), STATION_MIN, STATION_MAX, TIME_NBINS, TIME_MIN, TIME_MAX))
    for i, s in enumerate(STATIONS, start=1):
        earlist_hitTime_vs_vetoStation.GetXaxis().SetBinLabel(i, f"{s}")


    # 1D per-station
    h_earliest_s = {1: earliest_hitTime_s1, 2: earliest_hitTime_s2, 3: earliest_hitTime_s3}
    h_latest_s   = {1: latest_hitTime_s1,   2: latest_hitTime_s2,   3: latest_hitTime_s3}
    h_vetocnt_s  = {1: veto_count_s1,       2: veto_count_s2,       3: veto_count_s3}

    # 2D per-station
    h_start_vs_earlist_s   = {1: start_vs_earlist_hitTime_s1, 2: start_vs_earlist_hitTime_s2, 3: start_vs_earlist_hitTime_s3}
    h_start_vs_vetocnt_s   = {1: start_vs_veto_count_s1,      2: start_vs_veto_count_s2,      3: start_vs_veto_count_s3}
    h_hitTime_vs_vetocnt_s = {1: hitTime_vs_veto_count_s1,    2: hitTime_vs_veto_count_s2,    3: hitTime_vs_veto_count_s3}

    # ------------------------------------------------------------------
    n_entries = digi_chain.GetEntries()

    # -------- Event loop --------
    for i in tqdm(range(n_entries), desc="events"):
        # if i > 500:
        #     break
        digi_chain.GetEntry(i)

        # Keep only MuFilter system==1 and valid
        hits = [h for h in digi_chain.Digi_MuFilterHits if h.isValid() and h.GetSystem() == 1]
        if not hits:
            continue

        # SciFi multiplicity gate
        try:
            n_scifi = digi_chain.Digi_ScifiHits.GetEntriesFast()
        except AttributeError:
            print("error when doing digi_chain.Digi_ScifiHits.GetEntriesFast()")
            n_scifi = len(digi_chain.Digi_ScifiHits)
        if n_scifi <= count_scifi_threshold:
            continue

        # ==== per-event accumulators ====
        # Use MCTrack[1] if it exists, else sentinel -999
        start_position = (
            digi_chain.MCTrack[1].GetStartZ()
            if hasattr(digi_chain, "MCTrack") and digi_chain.MCTrack and digi_chain.MCTrack.GetEntries() > 1
            else -999.0
        )

        station_counts = {1: 0, 2: 0, 3: 0}
        hit_times_by_station = {1: [], 2: [], 3: []}
        all_hit_times = []

        # --- per-event temporary 1D time-count hists ---
        tmp_time_counts_all = ROOT.TH1F("tmp_time_counts_all", "", TIME_NBINS, TIME_MIN, TIME_MAX); tmp_time_counts_all.SetDirectory(0)
        tmp_time_counts_s = {
            1: ROOT.TH1F("tmp_time_counts_s1", "", TIME_NBINS, TIME_MIN, TIME_MAX),
            2: ROOT.TH1F("tmp_time_counts_s2", "", TIME_NBINS, TIME_MIN, TIME_MAX),
            3: ROOT.TH1F("tmp_time_counts_s3", "", TIME_NBINS, TIME_MIN, TIME_MAX),
        }
        for _h in tmp_time_counts_s.values():
            _h.SetDirectory(0)

        # ---- first pass: collect per-hit data ----
        for aHit in hits:
            t = aHit.GetTime()
            detID = aHit.GetDetectorID()
            s = ((detID // 1000) % 10) + 1   # your station decode {1,2,3}

            all_hit_times.append(t)
            h_hitTime.Fill(t)

            tmp_time_counts_all.Fill(t)
            if s in STATIONS:
                station_counts[s] += 1
                hit_times_by_station[s].append(t)
                tmp_time_counts_s[s].Fill(t)

        # ---- per-event scalars ----
        total_count = len(all_hit_times)
        veto_count_all.Fill(total_count)
        for s in STATIONS:
            h_vetocnt_s[s].Fill(station_counts[s])
            count_vs_vetoStation.Fill(s, station_counts[s])

        if total_count > 0:
            t_earliest_all = min(all_hit_times)
            t_latest_all   = max(all_hit_times)

            earlist_hitTime_all.Fill(t_earliest_all)
            latest_hitTime_all.Fill(t_latest_all)

            # start position correlations (all) — require finite and not sentinel
            if math.isfinite(start_position) and start_position != -999.0:
                start_vs_earlist_hitTime_all.Fill(start_position, t_earliest_all)
                start_vs_veto_count_all.Fill(start_position, total_count)

        # per station earliest/latest + start correlations
        for s in STATIONS:
            times = hit_times_by_station[s]
            if not times:
                continue
            tmin, tmax = min(times), max(times)
            h_earliest_s[s].Fill(tmin)
            h_latest_s[s].Fill(tmax)
            if math.isfinite(start_position) and start_position != -999.0:
                h_start_vs_earlist_s[s].Fill(start_position, tmin)
                h_start_vs_vetocnt_s[s].Fill(start_position, station_counts[s])
            earlist_hitTime_vs_vetoStation.Fill(tmin, s)
            

        # ---- project per-event 1D counts into 2D hists ----
        # For each time bin, take the count in that bin and fill 2D with (time_bin_center, count)
        xa_all = tmp_time_counts_all.GetXaxis()
        for ib in range(1, TIME_NBINS + 1):
            x = xa_all.GetBinCenter(ib)
            c = int(tmp_time_counts_all.GetBinContent(ib))
            if c > 0:
                # hitTime vs veto_count (ALL)
                hitTime_vs_veto_count_all.Fill(x, c)

        # Per-station projections (and hitTime vs vetoStation with weight = count)
        for s in STATIONS:
            xa_s = tmp_time_counts_s[s].GetXaxis()
            for ib in range(1, TIME_NBINS + 1):
                x = xa_s.GetBinCenter(ib)
                c = int(tmp_time_counts_s[s].GetBinContent(ib))
                if c > 0:
                    h_hitTime_vs_vetocnt_s[s].Fill(x, c)
                    hitTime_vs_vetoStation.Fill(s, x, c)

        # ---- clean up tmp hists to avoid memory growth ----
        tmp_time_counts_all.Delete()
        for _h in tmp_time_counts_s.values():
            _h.Delete()

    # Open a ROOT file for writing
    out_file = f"{out_dir}/{data_type}_scifi_gt_{count_scifi_threshold}_hist.root"
    output_file = ROOT.TFile(out_file, "RECREATE")

    # Write 1D histograms
    veto_count_all.Write()
    veto_count_s1.Write()
    veto_count_s2.Write()
    veto_count_s3.Write()

    h_hitTime.Write()

    earlist_hitTime_all.Write()
    earliest_hitTime_s1.Write()
    earliest_hitTime_s2.Write()
    earliest_hitTime_s3.Write()

    latest_hitTime_all.Write()
    latest_hitTime_s1.Write()
    latest_hitTime_s2.Write()
    latest_hitTime_s3.Write()

    # Write 2D histograms
    start_vs_earlist_hitTime_all.Write()
    start_vs_earlist_hitTime_s1.Write()
    start_vs_earlist_hitTime_s2.Write()
    start_vs_earlist_hitTime_s3.Write()

    start_vs_veto_count_all.Write()
    start_vs_veto_count_s1.Write()
    start_vs_veto_count_s2.Write()
    start_vs_veto_count_s3.Write()

    hitTime_vs_veto_count_all.Write()
    hitTime_vs_veto_count_s1.Write()
    hitTime_vs_veto_count_s2.Write()
    hitTime_vs_veto_count_s3.Write()

    hitTime_vs_vetoStation.Write()
    count_vs_vetoStation.Write()
    earlist_hitTime_vs_vetoStation.Write()

    # Finalize
    output_file.Close()
    print(f"✅ Histograms saved to {out_file}")

  
def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo
    
    


def main():
    digi_chain, new_digi_chain = load_chains_from_neutrinoMC("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv", max_rows=1)
    digi_chain_realData = load_chains_from_metadata("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2024_skim_runs_metadata.csv", max_rows=20)
    digi_chain_ve = load_chains_from_metadata("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_2024_ve_metadata.csv", max_rows=1000)

    
    geo_path = '/eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root'
    snd_geo = setup_geometry(geo_path)
       
    count_scifi_threshold = 200
    hists_ve = fill_vetoCount_vetoTime(digi_chain_ve, snd_geo, "ve", count_scifi_threshold)
    hists_realData = fill_vetoCount_vetoTime(digi_chain_realData, snd_geo, "readData", count_scifi_threshold)
    
    #out_dir = f'veto_hist_scifi_gt_{count_scifi_threshold}'
    #plot_ve_realData_hists(hists_realData, hists_ve, out_dir)
    
    
    
    

if __name__ == "__main__":
    main()