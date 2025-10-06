import ROOT
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import SndlhcGeo
import numpy as np
import re
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

def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo

def check_1ns_events(digi_chain, count_scifi_threshold):
    os.makedirs("./check_1ns", exist_ok=True)
    output_file = "./check_1ns/1ns_events.txt"

    with open(output_file, "w") as f_out:
        n_entries = digi_chain.GetEntries()

        current_file = None
        local_event_index = 0  # Index within current file

        for i in tqdm(range(n_entries), desc="events"):
            digi_chain.GetEntry(i)

            file_name = digi_chain.GetFile().GetName()
            runId = digi_chain.EventHeader.GetRunId()

            # Reset local event index if file changes
            if file_name != current_file:
                current_file = file_name
                local_event_index = 0  # Reset for new file

            try:
                eventId = digi_chain.EventHeader.GetEventNumber()
            except Exception:
                eventId = digi_chain.EventHeader.GetMCEntryNumber()

            # Keep only MuFilter system==1 and valid
            hits = [h for h in digi_chain.Digi_MuFilterHits if h.isValid() and h.GetSystem() == 1]
            if not hits:
                local_event_index += 1
                continue

            # SciFi multiplicity gate
            try:
                n_scifi = digi_chain.Digi_ScifiHits.GetEntriesFast()
            except AttributeError:
                f_out.write("error when doing digi_chain.Digi_ScifiHits.GetEntriesFast()\n")
                n_scifi = len(digi_chain.Digi_ScifiHits)
            if n_scifi <= count_scifi_threshold:
                local_event_index += 1
                continue

            all_hit_times = [h.GetTime() for h in hits]
            t_earliest_all = min(all_hit_times)

            if t_earliest_all < 1:
                f_out.write(
                    f"file: {file_name}, runId: {runId}, local_event: {local_event_index}, "
                    f"eventId: {eventId}, earliest veto hit time: {t_earliest_all:.2f} [ns]\n"
                )

            local_event_index += 1  # Advance index in any case

        

def main():
    digi_chain, new_digi_chain = load_chains_from_neutrinoMC("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv", max_rows=1)
    digi_chain_realData = load_chains_from_metadata("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2024_skim_runs_metadata.csv", max_rows=1)
    digi_chain_ve = load_chains_from_metadata("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_2024_ve_metadata.csv", max_rows=100)
   
    
    geo_path = '/eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root'
    snd_geo = setup_geometry(geo_path)
    #fill_origin_and_plot(digi_chain)
    
    count_scifi_threshold = 0
    
    
    check_1ns_events(digi_chain_ve, count_scifi_threshold)
    
    
    
    

if __name__ == "__main__":
    main()