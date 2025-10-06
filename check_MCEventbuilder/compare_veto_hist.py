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

def group_pdg_event_stats(pdg_event_stats):
    """
    Groups raw per-PDG event statistics into categories (ve, vm, vt, NC, other),
    and computes derived statistics like without_hitTime and hitTime_le_25.

    Parameters:
        pdg_event_stats (dict): Mapping from PDG code → {"total", "with_hitTime", "hitTime_gt_25"}

    Returns:
        pd.DataFrame: Aggregated and annotated stats per particle type
    """

    # Map PDG codes to categories
    pdg_label_map = {
        12: "ve",
        14: "vm",
        16: "vt",
        112: "NC",
        114: "NC",
        116: "NC",
    }

    # Group counts into labels
    grouped_stats = defaultdict(lambda: {"total": 0, "with_hitTime": 0, "hitTime_gt_25": 0})
    for pdg, stats in pdg_event_stats.items():
        label = pdg_label_map.get(abs(pdg), "other")  # Use absolute value to ignore sign
        for key in stats:
            grouped_stats[label][key] += stats[key]

    # Convert to DataFrame
    df_stats = pd.DataFrame.from_dict(grouped_stats, orient='index')
    
    totals = df_stats.sum(numeric_only=True)
    df_stats.loc["total"] = totals

    # Derived absolute counts
    df_stats["without_hitTime"] = df_stats["total"] - df_stats["with_hitTime"]
    df_stats["without_hitTime_or_gt_25"] = df_stats["without_hitTime"] + df_stats["hitTime_gt_25"]

    # Clamp negatives to zero (just in case)
    df_stats["without_hitTime"] = df_stats["without_hitTime"]

    # Derived fractions
    df_stats["frac_without_hitTime"] = df_stats["without_hitTime"] / df_stats["total"]
    df_stats["frac_hitTime_gt_25"] = df_stats["hitTime_gt_25"] / df_stats["total"]
    
    df_stats["frac_without_hitTime_or_gt_25"] = df_stats["without_hitTime_or_gt_25"] / df_stats["total"]

    # Replace any NaNs (from division by zero) with 0
    df_stats = df_stats.fillna(0)
    
    

    return df_stats

def plot_event_stats_table(df_stats, name_suffix):
    # Column order and rename mapping
    cols = [
        "total",
        "without_hitTime",
        "hitTime_gt_25",
        "without_hitTime_or_gt_25",
        "frac_without_hitTime",
        "frac_hitTime_gt_25",
        "frac_without_hitTime_or_gt_25"
    ]
    rename_map = {
        "total": "Total",
        "without_hitTime": "No\nhitTime",
        "hitTime_gt_25": "hitTime >\n25 ns",
        "without_hitTime_or_gt_25": "No hitTime\nor hitTime > 25 ns",
        "frac_without_hitTime": "Fraction\nNo hitTime",
        "frac_hitTime_gt_25": "Fraction\nhitTime > 25 ns",
        "frac_without_hitTime_or_gt_25": "Fraction\nNo hitTime or > 25 ns"
    }

    # Validate columns
    missing_cols = [col for col in cols if col not in df_stats.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in df_stats: {missing_cols}")

    # Prepare DataFrame
    df = df_stats.copy()
    row_order = ["vm", "ve", "vt", "NC", "total"]
    df = df.reindex(row_order)
    df = df[cols].round(4)
    df.columns = [rename_map[col] for col in df.columns]

    os.makedirs("./plots", exist_ok=True)
    
    

    # Adjust figure size
    fig, ax = plt.subplots(figsize=(16, 0.6 * len(df)))  # Wider for clarity
    ax.axis("off")
    fig.suptitle(f"Veto Hit Cut Efficiency: {name_suffix}", fontsize=14, y=1.02)

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     loc='center',
                     cellLoc='center',
                     rowLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.4, 2)
    n_cols = len(df.columns)
    for col in range(n_cols):
        cell = table[0, col]
        cell.set_height(cell.get_height() * 2.0)

    filename = f"./plots/event_stats_table_{name_suffix}.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved high-res table to: {filename}")
    return filename

def fill_hist_oldDigi(digi_chain, nbins=210, tmin=-10.0, tmax=200.0,):
    """
    Build & fill time histograms from a digi TChain.

    Histograms:
      - neutrino_hitTime : all neutrino events
      - ve_hitTime       : νe CC
      - vm_hitTime       : νμ CC
      - vt_hitTime       : ντ CC
      - NC_hitTime       : neutral current (PDG ±(12/14/16) +/-100)

    Args:
        digi_chain (ROOT.TChain): chain with Digi_MuFilterHits and MCTrack
        nbins, tmin, tmax: binning for hit-time histograms

    Returns:
        dict[str, ROOT.TH1F]
    """
    # Create histograms
    hists = {
        "neutrino_hitTime": ROOT.TH1F("neutrino_hitTime", "All neutrino events;min veto hit time [ns];events", nbins, tmin, tmax),
        "ve_hitTime":       ROOT.TH1F("ve_hitTime",       "nu_e CC;min veto hit time [ns];events",            nbins, tmin, tmax),
        "vm_hitTime":       ROOT.TH1F("vm_hitTime",       "nu_mu CC;min veto hit time [ns];events",            nbins, tmin, tmax),
        "vt_hitTime":       ROOT.TH1F("vt_hitTime",       "nu_tau CC;min veto hit time [ns];events",           nbins, tmin, tmax),
        "NC_hitTime":       ROOT.TH1F("NC_hitTime",       "Neutral Current;min veto hit time [ns];events",     nbins, tmin, tmax),
    }
    for h in hists.values():
        h.Sumw2()

    # Quick branch checks
    if not hasattr(digi_chain, "Digi_MuFilterHits"):
        print("Warning: Digi_MuFilterHits branch not found.")
    if not hasattr(digi_chain, "MCTrack"):
        print("Warning: MCTrack branch not found.")

    neutrino_pdgs = {12, -12, 14, -14, 16, -16}


    n_entries = digi_chain.GetEntries()
    eventId_counts = defaultdict(int)
    seen_eventIds = set()
    pdg_event_stats = defaultdict(lambda: {"total": 0, "with_hitTime": 0, "hitTime_gt_25": 0})
    eventId_to_pdg = dict()
    for i in tqdm(range(n_entries)):
        
        digi_chain.GetEntry(i)
        
        try:
            eventId = digi_chain.EventHeader.GetEventNumber()
        except Exception:
            try:
                eventId = digi_chain.EventHeader.GetMCEntryNumber()
            except Exception:
                try:
                    eventId = digi_chain.MCEventHeader.GetEventNumber()
                except Exception:
                    try:
                        eventId = digi_chain.MCEventHeader.GetMCEntryNumber()
                    except Exception:
                        eventId = None
            
            MCEventHeader

        runId = digi_chain.EventHeader.GetRunId()
        
        unique_id = f"{int(runId)}_{int(eventId)}"
        eventId_counts[unique_id] += 1
        
        seen_eventIds.add(unique_id)
        # --- classify interaction using first two MC tracks ---
        pdg0 = None
        pdg1 = None
        if hasattr(digi_chain, "MCTrack"):
            try:
                #print(dir(digi_chain.MCTrack[0]))
                if len(digi_chain.MCTrack) > 0:
                    pdg0 = digi_chain.MCTrack[0].GetPdgCode()
                if len(digi_chain.MCTrack) > 1:
                    pdg1 = digi_chain.MCTrack[1].GetPdgCode()
                if len(digi_chain.MCTrack) > 2:
                    pdg2 = digi_chain.MCTrack[2].GetPdgCode()
            except Exception:
                pass
        # If the two leading tracks have the same PDG and it's a neutral current neutrino, mark NC by ±100 shift.
        if (pdg0 == pdg1) and (pdg0 in neutrino_pdgs):
            pdg_code_event = pdg0 - 100 if pdg0 < 0 else pdg0 + 100
            #print(f"NC event: {eventId}")
        else:
            pdg_code_event = pdg0  # CC flavor: 12/14/16 
        #if (abs(pdg_code_event) in (112, 114, 116)):
        
        eventId_to_pdg[unique_id] = pdg_code_event
        #n_scifi = digi_chain.Digi_ScifiHits.GetEntries()
        #print(f"PDG values ->pdg0: {pdg0}, pdg1: {pdg1}, pdg2: {pdg2}, with scifi count: {n_scifi}")
        if runId == 1656430568:
            print(f"event: runId={runId}, eventId={eventId}, key={unique_id}")
            
        if  pdg_code_event==None:
            n_scifi = digi_chain.Digi_ScifiHits.GetEntries()
            print(f"Didn't find mc track in event{unique_id}, with scifi count: {n_scifi}")
            print()
            continue
        
        
        pdg_event_stats[abs(pdg_code_event)]["total"] += 1
        # --- find min veto hit time for this event ---
        hitTime_min = None
        if hasattr(digi_chain, "Digi_MuFilterHits"):
            for aHit in digi_chain.Digi_MuFilterHits:
                try:
                    if aHit.GetSystem() == 1:  # Veto
                        t = aHit.GetTime()
                        if hitTime_min is None or t < hitTime_min:
                            hitTime_min = t
                except Exception:
                    # In case aHit is a proxy without the expected methods
                    continue

        # If no veto hits, you may choose to skip filling. Here we skip.
        if hitTime_min is None:
            continue
        
        pdg_event_stats[abs(pdg_code_event)]["with_hitTime"] += 1
        if (hitTime_min>25):
            pdg_event_stats[abs(pdg_code_event)]["hitTime_gt_25"] += 1
        # --- fill histograms ---
        # Fill all-neutrino hist always
        hists["neutrino_hitTime"].Fill(hitTime_min)

        
        apdg = abs(pdg_code_event)
        #if (apdg==0):
            #print(f"PDG values -> apdg: {apdg}, pdg0: {pdg0}, pdg1: {pdg1}, pdg2: {pdg2}")

        # --- fill flavor/NC hists ---
        if apdg in (112, 114, 116):
            hists["NC_hitTime"].Fill(hitTime_min)
        elif apdg == 12:
            hists["ve_hitTime"].Fill(hitTime_min)
        elif apdg == 14:
            hists["vm_hitTime"].Fill(hitTime_min)
        elif apdg == 16:
            hists["vt_hitTime"].Fill(hitTime_min)
        # else: not recognized → ignore

    print("Histogram entries:")
    for name, hist in hists.items():
        print(f"{name:<20} : {int(hist.GetEntries())}")
        
    duplicates = {eid: count for eid, count in eventId_counts.items() if count > 1}

    # if duplicates:
    #     print("Duplicate event IDs found:")
    #     for eid, count in sorted(duplicates.items()):
    #         print(f"  EventID {eid} appears {count} times")
    # else:
    #     print("No duplicate event IDs found.")
    
    df = group_pdg_event_stats(pdg_event_stats)
    
    return hists, df, eventId_to_pdg



def fill_hist_MCEventBuilder(digi_chain, eventId_to_pdg, nbins=210, tmin=-10.0, tmax=200.0):
    """
    Build & fill time histograms from a digi TChain.

    Histograms:
      - neutrino_hitTime : all neutrino events
      - ve_hitTime       : νe CC
      - vm_hitTime       : νμ CC
      - vt_hitTime       : ντ CC
      - NC_hitTime       : neutral current (PDG ±(12/14/16) +/-100)

    Args:
        digi_chain (ROOT.TChain): chain with Digi_MuFilterHits and MCTrack
        nbins, tmin, tmax: binning for hit-time histograms

    Returns:
        dict[str, ROOT.TH1F]
    """
    # Create histograms
    hists = {
        "neutrino_hitTime": ROOT.TH1F("neutrino_hitTime", "All neutrino events;min veto hit time [ns];events", nbins, tmin, tmax),
        "ve_hitTime":       ROOT.TH1F("ve_hitTime",       "nu_e CC;min veto hit time [ns];events",            nbins, tmin, tmax),
        "vm_hitTime":       ROOT.TH1F("vm_hitTime",       "nu_mu CC;min veto hit time [ns];events",            nbins, tmin, tmax),
        "vt_hitTime":       ROOT.TH1F("vt_hitTime",       "nu_tau CC;min veto hit time [ns];events",           nbins, tmin, tmax),
        "NC_hitTime":       ROOT.TH1F("NC_hitTime",       "Neutral Current;min veto hit time [ns];events",     nbins, tmin, tmax),
    }
    for h in hists.values():
        h.Sumw2()

    # Quick branch checks
    if not hasattr(digi_chain, "Digi_MuFilterHits"):
        print("Warning: Digi_MuFilterHits branch not found.")
    if not hasattr(digi_chain, "MCTrack"):
        print("Warning: MCTrack branch not found.")

    neutrino_pdgs = {12, -12, 14, -14, 16, -16}
    pdg_label_map = {
        12: "ve",
        14: "vm",
        16: "vt",
        112: "NC",
        114: "NC",
        116: "NC",
    }

    n_entries = digi_chain.GetEntries()
    print(n_entries)
    eventId_counts = defaultdict(int)
    seen_eventIds = set()
    pdg_event_stats = defaultdict(lambda: {"total": 0, "with_hitTime": 0, "hitTime_gt_25": 0})
    new_eventId_to_pdg = dict()
    for i in tqdm(range(n_entries)):
        
        digi_chain.GetEntry(i)
        #print(dir(digi_chain.MCEventHeader))
        
        try:
            eventId = digi_chain.EventHeader.GetEventNumber()
            runId = digi_chain.EventHeader.GetRunId()
        except Exception:
            try:
                eventId = digi_chain.EventHeader.GetMCEntryNumber()
                runId = digi_chain.EventHeader.GetRunId()
            except Exception:
                try:
                    eventId = digi_chain.MCEventHeader.GetEventID()
                    runId = digi_chain.MCEventHeader.GetRunID()
                except Exception:
                        eventId = None
                        runId = None

        
        
        unique_id = f"{int(runId)}_{int(eventId)}"
        eventId_counts[unique_id] += 1
        if unique_id not in seen_eventIds:
            seen_eventIds.add(unique_id)
            try:
                correct_pdg = eventId_to_pdg[unique_id]
            except KeyError:
                print(f"[Warning] No PDG info found for event: runId={runId}, eventId={eventId}, key={unique_id}")
                continue
            new_eventId_to_pdg[unique_id] = correct_pdg
            # --- classify interaction using first two MC tracks ---
            pdg0 = None
            pdg1 = None
            if hasattr(digi_chain, "MCTrack"):
                try:
                    #print(dir(digi_chain.MCTrack[0]))
                    if len(digi_chain.MCTrack) > 0:
                        pdg0 = digi_chain.MCTrack[0].GetPdgCode()
                    if len(digi_chain.MCTrack) > 1:
                        pdg1 = digi_chain.MCTrack[1].GetPdgCode()
                    if len(digi_chain.MCTrack) > 2:
                        pdg2 = digi_chain.MCTrack[2].GetPdgCode()
                except Exception:
                    pass
            # If the two leading tracks have the same PDG and it's a neutral current neutrino, mark NC by ±100 shift.
            if ((pdg1==0) and (pdg0 in neutrino_pdgs)) or (pdg0 == pdg1) and (pdg0 in neutrino_pdgs):
                pdg_code_event = pdg0 - 100 if pdg0 < 0 else pdg0 + 100
                #print(f"NC event: {eventId}")
            else:
                pdg_code_event = pdg0  # CC flavor: 12/14/16 
            #if (abs(pdg_code_event) in (112, 114, 116)):
            
            
            pdg_code_event = correct_pdg
            
            if  pdg_code_event==None:
                n_scifi = digi_chain.Digi_ScifiHits.GetEntries()
                print(f"Didn't find mc track in event{unique_id}, with scifi count: {n_scifi}")
                print()
                continue
            
            #if correct_pdg != pdg_code_event:
            #     n_scifi = digi_chain.Digi_ScifiHits.GetEntries()
            #     print(f"[Mismatch] Correct PDG: {correct_pdg}, Assigned PDG: {pdg_code_event}, SciFi hits: {n_scifi}")
            #     n_tracks = digi_chain.MCTrack.GetEntries()
            #     print(f"    Total MCTracks: {n_tracks}")
            #     for i in range(min(10, n_tracks)):
            #         pdg = digi_chain.MCTrack[i].GetPdgCode()
            #         motherId = digi_chain.MCTrack[i].GetMotherId()
            #         print(f"    MCTrack[{i}] PDG: {pdg}, motherId: {motherId}")
                    
                
                                    
                
            
            
            
            pdg_event_stats[abs(pdg_code_event)]["total"] += 1
            # --- find min veto hit time for this event ---
            hitTime_min = None
            if hasattr(digi_chain, "Digi_MuFilterHits"):
                for aHit in digi_chain.Digi_MuFilterHits:
                    try:
                        if aHit.GetSystem() == 1:  # Veto
                            t = aHit.GetTime()
                            if hitTime_min is None or t < hitTime_min:
                                hitTime_min = t
                    except Exception:
                        # In case aHit is a proxy without the expected methods
                        continue

            # If no veto hits, you may choose to skip filling. Here we skip.
            if hitTime_min is None:
                continue
            
            pdg_event_stats[abs(pdg_code_event)]["with_hitTime"] += 1
            if (hitTime_min>25):
                pdg_event_stats[abs(pdg_code_event)]["hitTime_gt_25"] += 1
            # --- fill histograms ---
            # Fill all-neutrino hist always
            hists["neutrino_hitTime"].Fill(hitTime_min)

            
            apdg = abs(pdg_code_event)
            #if (apdg==0):
                #print(f"PDG values -> apdg: {apdg}, pdg0: {pdg0}, pdg1: {pdg1}, pdg2: {pdg2}")

            # --- fill flavor/NC hists ---
            if apdg in (112, 114, 116):
                hists["NC_hitTime"].Fill(hitTime_min)
            elif apdg == 12:
                hists["ve_hitTime"].Fill(hitTime_min)
            elif apdg == 14:
                hists["vm_hitTime"].Fill(hitTime_min)
            elif apdg == 16:
                hists["vt_hitTime"].Fill(hitTime_min)
            # else: not recognized → ignore

    print("Histogram entries:")
    for name, hist in hists.items():
        print(f"{name:<20} : {int(hist.GetEntries())}")
        
    duplicates = {eid: count for eid, count in eventId_counts.items() if count > 1}

    # if duplicates:
    #     print("Duplicate event IDs found:")
    #     for eid, count in sorted(duplicates.items()):
    #         print(f"  EventID {eid} appears {count} times")
    # else:
    #     print("No duplicate event IDs found.")
    
    df = group_pdg_event_stats(pdg_event_stats)
    
    return hists, df, new_eventId_to_pdg


def plot_hist(digi_hists, new_digi_hists, outdir="./plots", normalize=False):
    # --- important for batch/headless nodes ---

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    def save_canvas(c, name):
        # ensure canvas is fully built
        c.Modified(); c.Update()
        pdf = os.path.join(outdir, f"{name}.pdf")
        try:
            c.SaveAs(pdf)
        except Exception as e:
            print(f"[warn] SaveAs PDF failed for {name}: {e}")

    def prep_hist(h, color, marker=20, width=2):
        h.SetDirectory(0)  # decouple from any file/dir
        h.SetLineColor(color)
        h.SetMarkerColor(color)
        h.SetMarkerStyle(marker)
        h.SetLineWidth(width)
        h.Sumw2()

    colors = {
        "ve": ROOT.kAzure + 1,
        "vm": ROOT.kRed + 1,
        "vt": ROOT.kGreen + 2,
        "NC": ROOT.kMagenta + 2,
        "neutrino": ROOT.kBlack
    }
    markers = {"ve": 20, "vm": 21, "vt": 22, "NC": 23}

    keep_alive = []  # keep refs to hist/legend/stack so Python GC won't kill them

    # -------- Plot 1: digi stack (vm, ve, vt, NC) --------
    c1 = ROOT.TCanvas("c1", "digi stack", 800, 600)
    stack1 = ROOT.THStack("stack1", "Digi: Neutrino types;min veto hit time [ns];Events")
    keep_alive.append(stack1)

    stack_order = [("vm", "nu_mu CC"), ("ve", "nu_e CC"), ("vt", "nu_tau CC"), ("NC", "Neutral Current")]
    digi_stack_parts = {}

    for key, title in stack_order:
        h = digi_hists[f"{key}_hitTime"].Clone(f"{key}_stack1")
        h.SetDirectory(0)
        h.SetFillColor(colors[key])
        h.SetLineColor(colors[key])
        h.SetLineWidth(1)
        stack1.Add(h)
        digi_stack_parts[key] = h
        keep_alive.append(h)

    stack1.Draw("HIST")
    stack1.GetYaxis().SetTitleOffset(1.2)

    leg1 = ROOT.TLegend(0.65, 0.68, 0.88, 0.88)
    leg1.SetBorderSize(0)
    leg1.SetFillStyle(0)
    for key, title in stack_order:
        leg1.AddEntry(digi_stack_parts[key], title, "f")
    leg1.Draw()
    keep_alive.append(leg1)

    save_canvas(c1, "1_digi_stack")
    
    # -------- Plot 2: newDigi stack (vm, ve, vt, NC) --------
    c2 = ROOT.TCanvas("c2", "newDigi stack", 800, 600)
    stack2 = ROOT.THStack("stack2", "NewDigi: Neutrino types;min veto hit time [ns];Events")
    keep_alive.append(stack2)

    new_stack_parts = {}

    for key, title in stack_order:
        h = new_digi_hists[f"{key}_hitTime"].Clone(f"{key}_stack2")
        h.SetDirectory(0)
        h.SetFillColor(colors[key])
        h.SetLineColor(colors[key])
        h.SetLineWidth(1)
        stack2.Add(h)
        new_stack_parts[key] = h
        keep_alive.append(h)

    stack2.Draw("HIST")
    stack2.GetYaxis().SetTitleOffset(1.2)

    leg2 = ROOT.TLegend(0.65, 0.68, 0.88, 0.88)
    leg2.SetBorderSize(0)
    leg2.SetFillStyle(0)
    for key, title in stack_order:
        leg2.AddEntry(new_stack_parts[key], title, "f")
    leg2.Draw()
    keep_alive.append(leg2)

    save_canvas(c2, "2_newDigi_stack")


    # -------- helper: two-hist overlay with errors --------
    def overlay_two(idx, title, key):
        c = ROOT.TCanvas(f"c{idx}", f"{key} compare", 800, 600)
        h_d = digi_hists[f"{key}_hitTime"].Clone(f"{key}_digi_cmp")
        h_n = new_digi_hists[f"{key}_hitTime"].Clone(f"{key}_new_cmp")
        h_d.SetDirectory(0); h_n.SetDirectory(0)
        h_d.Sumw2(); h_n.Sumw2()
        if normalize:
            if h_d.Integral() > 0: h_d.Scale(1.0 / h_d.Integral())
            if h_n.Integral() > 0: h_n.Scale(1.0 / h_n.Integral())
        prep_hist(h_d, ROOT.kBlue, marker=20)
        prep_hist(h_n, ROOT.kRed,  marker=21)
        h_d.SetTitle(f"{title};min veto hit time [ns];Events")
        h_d.Draw("E1")
        h_n.Draw("E1 SAME")
        leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.AddEntry(h_d, "Digi", "lep")
        leg.AddEntry(h_n, "NewDigi", "lep")
        leg.Draw()
        keep_alive.extend([h_d, h_n, leg])
        save_canvas(c, f"{idx}_{key}_compare")

    # -------- Plots 3–7 --------
    overlay_two(3, "All Neutrinos", "neutrino")
    overlay_two(4, "nu_e CC",       "ve")
    overlay_two(5, "nu_tau CC",     "vt")
    overlay_two(6, "nu_mu CC",      "vm")
    overlay_two(7, "Neutral Current","NC")

    # -------- Plot 8: within digi overlay (errors) --------
    c8 = ROOT.TCanvas("c8", "digi overlap", 800, 600)
    leg8 = ROOT.TLegend(0.60, 0.68, 0.88, 0.88); leg8.SetBorderSize(0)
    keep_alive.append(leg8)
    first = True
    for key in ["vm", "ve",  "vt", "NC"]:
        h = digi_hists[f"{key}_hitTime"].Clone(f"{key}_digi_ol")
        h.SetDirectory(0)
        h.Sumw2()
        if normalize and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        prep_hist(h, colors[key], marker=markers[key])
        h.SetTitle("Digi: overlap of flavors;min veto hit time [ns];Events")
        h.Draw("E1" if first else "E1 SAME")
        first = False
        leg8.AddEntry(h, key, "lep")
        keep_alive.append(h)
    leg8.Draw()
    save_canvas(c8, "8_digi_overlap")

    # -------- Plot 9: within newDigi overlay (errors) --------
    c9 = ROOT.TCanvas("c9", "newDigi overlap", 800, 600)
    leg9 = ROOT.TLegend(0.60, 0.68, 0.88, 0.88); leg9.SetBorderSize(0)
    keep_alive.append(leg9)
    first = True
    for key in ["vm", "ve",  "vt", "NC"]:
        h = new_digi_hists[f"{key}_hitTime"].Clone(f"{key}_newDigi_ol")
        h.SetDirectory(0)
        h.Sumw2()
        if normalize and h.Integral() > 0:
            h.Scale(1.0 / h.Integral())
        prep_hist(h, colors[key], marker=markers[key])
        h.SetTitle("NewDigi: overlap of flavors;min veto hit time [ns];Events")
        h.Draw("E1" if first else "E1 SAME")
        first = False
        leg9.AddEntry(h, key, "lep")
        keep_alive.append(h)
    leg9.Draw()
    save_canvas(c9, "9_newDigi_overlap")

    
    
def check_missing_events(eventId_to_pdg, new_eventId_to_pdg):
    """
    Compare original and chunked eventId-to-pdg dictionaries to find missing events.

    Parameters:
        eventId_to_pdg (dict): Original dictionary of eventId -> pdg.
        new_eventId_to_pdg (dict): Possibly chunked dictionary with eventId -> pdg.

    Returns:
        missing_event_ids (set): Set of missing event IDs.
    """
    original_event_ids = set(eventId_to_pdg.keys())
    new_event_ids = set(new_eventId_to_pdg.keys())

    missing_event_ids = original_event_ids - new_event_ids

    total = len(original_event_ids)
    missing = len(missing_event_ids)
    percentage_missing = (missing / total * 100) if total > 0 else 0.0

    if missing_event_ids:
        print("Missing Run IDs and event IDs and their PDG codes:")
        for event_id in sorted(missing_event_ids):
            pdg = eventId_to_pdg.get(event_id, "Unknown")
            print(f"  ID: {event_id}, PDG: {pdg}")
    else:
        print("No missing event IDs.")

    print(f"\nTotal events: {total}")
    print(f"Missing events: {missing}")
    print(f"Missing percentage: {percentage_missing:.2f}%")

    return missing_event_ids

def fill_vetoCount_vetoTime(
    digi_chain,
    snd_geo,
    data_type,
    count_scifi_threshold=200,
    out_dir="voto_hist_seperate_plots",
    prefix="veto",
    show=False
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
        show (bool): if True, canvases stay interactive (no batch mode)
    Returns:
        dict of histograms
    """
    if not show:
        ROOT.gROOT.SetBatch(True)

    os.makedirs(out_dir, exist_ok=True)

    # -------- Histograms (global) --------
    h_count_veto     = ROOT.TH1F("count_veto", "Number of Veto Hits per Event;veto count;events", 36, -0.5, 35.5); h_count_veto.SetDirectory(0)

    h_count_veto_s1  = ROOT.TH1F("count_veto_s0", "Veto Hits per Event in Station 0;veto count;events", 12, -0.5, 11.5); h_count_veto_s1.SetDirectory(0)
    h_count_veto_s2  = ROOT.TH1F("count_veto_s1", "Veto Hits per Event in Station 1;veto count;events", 12, -0.5, 11.5); h_count_veto_s2.SetDirectory(0)
    h_count_veto_s3  = ROOT.TH1F("count_veto_s2", "Veto Hits per Event in Station 2;veto count;events", 12, -0.5, 11.5); h_count_veto_s3.SetDirectory(0)

    h_earliest_hitTime = ROOT.TH1F("earliest_hitTime", "Earliest Veto Hit Time per Event;time [ns];events", 50, 0, 50); h_earliest_hitTime.SetDirectory(0)
    h_hitTime          = ROOT.TH1F("hitTime", "All Veto Hit Times;time [ns];entries", 50, 0, 50); h_hitTime.SetDirectory(0)

    # Station histogram: x=station index, weight=count per event
    h_station_veto_count = ROOT.TH1F("station_veto_count", "Veto Hits per Station;station;entries", 5, -1-0.5, 3+0.5); h_station_veto_count.SetDirectory(0)
    for s in (-1, 0, 1, 2, 3):
        h_station_veto_count.GetXaxis().SetBinLabel(h_station_veto_count.GetXaxis().FindBin(s), str(s))

    # 2D: earliest time vs per-event veto count
    h_2d_hitTime_countVeto = ROOT.TH2F(
        "h_2d_hitTime_countVeto",
        "Earliest veto time vs veto count;earliest time [ns];veto count",
        50, -0.5, 49.5,
        36, -0.5, 35.5
    ); h_2d_hitTime_countVeto.SetDirectory(0)

    
    h_2d_lateHitTime_countVeto = ROOT.TH2F(
        "h_2d_lateHitTime_countVeto",
        "Latest veto time vs veto count;latest time [ns];veto count",
        50, -0.5, 49.5,
        36, -0.5, 35.5
    ); h_2d_lateHitTime_countVeto.SetDirectory(0)

    h_z_position_veto_count = ROOT.TH1F("z_position_veto_count", "Z Position of Veto Hits;z [cm];entries", 20, 270, 290); h_z_position_veto_count.SetDirectory(0)

    # -------- NEW: per-station time histograms --------
    h_earliest_hitTime_s = {
        0: ROOT.TH1F("earliest_hitTime_s0", "Earliest Veto Hit Time per Event (Station 0);time [ns];events", 50, 0, 50),
        1: ROOT.TH1F("earliest_hitTime_s1", "Earliest Veto Hit Time per Event (Station 1);time [ns];events", 50, 0, 50),
        2: ROOT.TH1F("earliest_hitTime_s2", "Earliest Veto Hit Time per Event (Station 2);time [ns];events", 50, 0, 50),
    }
    for h in h_earliest_hitTime_s.values(): h.SetDirectory(0)

    h_hitTime_s = {
        0: ROOT.TH1F("hitTime_s0", "All Veto Hit Times (Station 0);time [ns];entries", 50, 0, 50),
        1: ROOT.TH1F("hitTime_s1", "All Veto Hit Times (Station 1);time [ns];entries", 50, 0, 50),
        2: ROOT.TH1F("hitTime_s2", "All Veto Hit Times (Station 2);time [ns];entries", 50, 0, 50),
    }
    for h in h_hitTime_s.values(): h.SetDirectory(0)

    # -------- Setup / fast bindings --------
    MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()

    n_entries = digi_chain.GetEntries()
    get_entry = digi_chain.GetEntry
    get_pos = MuFilter.GetPosition

    # speed up Python calls
    fill_hitTime = h_hitTime.Fill
    fill_hitTime_s = {s: h_hitTime_s[s].Fill for s in (0, 1, 2)}
    fill_earliest = h_earliest_hitTime.Fill
    fill_earliest_s = {s: h_earliest_hitTime_s[s].Fill for s in (0, 1, 2)}
    fill_z = h_z_position_veto_count.Fill
    fill_count_veto = h_count_veto.Fill
    fill_s1 = h_count_veto_s1.Fill
    fill_s2 = h_count_veto_s2.Fill
    fill_s3 = h_count_veto_s3.Fill
    fill_station = h_station_veto_count.Fill
    fill_h2d = h_2d_hitTime_countVeto.Fill
    fill_h2d_late = h_2d_lateHitTime_countVeto.Fill

    # -------- Event loop --------
    for i in tqdm(range(n_entries), desc="events"):
        get_entry(i)

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
        if n_scifi < count_scifi_threshold:
            continue

        # ensure zeros recorded for per-event station counts
        station_counts = Counter({0: 0, 1: 0, 2: 0})
        earliest_t = None
        earliest_t_s = {0: None, 1: None, 2: None}
        latest_t = None

        for aHit in hits:
            t = aHit.GetTime()
            if earliest_t is None or t < earliest_t:
                earliest_t = t
                
            if latest_t is None or t > latest_t:
                latest_t = t

            detID = aHit.GetDetectorID()
            station = (detID // 1000) % 10
            if station < 0 or station > 2:
                # unexpected station; ignore quietly
                continue

            station_counts[station] += 1
            fill_hitTime(t)
            fill_hitTime_s[station](t)

            if earliest_t_s[station] is None or t < earliest_t_s[station]:
                earliest_t_s[station] = t

            get_pos(detID, A, B)
            z_mid = 0.5 * (A.z() + B.z())
            fill_z(z_mid)

        # per-event totals
        veto_count = sum(station_counts.values())
        fill_count_veto(veto_count)
        fill_s1(station_counts[0])
        fill_s2(station_counts[1])
        fill_s3(station_counts[2])

        if earliest_t is not None:
            fill_earliest(earliest_t)
            fill_h2d(earliest_t, veto_count)
        if latest_t is not None:
            fill_h2d_late(latest_t, veto_count)

        # per-station earliest (only if station had a hit this event)
        for s in (0, 1, 2):
            if earliest_t_s[s] is not None:
                fill_earliest_s[s](earliest_t_s[s])

        # aggregate per-station entries with weight = count for this event
        for s in (0, 1, 2):
            fill_station(s, station_counts[s])

    # -------- Save plots --------
    def save_hist(h, name, logy=False, draw_opt="HIST"):
        c = ROOT.TCanvas(f"c_{name}", "", 800, 600)
        if logy:
            c.SetLogy()
        h.SetLineWidth(2)
        h.Draw(draw_opt)
        c.SaveAs(os.path.join(out_dir, f"scifi_gt_{count_scifi_threshold}_{prefix}_{data_type}_{name}.pdf"))
        if not show:
            c.Close()

    save_hist(h_count_veto, "count_veto", logy=True)
    save_hist(h_count_veto_s1, "count_veto_s0", logy=True)
    save_hist(h_count_veto_s2, "count_veto_s1", logy=True)
    save_hist(h_count_veto_s3, "count_veto_s2", logy=True)

    save_hist(h_earliest_hitTime, "earliest_hitTime", logy=False)
    save_hist(h_hitTime, "hitTime", logy=True)
    save_hist(h_station_veto_count, "station_veto_count", logy=True)
    save_hist(h_z_position_veto_count, "z_position_veto_count", logy=False)

    # 2D: draw with COLZ
    c2 = ROOT.TCanvas("c_h2", "", 800, 600)
    h_2d_hitTime_countVeto.Draw("COLZ")
    c2.SaveAs(os.path.join(out_dir, f"scifi_gt_{count_scifi_threshold}_{prefix}_{data_type}_earliestTime_vs_countVeto.pdf"))
    if not show:
        c2.Close()
        
    c2 = ROOT.TCanvas("c_h2", "", 800, 600)
    h_2d_lateHitTime_countVeto.Draw("COLZ")
    c2.SaveAs(os.path.join(out_dir, f"scifi_gt_{count_scifi_threshold}_{prefix}_{data_type}_latestTime_vs_countVeto.pdf"))
    if not show:
        c2.Close()

    # NEW: per-station saving
    save_hist(h_earliest_hitTime_s[0], "earliest_hitTime_s0", logy=False)
    save_hist(h_earliest_hitTime_s[1], "earliest_hitTime_s1", logy=False)
    save_hist(h_earliest_hitTime_s[2], "earliest_hitTime_s2", logy=False)

    save_hist(h_hitTime_s[0], "hitTime_s0", logy=True)
    save_hist(h_hitTime_s[1], "hitTime_s1", logy=True)
    save_hist(h_hitTime_s[2], "hitTime_s2", logy=True)

    if not show:
        # tidy up canvases in batch
        ROOT.gROOT.GetListOfCanvases().Delete()

    # -------- Return all hists --------
    return {
        "count_veto": h_count_veto,
        "count_veto_s0": h_count_veto_s1,
        "count_veto_s1": h_count_veto_s2,
        "count_veto_s2": h_count_veto_s3,
        "earliest_hitTime": h_earliest_hitTime,
        "hitTime": h_hitTime,
        "station_veto_count": h_station_veto_count,
        "z_position_veto_count": h_z_position_veto_count,
        "h_2d_hitTime_countVeto": h_2d_hitTime_countVeto,
        "h_2d_lateHitTime_countVeto":h_2d_lateHitTime_countVeto,
        "earliest_hitTime_s0": h_earliest_hitTime_s[0],
        "earliest_hitTime_s1": h_earliest_hitTime_s[1],
        "earliest_hitTime_s2": h_earliest_hitTime_s[2],
        "hitTime_s0": h_hitTime_s[0],
        "hitTime_s1": h_hitTime_s[1],
        "hitTime_s2": h_hitTime_s[2],
    }
    
    #     # ---------- Plot & Save ----------
    # def _style(h):
    #     h.SetLineWidth(2)
    #     if isinstance(h, ROOT.TH1):
    #         h.SetStats(1)
    #     return h
    # def _save_pdf(h, name, logy=False):
    #     c = ROOT.TCanvas(f"c_{name}", h.GetTitle(), 900, 700)
    #     c.SetGrid()
    #     if logy:
    #         c.SetLogy()
    #     draw_opt = "HIST"
    #     _style(h).Draw(draw_opt)
    #     c.Update()
    #     pdf_name = f"{prefix}_{data_type}_{name}.pdf"
    #     pdf_path = os.path.join(out_dir, pdf_name)
    #     c.SaveAs(pdf_path)
    #     return c

    # _save_pdf(h_count_veto, "count_veto", logy=True)
    # _save_pdf(h_earliest_hitTime, "earliest_hitTime")
    # _save_pdf(h_hitTime, "hitTime", logy=True)
    # _save_pdf(h_station_veto_count, "station_veto_count")
    # _save_pdf(h_z_position_veto_count, "z_position_veto_count")


def plot_ve_realData_hists(
    hists_realData: dict,
    hists_ve: dict,
    out_dir="plots_compare",
    prefix="compare",
    label_real="realData",
    label_ve="Ve",
    normalize=True,
    make_ratio=False,
    rebin_map=None,
    logy_map=None
):
    ROOT.gStyle.SetOptStat(0)
    os.makedirs(out_dir, exist_ok=True)
    rebin_map = rebin_map or {}
    logy_map = logy_map or {}

    def _clone(h, suffix):
        h2 = h.Clone(h.GetName() + suffix)
        h2.SetDirectory(0)
        return h2

    def _style_rd(h):
        h.SetLineColor(ROOT.kBlack)
        h.SetMarkerColor(ROOT.kBlack)
        h.SetMarkerStyle(20)
        h.SetLineWidth(2)
        return h

    def _style_ve(h):
        h.SetLineColor(ROOT.kBlue + 1)
        h.SetMarkerColor(ROOT.kBlue + 1)
        h.SetMarkerStyle(24)
        h.SetLineWidth(2)
        return h
    
    def _bins_compatible_2d(a: ROOT.TH2, b: ROOT.TH2) -> bool:
        if a.GetNbinsX() != b.GetNbinsX(): return False
        if a.GetNbinsY() != b.GetNbinsY(): return False
        ax, bx = a.GetXaxis(), b.GetXaxis()
        ay, by = a.GetYaxis(), b.GetYaxis()
        if abs(ax.GetXmin() - bx.GetXmin()) > 1e-9 or abs(ax.GetXmax() - bx.GetXmax()) > 1e-9: return False
        if abs(ay.GetXmin() - by.GetXmin()) > 1e-9 or abs(ay.GetXmax() - by.GetXmax()) > 1e-9: return False
        return True

    def _infer_station_tag(k: str) -> str:
        m = re.search(r"(?:^|_)(s[0-9])(?:$|_)", k)
        return m.group(1) if m else ""

    keys = sorted(set(hists_realData.keys()) & set(hists_ve.keys()))
    for key in keys:
        hR0, hV0 = hists_realData[key], hists_ve[key]
        
        # ------------- TH2 branch (no rebin, no normalization, no ratio) -------------
        if isinstance(hR0, ROOT.TH2) and isinstance(hV0, ROOT.TH2):
            hR, hV = _clone(hR0, "_rd"), _clone(hV0, "_ve")
            if not _bins_compatible_2d(hR, hV):
                print(f"[skip 2D] {key}: incompatible binning")
                continue

            # Canvas with two panels: real (left), ve (right)
            c = ROOT.TCanvas(f"c2_{prefix}_{key}", key, 1400, 650)
            c.Divide(2, 1)

            # Common Z range (no normalization)
            # Use positive minimum for logZ if requested
            use_logz = bool(logy_map.get(key, False))
            zmin_pos = min([v for v in [hR.GetMinimum(1e-300), hV.GetMinimum(1e-300)] if v > 0] or [1e-12])
            zmax = max(hR.GetMaximum(), hV.GetMaximum())
            if zmax <= 0: zmax = 1.0

            # Draw real
            c.cd(1)
            ROOT.gPad.SetRightMargin(0.16); ROOT.gPad.SetLeftMargin(0.12)
            if use_logz: ROOT.gPad.SetLogz()
            hR.SetTitle(f"{key} - {label_real}")
            hR.SetMinimum(zmin_pos if use_logz else 0.0)
            hR.SetMaximum(zmax)
            hR.Draw("COLZ")

            # Draw ve
            c.cd(2)
            ROOT.gPad.SetRightMargin(0.16); ROOT.gPad.SetLeftMargin(0.12)
            if use_logz: ROOT.gPad.SetLogz()
            hV.SetTitle(f"{key} - {label_ve}")
            hV.SetMinimum(zmin_pos if use_logz else 0.0)
            hV.SetMaximum(zmax)
            hV.Draw("COLZ")

            # Title banner
            c.cd(0)
            pave = ROOT.TPaveText(0.12, 0.93, 0.88, 0.99, "NDC")
            pave.SetFillStyle(0); pave.SetBorderSize(0); pave.SetTextFont(42); pave.SetTextAlign(21)
            pave.AddText(f"{key}  -  {label_real} vs {label_ve}")
            pave.Draw()

            tag = _infer_station_tag(key)
            pdf_name = f"{prefix}{('_' + tag) if tag else ''}_{key}_{label_real}_vs_{label_ve}_2D.pdf"
            pdf_path = os.path.join(out_dir, pdf_name)
            c.SaveAs(pdf_path)
            del c
            continue
        
        
        if not isinstance(hR0, ROOT.TH1) or not isinstance(hV0, ROOT.TH1):
            print(f"[skip] {key}: not TH1")
            continue

        hR, hV = _clone(hR0, "_rd"), _clone(hV0, "_ve")

        rb = int(rebin_map.get(key, 1) or 1)
        if rb > 1:
            hR.Rebin(rb)
            hV.Rebin(rb)

        nR, nV = hR.Integral(), hV.Integral()
        if normalize:
            if nR > 0: hR.Scale(1.0 / nR)
            if nV > 0: hV.Scale(1.0 / nV)

        c = ROOT.TCanvas(f"c_{prefix}_{key}", key, 900, 800 if make_ratio else 700)
        if make_ratio:
            c.Divide(1, 2)
            top = c.cd(1)
            top.SetPad(0, 0.30, 1, 1)
            top.SetBottomMargin(0.04)
            if logy_map.get(key, False):
                top.SetLogy()
            bot = c.cd(2)
            bot.SetPad(0, 0, 1, 0.30)
            bot.SetTopMargin(0.04)
            bot.SetBottomMargin(0.32)
            bot.SetGridy()
            c.cd(1)
        else:
            if logy_map.get(key, False): c.SetLogy()
            c.SetGrid()

        maxR, maxV = hR.GetMaximum(), hV.GetMaximum()
        ypad = 1.35 if not logy_map.get(key, False) else 10.0
        yMax = max(maxR, maxV) * ypad
        yMin = 0.0 if not logy_map.get(key, False) else 1e-6

        frame = _clone(hR, "_frame")
        frame.Reset("ICESM")
        frame.SetStats(0)
        frame.SetMaximum(yMax)
        frame.SetMinimum(yMin)
        frame.SetTitle(hR.GetTitle())

        x_title = hR.GetXaxis().GetTitle()
        y_title = "Probability Density" if normalize else hR.GetYaxis().GetTitle()
        frame.GetXaxis().SetTitle(x_title)
        frame.GetYaxis().SetTitle(y_title)

        frame.Draw("AXIS")

        _style_rd(hR)
        _style_ve(hV)
        if maxR >= maxV:
            hR.Draw("E SAME"); hV.Draw("HIST SAME")
        else:
            hV.Draw("HIST SAME"); hR.Draw("E SAME")

        # Legend with counts
        leg = ROOT.TLegend(0.60, 0.72, 0.88, 0.88)
        leg.SetBorderSize(0); leg.SetFillStyle(0)

        label_r = f"{label_real} (veto tagged)"
        label_v = f"{label_ve}"
        leg.AddEntry(hR, label_r, "lep")
        leg.AddEntry(hV, label_v, "l")
        leg.Draw()
        
        # Title above plot
        title_box = ROOT.TPaveText(0.12, 0.92, 0.88, 0.99, "NDC")
        title_box.SetFillStyle(0)
        title_box.SetBorderSize(0)
        title_box.SetTextFont(42)
        title_box.SetTextAlign(21)  # center
        title_box.AddText(f"{key} - {label_real} vs {label_ve}")
        title_box.Draw()


        # Ratio
        if make_ratio:
            bot.cd()
            ratio = _clone(hR, "_ratio")
            ratio.SetTitle("")
            for b in range(1, ratio.GetNbinsX() + 1):
                den = hV.GetBinContent(b)
                num = hR.GetBinContent(b)
                if den > 0:
                    ratio.SetBinContent(b, num / den)
                    e_num, e_den = hR.GetBinError(b), hV.GetBinError(b)
                    err = (num / den) * ((e_num / num) ** 2 + (e_den / den) ** 2) ** 0.5 if num > 0 else 0
                    ratio.SetBinError(b, err)
                else:
                    ratio.SetBinContent(b, 0); ratio.SetBinError(b, 0)

            ratio.SetMarkerStyle(20); ratio.SetMarkerSize(0.8); ratio.SetLineWidth(1)
            ratio.GetYaxis().SetTitle(f"{label_real}/{label_ve}")
            ratio.GetYaxis().SetTitleSize(0.12)
            ratio.GetYaxis().SetTitleOffset(0.45)
            ratio.GetYaxis().SetLabelSize(0.10)
            ratio.GetYaxis().SetNdivisions(505)
            ratio.GetXaxis().SetTitleSize(0.12)
            ratio.GetXaxis().SetLabelSize(0.10)
            ratio.GetXaxis().SetTitle(x_title)
            ratio.SetMaximum(4.95)
            ratio.SetMinimum(0.05)
            ratio.Draw("E")

            line = ROOT.TLine(ratio.GetXaxis().GetXmin(), 1.0, ratio.GetXaxis().GetXmax(), 1.0)
            line.SetLineStyle(2)
            line.Draw()
            c.cd(1)

        # Save PDF
        pdf_name = f"{prefix}_station0_{key}_{label_real}_vs_{label_ve}.pdf"
        pdf_path = os.path.join(out_dir, pdf_name)
        c.SaveAs(pdf_path)

        del c, frame

   
def fill_origin_and_plot(digi_chain, outdir="./plots", show=False):
    """
    Count Veto-hit 'origins' (from linked SciFiPoints) per classified particle label,
    and plot a histogram for each particle label.

    Parameters
    ----------
    digi_chain : TChain
        Must provide MCTrack, Digi_MuFilterHits2MCPoints, Digi_MuFilterHits, ScifiPoint
    pdg_label_map : dict[int,str]
        Maps |pdg_code_event| to a label string (e.g. {12:'nu_e', 14:'nu_mu', ...})
    pdg_db : ROOT.TDatabasePDG
        For resolving PDG codes to names (origins)
    outdir : str|None
        If given, save each histogram to this directory as PNG
    show : bool
        Whether to call plt.show() for each figure
    """
    neutrino_pdgs = {12, -12, 14, -14, 16, -16}

    # Counter of weighted counts per (particle_label -> origin_name)
    origin_counts_by_particle = defaultdict(Counter)

    n_entries = digi_chain.GetEntries()
    for i in tqdm(range(n_entries), desc="events"):
        digi_chain.GetEntry(i)

        # --- classify event using first two MC tracks ---
        pdg0 = pdg1 = None
        if hasattr(digi_chain, "MCTrack"):
            try:
                if len(digi_chain.MCTrack) > 0:
                    pdg0 = digi_chain.MCTrack[0].GetPdgCode()
                if len(digi_chain.MCTrack) > 1:
                    pdg1 = digi_chain.MCTrack[1].GetPdgCode()
            except Exception:
                pass

        # Neutral-current heuristic: same-PDG neutrino in leading two tracks → shift ±100
        if (pdg0 == pdg1) and (pdg0 in neutrino_pdgs):
            pdg_code_event = pdg0 - 100 if pdg0 < 0 else pdg0 + 100
        else:
            pdg_code_event = pdg0

        # Fallback if pdg0 missing
        particle_label = pdg_label_map.get(abs(pdg_code_event) if pdg_code_event is not None else None, "other")

        # Map hits → MC points (weights)
        hit2MC = digi_chain.Digi_MuFilterHits2MCPoints[0]

        for aHit in digi_chain.Digi_MuFilterHits:
            if (not aHit.isValid()) or (aHit.GetSystem() != 1):  # only Veto
                continue

            detID = aHit.GetDetectorID()
            linksToMCPoints = hit2MC.wList(detID)  # iterable of (mc_point_i, weight)

            for mc_point_i, weight in linksToMCPoints:
                if mc_point_i >= digi_chain.ScifiPoint.GetEntries(): #to prevent segmentation fault
                    continue
                scifi_point = digi_chain.ScifiPoint[mc_point_i]
                scifi_point_pdg = scifi_point.PdgCode()

                # Resolve origin name via TDatabasePDG, fallback to numeric code
                p = pdg_db.GetParticle(scifi_point_pdg)
                origin_name = p.GetName() if p else str(scifi_point_pdg)

                # Accumulate weighted count
                origin_counts_by_particle[particle_label][origin_name] += float(weight)

    # ==== Plot: one histogram per particle label ====
    for pid_label, counts_counter in origin_counts_by_particle.items():
        # Sort origins by count (high to low)
        sorted_items = sorted(counts_counter.items(), key=lambda x: x[1], reverse=True)
        origins, counts = zip(*sorted_items) if sorted_items else ([], [])

        plt.figure()
        plt.bar(origins, counts)
        plt.title(f"Veto hit origins - {pid_label}")
        plt.xlabel("Origin (PDG name)")
        plt.ylabel("Weighted hit count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if outdir:
            import os
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(f"{outdir}/veto_origins_{pid_label}.png", dpi=150)

        if show:
            plt.show()
        else:
            plt.close()

    return origin_counts_by_particle
        
def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo
    
    


def main():
    digi_chain, new_digi_chain = load_chains_from_neutrinoMC("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv", max_rows=1)
    digi_chain_realData = load_chains_from_metadata("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2024_skim_runs_metadata.csv", max_rows=5)
    digi_chain_ve = load_chains_from_metadata("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_2024_ve_metadata.csv", max_rows=1000)
    
    # digi_hists, digi_df, eventId_to_pdg = fill_hist_oldDigi(digi_chain)
    # new_digi_hists, new_digi_df, new_eventId_to_pdg = fill_hist_MCEventBuilder(new_digi_chain, eventId_to_pdg)
    # missing_event_ids = check_missing_events(eventId_to_pdg, new_eventId_to_pdg)
    # plot_event_stats_table(new_digi_df, "withMCEventBuilder")
    # plot_event_stats_table(digi_df, "oldDigi")
    # plot_hist(digi_hists, new_digi_hists)
    
    geo_path = '/eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root'
    snd_geo = setup_geometry(geo_path)
    #fill_origin_and_plot(digi_chain)
    
    count_scifi_threshold = 200
    hists_ve = fill_vetoCount_vetoTime(digi_chain_ve, snd_geo, "ve", count_scifi_threshold)
    hists_realData = fill_vetoCount_vetoTime(digi_chain_realData, snd_geo, "readData", count_scifi_threshold)
    
    out_dir = f'veto_hist_scifi_gt_{count_scifi_threshold}'
    plot_ve_realData_hists(hists_realData, hists_ve, out_dir)
    
    
    
    

if __name__ == "__main__":
    main()