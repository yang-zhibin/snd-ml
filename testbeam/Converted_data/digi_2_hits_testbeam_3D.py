import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
from array import array
from tqdm import tqdm
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


def process_hits(event, snd_geo, hits, offset_df):
    """Process all hits in the event and update hits array and averages."""
    Scifi = snd_geo.modules['Scifi']
    A, B = ROOT.TVector3(), ROOT.TVector3()
    
    # ---------------------------
    # Pass 1: collect clock cycles
    # ---------------------------
    cc_all = []
    clock_period = 6.25
    nbins=100
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue

        # Define/compute your clock cycle consistently with your analysis
        # Example: if GetTime() is in "clock cycles already", then:
        #   cc = float(aHit.GetTime())
        # If GetTime() is in ns and you want cycle index, then:
        #   cc = aHit.GetTime() / clock_period
        cc = aHit.GetTime() / clock_period
        cc_all.append(cc)

    if not cc_all:
        return

    # ---------------------------
    # MPV via histogram mode
    # ---------------------------
    cmin, cmax = min(cc_all), max(cc_all)
    if cmin == cmax:
        mpv_cc = float(cmin)
    else:
        pad = 0.05 * (cmax - cmin)
        h = ROOT.TH1F("h_cc_tmp_process_hits", "", nbins, cmin - pad, cmax + pad)
        h.SetDirectory(0)  # avoid ROOT directory ownership / memory buildup
        for x in cc_all:
            h.Fill(x)
        mpv_cc = float(h.GetBinCenter(h.GetMaximumBin()))
        # optional cleanup
        # del h

    lo, hi = mpv_cc - 0.5, mpv_cc + 2.3

    # Process SciFi hits
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        cc = aHit.GetTime() / clock_period
        if not (lo <= cc <= hi):
            continue
        
        detID = aHit.GetDetectorID()
        
        st  = detID // 1000000
        ori = int(aHit.isVertical())           # 0/1
        mat = (detID % 100000) // 10000
        local_ch  = detID % 1000
        tofpet_id = (detID % 10000) // 1000
        ch = tofpet_id*128 + local_ch

        Scifi.GetSiPMPosition(detID, A, B)
        hit = hits.ConstructedAt(hits.GetEntries())
        hit.orientation = aHit.isVertical()
        hit.x1, hit.y1, hit.z1 = A.x(), A.y(), A.z()
        hit.x2, hit.y2, hit.z2 = B.x(), B.y(), B.z()
        hit.detType = 0  # 1: scifi, 2: veto, 3: us, 4: ds
        hit.hitTime = aHit.GetTime()
        hit.detId = detID

        qdc = aHit.GetSignal(0)
        if st == 2 and mat == 0 and ori==0 and ('real' in  args.type):
            mpv_value = offset_df.loc[ch, "mpv_data_avg"]
            qdc = qdc-mpv_value
        if qdc<0:
            qdc = 0
        hit.qdc = qdc


def main(args):
    print("start processing digi to hit")
    snd_geo = setup_geometry(args.geo_path)
    raw_data, raw_tree = open_root_file(args.digi_path)

    beam_type = args.pdg
        
    out_file, new_tree = create_output_file(args.out_path, args.mode)
    
    # Define branches (assuming branch setup functions are defined)
    ROOT.gROOT.ProcessLine(f".L {args.work_path}/snd-ml/testbeam/Converted_data/EventClass.h+")
    offset_df = pd.read_csv("/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/evaluation/QDC_offset/QDC_offset_st2_mat0_ori0.csv")
    offset_df = offset_df.set_index("channel")
    ids = ROOT.Id()
    hits = ROOT.TClonesArray("Hit")
    
    new_tree.Branch("Id", ids)
    new_tree.Branch("Hits", hits)
    
    n = raw_tree.GetEntries()
    if (n>args.max_event) and ('real' in  args.type):
        n = args.max_event
    
    for i in tqdm(range(n), total=n, desc="Processing events"):
        entry_number = i
        raw_tree.GetEntry(entry_number)
        
        hits.Clear()
        ids.clear()
        ids.runId = raw_tree.EventHeader.GetRunId()
        
        if ('MC' in  args.type):
            ids.isMC = 1
            try:
                ids.eventId = raw_tree.EventHeader.GetEventNumber()
            except Exception:
                ids.eventId = raw_tree.EventHeader.GetMCEntryNumber()

            event_pdg0 = raw_tree.MCTrack[0].GetPdgCode()
            ids.pdgCode = event_pdg0
                
        elif('real' in  args.type):
            if beam_type != 'no type':
                if 'pi+' in beam_type:
                    ids.pdgCode = 211
                elif 'pi-' in beam_type:
                    ids.pdgCode = -211
                elif 'mu-' in beam_type:
                    ids.pdgCode = 13
                elif 'e-' in beam_type:
                    ids.pdgCode = 11
                elif 'e+' in beam_type:
                    ids.pdgCode = -11
                else:
                    ids.pdgCode = 0
            else:
                ids.pdgCode = 0
            ids.isMC = 0
            ids.eventId = raw_tree.EventHeader.GetEventNumber()
        
        process_hits(raw_tree, snd_geo, hits, offset_df)
        new_tree.Fill()

    # Finalize the output file
    new_tree.Write()
    out_file.Close()
    print("finish processing digi to hit")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default='RECREATE')
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)
    parser.add_argument("-pdg", "--pdg", dest="pdg", help="PDG code", required=False, default='no type')
    parser.add_argument("-w", "--work_path", dest="work_path", help="work path",required=True)
    parser.add_argument("-m", "--max_event", dest="max_event", help="max processed events", required=False, default=2000)

    args = parser.parse_args()

    main(args)
   # print('testing law, digi to hits')
   
# python digi_2_hits_testbeam.py -d /eos/experiment/sndlhc/MonteCarlo/testbeam2024/250GeV_211/X_neg37.93_Y_41.74_Z_315/sndLHC.PG_211-TGeant4_digCPP.root -g  /eos/experiment/sndlhc/MonteCarlo/testbeam2024/250GeV_211/X_neg37.93_Y_41.74_Z_315/geofile_full.PG_211-TGeant4.root -o /eos/user/s/sfrankha/sndlhc/MonteCarlo/testbeam2024/250GeV_211/X_neg37.93_Y_41.74_Z_315/hit_sndLHC.PG_211-TGeant4_digCPP.root -t MC_data -pdg pi+