import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
from array import array

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


def process_hits(event, snd_geo, hits):
    """Process all hits in the event and update hits array and averages."""
    Scifi = snd_geo.modules['Scifi']
    A, B = ROOT.TVector3(), ROOT.TVector3()

    # Process SciFi hits
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()

        Scifi.GetSiPMPosition(detID, A, B)
        hit = hits.ConstructedAt(hits.GetEntries())
        hit.orientation = aHit.isVertical()
        hit.x1, hit.y1, hit.z1 = A.x(), A.y(), A.z()
        hit.x2, hit.y2, hit.z2 = B.x(), B.y(), B.z()
        hit.detType = 0  # 1: scifi, 2: veto, 3: us, 4: ds
        hit.hitTime = aHit.GetTime()
        hit.detId = detID

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
        hit.qdc = this_qdc


def main(args):
    print("start processing digi to hit")
    snd_geo = setup_geometry(args.geo_path)
    raw_data, raw_tree = open_root_file(args.digi_path)

    beam_type = args.pdg
        
    out_file, new_tree = create_output_file(args.out_path, args.mode)
    
    # Define branches (assuming branch setup functions are defined)
    ROOT.gROOT.ProcessLine(".L /afs/.cern.ch/user/s/sfrankha/work/snd-ml/testbeam/Converted_data/EventClass.h+")

    ids = ROOT.Id()
    hits = ROOT.TClonesArray("Hit")
    
    new_tree.Branch("Id", ids)
    new_tree.Branch("Hits", hits)
    
    for i in range(raw_tree.GetEntries()):
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
        
        process_hits(raw_tree, snd_geo, hits)
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

    args = parser.parse_args()

    main(args)
   # print('testing law, digi to hits')
   
# python digi_2_hits_testbeam.py -d /eos/experiment/sndlhc/MonteCarlo/testbeam2024/250GeV_211/X_neg37.93_Y_41.74_Z_315/sndLHC.PG_211-TGeant4_digCPP.root -g  /eos/experiment/sndlhc/MonteCarlo/testbeam2024/250GeV_211/X_neg37.93_Y_41.74_Z_315/geofile_full.PG_211-TGeant4.root -o /eos/user/s/sfrankha/sndlhc/MonteCarlo/testbeam2024/250GeV_211/X_neg37.93_Y_41.74_Z_315/hit_sndLHC.PG_211-TGeant4_digCPP.root -t MC_data -pdg pi+