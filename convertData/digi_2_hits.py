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

def open_root_file(file_path, mode='read'):
    """Open and return a ROOT file and its primary tree."""
    file = ROOT.TFile(file_path, mode)
    tree = file.Get('cbmsim')
    return file, tree

def create_output_file(path, mode):
    """Create and return a new ROOT file and a new tree for output."""
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    out_file = ROOT.TFile(path, mode)
    new_tree = ROOT.TTree('snddata', 'converted SND hits tree')
    return out_file, new_tree


def process_hits(event, snd_geo, hits):
    """Process all hits in the event and update hits array and averages."""
    Scifi = snd_geo.modules['Scifi']
    MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()

    # Reset hits array
    hits.Clear()

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
        hit.detType = 0  # 0: scifi, 1: veto, 2: us, 3: ds
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


    # Process MuFilter hits
    for aHit in event.Digi_MuFilterHits:
        #drop invalid hit and veto hit not using in training
        if not aHit.isValid() or aHit.GetSystem() == 2:
            continue
        detID = aHit.GetDetectorID()

        MuFilter.GetPosition(detID, A, B)
        hit = hits.ConstructedAt(hits.GetEntries())
        hit.orientation = aHit.isVertical()
        hit.x1, hit.y1, hit.z1 = A.x(), A.y(), A.z()
        hit.x2, hit.y2, hit.z2 = B.x(), B.y(), B.z()
        hit.detType = aHit.GetSystem()  # 0: scifi, 1: veto, 2: us, 3: ds
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
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)
    
    
    out_file, new_tree = create_output_file(args.out_path, args.mode)
    
    # Define branches (assuming branch setup functions are defined)
    ROOT.gROOT.ProcessLine(".L /afs/cern.ch/user/z/zhibin/work/snd-ml/convertData/EventClass.h+")

    ids = ROOT.Id()
    hits = ROOT.TClonesArray("Hit")

    
    new_tree.Branch("Id", ids)
    new_tree.Branch("Hits", hits)

    
    # Process each event
    for i_event, event in enumerate(raw_tree):
        #reset
        print(i_event)
        hits.Clear()
        ids.clear()
        ids.runId = event.EventHeader.GetRunId()
        

        if ('MC' in  args.type):
            ids.isMC = 1
            if ('kaon' in args.type or 'neutron' in args.type):
                try:
                    ids.eventId = event.EventHeader.GetEventNumber()
                except Exception:
                    ids.eventId = event.EventHeader.GetMCEntryNumber()
            else:
                ids.eventId = event.EventHeader.GetMCEntryNumber()
            # Particle codes and initial position
            event_pdg0 = event.MCTrack[0].GetPdgCode()
            event_pdg1 = event.MCTrack[1].GetPdgCode()

            neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
            if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
                ids.pdgCode = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
            else:
                ids.pdgCode = event_pdg0

        elif('real' in  args.type):
            ids.isMC = 0
            ids.pdgCode=0
            ids.eventId = event.EventHeader.GetEventNumber()

        # add least one hit to process
        if not (event.Digi_ScifiHits.GetEntriesFast() or event.Digi_MuFilterHits.GetEntriesFast()):
            new_tree.Fill()
        else:
            process_hits(event, snd_geo, hits)
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

    args = parser.parse_args()

    main(args)
   # print('testing law, digi to hits')