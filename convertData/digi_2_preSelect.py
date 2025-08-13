import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import array

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

def process_hits(event, snd_geo, branch_vars):
    """Process all hits in the event and update hits array and averages."""
    # Scifi = snd_geo.modules['Scifi']
    # MuFilter = snd_geo.modules['MuFilter']
    # A, B = ROOT.TVector3(), ROOT.TVector3()


    scifi_counts = [0] * 5  # scifi1 to scifi5
    veto_counts = [0] * 3 # veto1 to veto3
    ds_counts = [0] * 4  # ds1 to ds4
    us_counts = [0] * 5  # us1 to us5


    # Process SciFi hits
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        station = detID // 1000000
        if 1 <= station <= 5:
            scifi_counts[station - 1] += 1


    # Process MuFilter hits
    for aHit in event.Digi_MuFilterHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        detType = aHit.GetSystem()
        station = (detID // 1000) % 10

        if detType == 1:
            veto_counts[station] += 1
            
        elif detType == 3 and 0 <= station <= 3:
            ds_counts[station ] += 1
        elif detType == 2 and 0 <= station <= 4:
            us_counts[station ] += 1

    
    return veto_counts, scifi_counts, us_counts, ds_counts


def main(args):
    print("start processing digi to preSelection")
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)
    
    out_file, new_tree = create_output_file(args.out_path, args.mode)

    branches = [
        ("runId", 'i'), ("eventId", 'i'), ("isMC", 'i'), ("eventIndex", 'i'),("pdgCode", 'i'),
        ("At_least_1_non_veto_hit", 'i'),
        ("veto", 'i'), 
        ("scifi", 'i'),
        ("cut_H_if_DS_hits_must_all_US_hits",'i'),
        ("cut_G_has_consecutive_scifi_hits",'i'),
        ("preSelect", 'i'),
        ("preSelect_vetoFree", 'i'),
        ("preSelect_vetoTagged", 'i'),
        #energy, 
    ]

    # Dictionary to hold branch variables
    branch_vars = {}

    # Create branches dynamically
    for name, dtype in branches:
        branch_vars[name] = array.array(dtype, [-999])  # Initialize the array
        new_tree.Branch(name, branch_vars[name], f"{name}/{dtype.upper()}")
        
    # Process each event
    for i_event, event in enumerate(raw_tree):
        if i_event % 10000 == 0:
            print(f"processed {i_event} events")
        
        
        branch_vars["eventIndex"][0] = i_event
        branch_vars["runId"][0] = event.EventHeader.GetRunId()

        if ('MC' in  args.type):
            branch_vars["isMC"][0] = 1
            #print(dir(event.EventHeader))
            
            try:
                branch_vars["eventId"][0] = event.EventHeader.GetEventNumber()
            except Exception:
                branch_vars["eventId"][0] = event.EventHeader.GetMCEntryNumber()
            # Particle codes and initial position
            
            event_pdg0 = raw_tree.MCTrack[0].GetPdgCode()
            event_pdg1 = raw_tree.MCTrack[1].GetPdgCode()

            neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
            if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
                branch_vars["pdgCode"][0] = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
            else:
                branch_vars["pdgCode"][0] = event_pdg0


        elif('real' in  args.type):
            branch_vars["isMC"][0] = 0
            branch_vars["pdgCode"][0] = 0
            branch_vars["eventId"][0] = event.EventHeader.GetEventNumber()
        
        veto_counts, scifi_counts, us_counts, ds_counts = process_hits(event, snd_geo, branch_vars)
        
        branch_vars["scifi"][0] = sum(scifi_counts)
        branch_vars["veto"][0] = sum(veto_counts)
        
        if ((sum(scifi_counts)+sum(us_counts)+sum(ds_counts)) > 0):
            branch_vars["At_least_1_non_veto_hit"][0] = 1
        else:
            branch_vars["At_least_1_non_veto_hit"][0] = 0
                
        has_consecutive_hits = any(scifi_counts[i] and scifi_counts[i + 1] for i in range(len(scifi_counts) - 1))
        if has_consecutive_hits:
            branch_vars["cut_G_has_consecutive_scifi_hits"][0] = 1
        else:
            branch_vars["cut_G_has_consecutive_scifi_hits"][0] = 0
            
        
            
        if (sum(ds_counts)) > 0 and not (
            all(us_counts[i] > 0 for i in range(len(us_counts)))
        ):
            branch_vars["cut_H_if_DS_hits_must_all_US_hits"][0] = 0
        else:
            branch_vars["cut_H_if_DS_hits_must_all_US_hits"][0] = 1
        
        if (branch_vars["cut_H_if_DS_hits_must_all_US_hits"][0] ==1 and branch_vars["cut_G_has_consecutive_scifi_hits"][0] == 1 and branch_vars["scifi"][0]>5 and branch_vars["veto"][0] == 0):
            branch_vars["preSelect_vetoFree"][0] = 1
        else:
            branch_vars["preSelect_vetoFree"][0] = 0
        
        if (branch_vars["cut_H_if_DS_hits_must_all_US_hits"][0] ==1 and branch_vars["cut_G_has_consecutive_scifi_hits"][0] == 1 and branch_vars["scifi"][0]>5 and branch_vars["veto"][0] > 0):
            branch_vars["preSelect_vetoTagged"][0] = 1
        else:
            branch_vars["preSelect_vetoTagged"][0] = 0
        
        if (branch_vars["cut_H_if_DS_hits_must_all_US_hits"][0] ==1 and branch_vars["cut_G_has_consecutive_scifi_hits"][0] == 1 and branch_vars["scifi"][0]>5):
            branch_vars["preSelect"][0] = 1
        else:
            branch_vars["preSelect"][0] = 0
        
            
        
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

    args = parser.parse_args()

    main(args)