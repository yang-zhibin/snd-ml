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
    new_tree = ROOT.TTree('snddata', 'converted cbmsim tree')
    return out_file, new_tree

def process_event(args, event, snd_geo, ids, labels, hits, scifiCluster,stage1_list):
    """Process each event and update data structures accordingly."""

    #print(process_event)
    # Handle IDs and labels
    ids.runId = event.EventHeader.GetRunId()
    ids.eventId = event.EventHeader.GetEventNumber()
    ids.partitionId = args.partition

    vm_selection.stage1 = 1 if ids.eventId in stage1_list else 0

    # Process hits
    veto_flag, scifi_avg_ver, scifi_avg_hor, DS_avg_ver, DS_avg_hor = process_hits(event, snd_geo, hits)

    labels.scifi_avg_ver =   scifi_avg_ver
    labels.scifi_avg_hor =  scifi_avg_hor
    labels.DS_avg_ver = DS_avg_ver
    labels.DS_avg_hor = DS_avg_hor

    #return ids, labels, vm_selection
    return veto_flag

def process_hits(event, snd_geo, branch_vars):
    """Process all hits in the event and update hits array and averages."""
    Scifi = snd_geo.modules['Scifi']
    MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()

    scifi_avg_ver = 0
    scifi_avg_hor = 0
    scifi_n_ver = 0
    scifi_n_hor = 0

    DS_avg_ver = 0
    DS_avg_hor = 0
    DS_n_ver = 0
    DS_n_hor = 0

    scifi_avg_x_pos = 0
    scifi_avg_y_pos = 0
    DS_avg_x_pos = 0
    DS_avg_y_pos = 0

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

        Scifi.GetSiPMPosition(detID, A, B)


        # Calculate average positions
        channel = aHit.GetSiPMChan()
        mat = aHit.GetMat()
        sipm = aHit.GetSiPM()
        x = channel + sipm * 128 + mat * 4 * 128
        if aHit.isVertical():
            scifi_avg_ver += x
            #print(f'x1:{hit.x1}, x2:{hit.x2}, y1:{hit.y1}, y2:{hit.y2}')
            scifi_avg_x_pos += A.x()
            scifi_n_ver += 1
        else:
            scifi_avg_hor += x
            scifi_avg_y_pos += A.y()
            scifi_n_hor += 1

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


        MuFilter.GetPosition(detID, A, B)

        # DS hit averaging, considering only system '3' which is downstream
        if aHit.GetSystem() == 3:
            x = detID % 1000
            if aHit.isVertical():
                DS_avg_ver += x
                DS_avg_x_pos +=A.x()
                DS_n_ver += 1
            else:
                DS_avg_hor += x
                DS_avg_y_pos +=A.y()
                DS_n_hor += 1

    # Compute final averages
    if scifi_n_hor > 0:
        scifi_avg_hor /= scifi_n_hor
        scifi_avg_y_pos /= scifi_n_hor
    else:
        scifi_avg_hor = -1
        scifi_avg_y_pos = -100

    if scifi_n_ver > 0:
        scifi_avg_ver /= scifi_n_ver
        scifi_avg_x_pos /= scifi_n_ver
    else:
        scifi_avg_ver = -1
        scifi_avg_x_pos = -100

    if DS_n_hor > 0:
        DS_avg_hor /= DS_n_hor
        DS_avg_y_pos /= DS_n_hor
    else:
        DS_avg_hor = -1
        DS_avg_y_pos = -100

    if DS_n_ver > 0:
        DS_avg_ver /= DS_n_ver
        DS_avg_x_pos /= DS_n_ver
    else:
        DS_avg_ver = -1
        DS_avg_x_pos =-100

    # Update branch_vars with computed values
    branch_vars["scifi_avg_ver"][0] = scifi_avg_ver
    branch_vars["scifi_avg_hor"][0] = scifi_avg_hor
    branch_vars["DS_avg_ver"][0] = DS_avg_ver
    branch_vars["DS_avg_hor"][0] = DS_avg_hor

    branch_vars["scifi_avg_x_pos"][0] = scifi_avg_x_pos
    branch_vars["scifi_avg_y_pos"][0] = scifi_avg_y_pos
    branch_vars["DS_avg_x_pos"][0] = DS_avg_x_pos
    branch_vars["DS_avg_y_pos"][0] = DS_avg_y_pos

    # Updating SciFi station counts
    for i in range(5):
        branch_vars[f"scifi{i+1}"][0] = scifi_counts[i]

    # Updating Veto counts
    for i in range(3):
        branch_vars[f"veto{i+1}"][0] = veto_counts[i]

    # Updating Downstream station counts
    for i in range(4):
        branch_vars[f"ds{i+1}"][0] = ds_counts[i]

    # Updating Upstream station counts
    for i in range(5):
        branch_vars[f"us{i+1}"][0] = us_counts[i]

    return 


def main(args):
    print("start processing digi to hit")
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)
    
    out_file, new_tree = create_output_file(args.out_path, args.mode)

    branches = [
        ("runId", 'i'), ("eventId", 'i'), ("pdgCode", 'i'), ("isMC", 'i'),  
        ("px", 'f'), ("py", 'f'), ("pz", 'f'),  # Floats
        ("x", 'f'), ("y", 'f'), ("z", 'f'),    # Floats
        ("scifi_avg_ver", 'd'), ("scifi_avg_hor", 'd'),  # Doubles
        ("DS_avg_ver", 'd'), ("DS_avg_hor", 'd'),
        ("scifi_avg_x_pos", 'd'), ("scifi_avg_y_pos", 'd'),
        ("DS_avg_x_pos", 'd'), ("DS_avg_y_pos", 'd'),
        ("veto1", 'i'), ("veto2", 'i'), ("veto3", 'i'),  # Integers
        ("scifi1", 'i'), ("scifi2", 'i'), ("scifi3", 'i'),("scifi4", 'i'), ("scifi5", 'i'),
        ("us1", 'i'), ("us2", 'i'), ("us3", 'i'),("us4", 'i'), ("us5", 'i'),
        ("ds1", 'i'), ("ds2", 'i'), ("ds3", 'i'), ("ds4", 'i')
    ]

    # Dictionary to hold branch variables
    branch_vars = {}

    # Create branches dynamically
    for name, dtype in branches:
        branch_vars[name] = array.array(dtype, [-1])  # Initialize the array
        new_tree.Branch(name, branch_vars[name], f"{name}/{dtype.upper()}")
        
    # Process each event
    for i_event, event in enumerate(raw_tree):

        # add least one hit
        if not (event.Digi_ScifiHits.GetEntriesFast() or event.Digi_MuFilterHits.GetEntriesFast()):
            continue
        
        branch_vars["runId"][0] = event.EventHeader.GetRunId()

        if ('MC' in  args.type):
            branch_vars["isMC"][0] = 1
            #print(dir(event.EventHeader))
            if ('kaon' in args.type or 'neutron' in args.type):
                try:
                    branch_vars["eventId"][0] = event.EventHeader.GetEventNumber()
                except Exception:
                    branch_vars["eventId"][0] = event.EventHeader.GetMCEntryNumber()
            else:
                branch_vars["eventId"][0] = event.EventHeader.GetMCEntryNumber()
            # Particle codes and initial position
            event_pdg0 = event.MCTrack[0].GetPdgCode()
            event_pdg1 = event.MCTrack[1].GetPdgCode()

            neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
            if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
                branch_vars["pdgCode"][0] = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
            else:
                branch_vars["pdgCode"][0] = event_pdg0

            branch_vars["x"][0]= event.MCTrack[1].GetStartX()
            branch_vars["y"][0]= event.MCTrack[1].GetStartY()
            branch_vars["z"][0]= event.MCTrack[1].GetStartZ()

            branch_vars["px"][0] = event.MCTrack[0].GetPx()
            branch_vars["py"][0] = event.MCTrack[0].GetPy()
            branch_vars["pz"][0] = event.MCTrack[0].GetPz()

        elif('real' in  args.type):
            branch_vars["isMC"][0] = 0
            branch_vars["pdgCode"][0]=0
            branch_vars["eventId"][0] = event.EventHeader.GetEventNumber()

        process_hits(event, snd_geo, branch_vars)
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