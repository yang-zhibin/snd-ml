import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo

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
    new_tree = ROOT.TTree('cbmsim', 'converted cbmsim tree')
    return out_file, new_tree

def process_event(args, event, snd_geo, ids, labels, hits, scifiCluster, reco_Muon, vm_selection, stage1_list, stage2_list, recoMuon_tree):
    """Process each event and update data structures accordingly."""

    #print(process_event)
    # Handle IDs and labels
    ids.runId = event.EventHeader.GetRunId()
    ids.eventId = event.EventHeader.GetEventNumber()
    ids.partitionId = args.partition
    if (args.type == 'MC'):
        ids.isMC = 1
    elif(args.type == 'real'):
        ids.isMC = 0
    if (args.type == 'MC'):
        # Particle codes and initial position
        event_pdg0 = event.MCTrack[0].GetPdgCode()
        event_pdg1 = event.MCTrack[1].GetPdgCode()

        neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
        if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
            labels.pdgCode = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
        else:
            labels.pdgCode = event_pdg0

        labels.x = event.MCTrack[1].GetStartX()
        labels.y = event.MCTrack[1].GetStartY()
        labels.z = event.MCTrack[1].GetStartZ()

        labels.px = event.MCTrack[0].GetPx()
        labels.py = event.MCTrack[0].GetPy()
        labels.pz = event.MCTrack[0].GetPz()

    # VM selection based on lists
    #print(stage1_list)
    vm_selection.stage1 = 1 if ids.eventId in stage1_list else 0
    vm_selection.stage2 = 1 if ids.eventId in stage2_list else 0

    # Process hits
    scifi_avg_ver, scifi_avg_hor, DS_avg_ver, DS_avg_hor, \
    scifi_avg_x_pos, scifi_avg_y_pos, DS_avg_x_pos, DS_avg_y_pos, \
    veto_counts, scifi_counts, us_counts, ds_counts = process_hits(event, snd_geo, hits)

    labels.scifi_avg_ver =   scifi_avg_ver
    labels.scifi_avg_hor =  scifi_avg_hor
    labels.DS_avg_ver = DS_avg_ver
    labels.DS_avg_hor = DS_avg_hor
    labels.scifi_avg_x_pos =   scifi_avg_x_pos
    labels.scifi_avg_y_pos =  scifi_avg_y_pos
    labels.DS_avg_x_pos = DS_avg_x_pos
    labels.DS_avg_y_pos = DS_avg_y_pos

    labels.veto1 = veto_counts[0]
    labels.veto2 = veto_counts[1]

    labels.scifi1 = scifi_counts[0]
    labels.scifi2 = scifi_counts[1]
    labels.scifi3 = scifi_counts[2]
    labels.scifi4 = scifi_counts[3]
    labels.scifi5 = scifi_counts[4]

    labels.us1 = us_counts[0]
    labels.us2 = us_counts[1]
    labels.us3 = us_counts[2]
    labels.us4 = us_counts[3]
    labels.us5 = us_counts[4]

    labels.ds1 = ds_counts[0]
    labels.ds2 = ds_counts[1]
    labels.ds3 = ds_counts[2]
    labels.ds4 = ds_counts[3]


    # Handle reconstructed muon tracks
    if recoMuon_tree:
        recoMuon_tree.GetEntry(event.EventHeader.GetEventNumber())
        muonTracks = recoMuon_tree.Reco_MuonTracks
        for j, track in enumerate(muonTracks):
            recoMuonTrack = reco_Muon.ConstructedAt(j)
            mom = track.getTrackMom()
            pos = track.getStart()
            recoMuonTrack.px = mom.x()
            recoMuonTrack.py = mom.y()
            recoMuonTrack.pz = mom.z()
            recoMuonTrack.x = pos.x()
            recoMuonTrack.y = pos.y()
            recoMuonTrack.z = pos.z()

    #return ids, labels, vm_selection

def process_hits(event, snd_geo, hits):
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
    veto_counts = [0] * 2 # veto1 to veto2
    ds_counts = [0] * 4  # ds1 to ds4
    us_counts = [0] * 5  # us1 to us5

    # Reset hits array
    hits.Clear()

    # Process SciFi hits
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        station = detID // 1000000
        if 1 <= station <= 5:
            scifi_counts[station - 1] += 1

        Scifi.GetSiPMPosition(detID, A, B)
        hit = hits.ConstructedAt(hits.GetEntries())
        hit.orientation = aHit.isVertical()
        hit.x1, hit.y1, hit.z1 = A.x(), A.y(), A.z()
        hit.x2, hit.y2, hit.z2 = B.x(), B.y(), B.z()
        hit.detType = 0  # 0: scifi, 1: veto, 2: us, 3: ds
        hit.hitTime = aHit.GetTime()
        hit.detId = detID

        # Calculate average positions
        channel = aHit.GetSiPMChan()
        mat = aHit.GetMat()
        sipm = aHit.GetSiPM()
        x = channel + sipm * 128 + mat * 4 * 128
        if aHit.isVertical():
            scifi_avg_ver += x
            #print(f'x1:{hit.x1}, x2:{hit.x2}, y1:{hit.y1}, y2:{hit.y2}')
            scifi_avg_x_pos += hit.x1
            scifi_n_ver += 1
        else:
            scifi_avg_hor += x
            scifi_avg_y_pos = hit.y1
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
        hit = hits.ConstructedAt(hits.GetEntries())
        hit.orientation = aHit.isVertical()
        hit.x1, hit.y1, hit.z1 = A.x(), A.y(), A.z()
        hit.x2, hit.y2, hit.z2 = B.x(), B.y(), B.z()
        hit.detType = aHit.GetSystem()  # 0: scifi, 1: veto, 2: us, 3: ds
        hit.hitTime = aHit.GetTime()
        hit.detId = detID

        # DS hit averaging, considering only system '3' which is downstream
        if aHit.GetSystem() == 3:
            x = detID % 1000
            if aHit.isVertical():
                DS_avg_ver += x
                DS_avg_x_pos +=hit.x1
                DS_n_ver += 1
            else:
                DS_avg_hor += x
                DS_avg_y_pos +=hit.y1
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

    # Update label averages
    #print(DS_avg_y_pos)
    return scifi_avg_ver, scifi_avg_hor, DS_avg_ver, DS_avg_hor, \
        scifi_avg_x_pos, scifi_avg_y_pos,  DS_avg_x_pos, DS_avg_y_pos, \
        veto_counts, scifi_counts, us_counts, ds_counts


def prepare_event_lists(stage1_tree, stage2_tree):
    """
    Create lists of event numbers from stage1 and stage2 trees.

    Args:
    stage1_tree (ROOT.TTree): The ROOT tree containing stage1 data.
    stage2_tree (ROOT.TTree): The ROOT tree containing stage2 data.

    Returns:
    tuple: A tuple containing two lists, (stage1_list, stage2_list).
    """
    stage1_list = []
    for event in stage1_tree:
        stage1_list.append(event.EventHeader.GetEventNumber())
    stage2_list = []
    for event in stage2_tree:
        stage2_list.append(event.EventHeader.GetEventNumber())
    return stage1_list, stage2_list


def main(args):
    snd_geo = setup_geometry(args.geo_file)
    raw_data, raw_tree = open_root_file(args.rawData_path)
    stage1_list, stage2_list = [], []
    if args.stage1_file and args.stage2_file:
        stage1, stage1_tree = open_root_file(args.stage1_file)
        stage2, stage2_tree = open_root_file(args.stage2_file)
        stage1_list, stage2_list = prepare_event_lists(stage1_tree, stage2_tree)

    recoMuon, recoMuon_tree = None, None
    if args.recoMuon_path:
        recoMuon, recoMuon_tree = open_root_file(args.recoMuon_path)
    
    out_file, new_tree = create_output_file(args.out_path, args.mode)
    
    # Define branches (assuming branch setup functions are defined)
    ROOT.gROOT.ProcessLine(".L EventClasses.h+")
    ids = ROOT.Id()
    labels = ROOT.Label()
    hits = ROOT.TClonesArray("Hit")
    scifiCluster = ROOT.TClonesArray("ScifiCluster")
    reco_Muon = ROOT.TClonesArray("RecoMuon")
    vm_selection = ROOT.VM_Selection()


    new_tree.Branch("Id", ids)
    new_tree.Branch("Label", labels)
    new_tree.Branch("Hits", hits)
    new_tree.Branch("ScifiCluster", scifiCluster)
    new_tree.Branch("RecoMuon", reco_Muon)
    new_tree.Branch("VmSeclection", vm_selection)
    
    # Process each event
    for i_event, event in enumerate(raw_tree):
        #reset
        ids.clear()
        labels.clear()
        vm_selection.clear()

        hits.Clear()
        scifiCluster.Clear()
        reco_Muon.Clear()

        if not (event.Digi_ScifiHits.GetEntriesFast() or event.Digi_MuFilterHits.GetEntriesFast()):
            continue

        process_event(args, event, snd_geo, ids, labels, hits, scifiCluster, reco_Muon, vm_selection, stage1_list, stage2_list, recoMuon_tree)
        #print(f"Processing event {ids.eventId}: Stage 1 Selected: {vm_selection.stage1}, Stage 2 Selected: {vm_selection.stage2}")
        #print(len(hits))
        #if (i_event >10):
        #    break
        new_tree.Fill()

    # Finalize the output file
    new_tree.Write()
    out_file.Close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--rawData", dest="rawData_path", help="raw MC data digitized file path", required=True)
    parser.add_argument("-m", "--recoMuon_path", dest="recoMuon_path", help="reco muon data path", required=False)
    parser.add_argument("-s1", "--stage1_file", dest="stage1_file", help="stage 1 filtered data path", required=False)
    parser.add_argument("-s2", "--stage2_file", dest="stage2_file", help="stage 2 filtered file",required=False)
    parser.add_argument("-g", "--geoFile", dest="geo_file", help="geo file", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output directory", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default='RECREATE')
    parser.add_argument("-id", "--partition", dest='partition', help='partition as file id', type=int, default=-1)
    parser.add_argument("-p", "--particle", dest='particle', help='particle type', required=False)
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)

    args = parser.parse_args()
    
    main(args)