import ROOT
#import SndlhcGeo
import os
from argparse import ArgumentParser

import SndlhcGeo



def main(args):
    # prepare geo  file
    snd_geo = SndlhcGeo.GeoInterface(args.geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    Scifi = snd_geo.modules['Scifi']
    Mufi = snd_geo.modules['MuFilter']
    nav = ROOT.gGeoManager.GetCurrentNavigator()

    A, B = ROOT.TVector3(), ROOT.TVector3()

    raw_data = ROOT.TFile(args.rawData_path, 'read')
    raw_tree = raw_data.Get('cbmsim')

    stage1 = ROOT.TFile(args.stage1_file, 'read')
    stage1_tree = stage1.Get('cbmsim')

    stage2 = ROOT.TFile(args.stage2_file, 'read')
    stage2_tree = stage2.Get('cbmsim')

    recoMuon = ROOT.TFile(args.recoMuon_path, 'read')
    recoMuon_tree = recoMuon.Get('cbmsim')
    

    out_file = ROOT.TFile(args.out_path, "RECREATE")
    new_tree = ROOT.TTree('cbmsim', 'converted cbmsim tree')

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

    stage1_list = []
    for event in stage1_tree:
        stage1_list.append(event.EventHeader.GetEventNumber())

    stage2_list = []
    for event in stage2_tree:

        stage2_list.append(event.EventHeader.GetEventNumber())


    for i_event, event in enumerate(raw_tree):
        # reset
        hits.Clear()
        scifiCluster.Clear()
        reco_Muon.Clear()
        
        # ids
        ids.runId = event.EventHeader.GetRunId()
        ids.eventId = event.EventHeader.GetEventNumber()

        #labels
        event_pdg0 = event.MCTrack[0].GetPdgCode()
        event_pdg1 = event.MCTrack[1].GetPdgCode()
        
        if event_pdg0 == event_pdg1:
            labels.pdgCode = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
        else:
            labels.pdgCode = event_pdg0
        
        labels.x = event.MCTrack[1].GetStartX()
        labels.y = event.MCTrack[1].GetStartY()
        labels.z = event.MCTrack[1].GetStartZ()

        labels.px = event.MCTrack[0].GetPx()
        labels.py = event.MCTrack[0].GetPy()
        labels.pz = event.MCTrack[0].GetPz()
        
        #vm selection
        if (event.EventHeader.GetEventNumber() in stage1_list):
            vm_selection.stage1 = 1
        else:
            vm_selection.stage1 = 0
            
        if (event.EventHeader.GetEventNumber() in stage2_list):
            vm_selection.stage2 = 1
        else:
            vm_selection.stage2 = 0

        #reconstructed muon tracks
        if (recoMuon_tree.GetEntries()>0):
            recoMuon_tree.GetEntry(i_event)
            muonTrack = recoMuon_tree.Reco_MuonTracks
            if(len(muonTrack)>1):
                j=0
                for track in muonTrack:
                    mom = track.getFittedState().getMom()
                    pos = track.getFittedState().getPos()
                    
                    recoMuonTrack = reco_Muon.ConstructedAt(j)
                    j+=1
                    recoMuonTrack.px, recoMuonTrack.py, recoMuonTrack.pz = mom.X(), mom.Y(), mom.Z()
                    recoMuonTrack.x, recoMuonTrack.y, recoMuonTrack.z = pos.X(), pos.Y(), pos.Z()

            #hits
       
        i_hit = 0
        for aHit in event.Digi_ScifiHits:
            detID = aHit.GetDetectorID()
            Scifi.GetSiPMPosition(detID, A, B)

            hit = hits.ConstructedAt(i_hit)
            i_hit+=1

            hit.orientation = aHit.isVertical()
            hit.x1, hit.y1, hit.z1 = A.x(), A.y(), A.x()
            hit.x2, hit.y2, hit.z2 = B.x(), B.y(), B.x()
            hit.detType = 1 # 1: scifi, 2: us, 3: ds
            hit.hitTime = aHit.GetTime()

        for aHit in event.Digi_MuFilterHits:
            detID = aHit.GetDetectorID()
            Mufi.GetPosition(detID, A, B)

            hit = hits.ConstructedAt(i_hit)
            i_hit+=1

            hit.orientation = aHit.isVertical()
            hit.x1, hit.y1, hit.z1 = A.x(), A.y(), A.x()
            hit.x2, hit.y2, hit.z2 = B.x(), B.y(), B.x()
            hit.detType = detID // 10000 # 1: scifi, 2: us, 3: ds
            hit.hitTime = aHit.GetTime()


        new_tree.Fill()
        #if (i_event>5):
        #    break
    new_tree.Write()
    out_file.Close()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--rawData", dest="rawData_path", help="raw MC data digitized file path", required=True)
    parser.add_argument("-m", "--recoMuon_path", dest="recoMuon_path", help="reco muon data path", required=True)
    parser.add_argument("-s1", "--stage1_file", dest="stage1_file", help="stage 1 filtered data path", required=True)
    parser.add_argument("-s2", "--stage2_file", dest="stage2_file", help="stage 2 filtered file",required=True)
    parser.add_argument("-g", "--geoFile", dest="geo_file", help="geo file", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output directory", required=True)

    args = parser.parse_args()
    
    main(args)




