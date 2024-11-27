# optimize the code for checking the root file

import ROOT
# check_hits2MCPoints
def check_hits2MCPoints():

    file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP.root"

    f = ROOT.TFile(file_path, 'read')
    tree = f.Get('cbmsim')

    for event in tree:
        hit2MC =event.Digi_ScifiHits2MCPoints
        scifi_hits =event.Digi_ScifiHits


        #print(dir(hit2MC))
        print("length of h2mc",len(hit2MC))
        print("length of hits", len(scifi_hits))

        print(dir(hit2MC))

        
        break

# check_hitsTime
def check_hitsTime():

    file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP.root"
    out_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/data_examination/plots/hist_time.png"
    f = ROOT.TFile(file_path, 'read')
    tree = f.Get('cbmsim')
    
    count=0
    hist_time = ROOT.TH1F("time", "Time of Hits in one Events", 100, 0, 5)
    for event in tree:
        scifi_hits =event.Digi_ScifiHits
        
        for hit in scifi_hits:

            #print(dir(hit))
            scifi_time = hit.GetTime()
            hist_time.Fill(scifi_time)
        
        count+=1
        print(count)
        if(count>10):
            break
    canvas = ROOT.TCanvas("canvas", "Number of Hits Distribution", 1600, 1200)
    hist_time.Draw()
    canvas.SaveAs(out_path)

# check_ScifiCluster
def check_ScifiCluster_withTchain():


    file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPPo.root"


    tchain = ROOT.TChain("cbmsim")
    tchain.Add(file_path)  
    for i_event, event in enumerate(tchain):
        if(len(event.Reco_MuonTracks)>1):
            print("event:", i_event)
            reco = event.Reco_MuonTracks
            reco.Print()
            break 

def check_ScifiCluster():

    file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP.root"

    f = ROOT.TFile(file_path, 'read')
    tree = f.Get('cbmsim')
    #tree.Print()
    i = 0
    for event in tree:
        clusters = event.Cluster_Scifi
        hits = event.Digi_ScifiHits
        i+=1
        print("event", i)
        print("N of cluster:",clusters.GetEntries())
        print("N of hits:",hits.GetEntries())
        for cluster in clusters:
            
            help(cluster)
            cluster.Print()
            break
        break 

# check_MuonReco
def check_MuonTrack():

    file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP_muonReco.root"

    f = ROOT.TFile(file_path, 'read')
    tree = f.Get('cbmsim')

    i = 0
    for event in tree:
        reco = event.Reco_MuonTracks
        i+=1
        print("event:", i)
        print("N of reco tracks:",reco.GetEntries())
        if(len(reco)>1):

            j=0
            for track in reco:
                print("track:", j)
                j+=1
                #track.Print()
                #help(track)
                mom = track.getFittedState().getMom()
                pos = track.getFittedState().getPos()
                mom.Print()
                pos.Print()
                #break
            break 



