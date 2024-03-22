import ROOT

def check_with_TChain():


    file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPPo.root"


    tchain = ROOT.TChain("cbmsim")
    tchain.Add(file_path)  
    for i_event, event in enumerate(tchain):
        if(len(event.Reco_MuonTracks)>1):
            print("event:", i_event)
            reco = event.Reco_MuonTracks
            reco.Print()
            break 

def check_with_TFile():

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

if __name__ == "__main__":

    check_with_TFile()
