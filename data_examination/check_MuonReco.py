import ROOT

def check_with_TChain():


    file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP_muonReco.root"


    tchain = ROOT.TChain("cbmsim")
    tchain.Add(file_path)  
    for i_event, event in enumerate(tchain):
        if(len(event.Reco_MuonTracks)>1):
            print("event:", i_event)
            reco = event.Reco_MuonTracks
            reco.Print()
            break 

def check_with_TFile():

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

if __name__ == "__main__":

    check_with_TFile()
