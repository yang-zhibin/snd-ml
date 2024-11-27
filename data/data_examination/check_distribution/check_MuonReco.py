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
    print('checking reconstructed muon tracks')

    for i in range (1, 2):
        print(i)
        file_path = f"/eos/user/z/zhibin/sndData/converted/Neutrinos/{i}/sndLHC.Genie-TGeant4_20240126_digCPP__muonReco.root"
        #file_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/test_data_1/sndLHC.PG_2112-TGeant4_20240126_digCPP__muonReco.root'
        print('opening ', file_path)
        f = ROOT.TFile(file_path, 'read')
        print('reading tree')
        tree = f.Get('cbmsim')
        print(tree.GetEntries())
        print('start looping')
        i = 0
        for event in tree:
            reco = event.Reco_MuonTracks
            i+=1

            if(len(reco)>=1):
                print("event:", i)
                print("N of reco tracks:",reco.GetEntries())

                j=0
                for track in reco:
                    print("track:", j)
                    j+=1
                    #print(dir(track))
                    #help(track)
                    mom = track.getTrackMom()
                    pos = track.getStart()
                    pos_end = track.getStop()
                    mom.Print()
                    pos.Print()
                    print(mom.x(), mom.y(), mom.z())
                    print(pos.x(), pos.y(), pos.z())
                    print(pos_end.x(), pos_end.y(), pos_end.z())
                    #print(mom.X(), mom.Y(), mom.Z())
                    #print(pos.X(), pos.Y(), pos.Z())
                    #print(dir(mom))
                    #print(dir(pos))
                    #break
                break 

if __name__ == "__main__":
    
    check_with_TFile()
