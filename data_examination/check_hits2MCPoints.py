import ROOT

def check_with_TFile():

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

if __name__ == "__main__":
    print("checking hits2MCPoints")
    check_with_TFile()