import ROOT

def check_with_TFile():

    file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP.root"

    f = ROOT.TFile(file_path, 'read')
    tree = f.Get('cbmsim')


    for event in tree:
        hit2MC =event.Digi_ScifiHits2MCPoints
        scifi_hits =event.Digi_ScifiHits


        for aHit in scifi_hits:

            detID = aHit.GetDetectorID()
            print("detId", detID)
            print("hit to mc")
            print(hit2MC[0].wList(detID))
            break
        print("length of h2mc",len(hit2MC))
        print("length of hits", len(scifi_hits))

        print("methods in Digi_ScifiHits2MCPoints: ")
        print(dir(hit2MC[0]))
        print(dir(hit2MC))

        print("Dumping of Digi_ScifiHits2MCPoints")
        hit2MC.Dump()

        print(type(hit2MC))
        print(type(hit2MC[0]))
        
        break

        
        

if __name__ == "__main__":
    print("checking hits2MCPoints")
    check_with_TFile()

    