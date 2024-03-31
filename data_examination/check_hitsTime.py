import ROOT

def check_with_TFile():

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

if __name__ == "__main__":
    print("checking hits2MCPoints")
    check_with_TFile()