import ROOT

def check_with_TFile():
    print("reading file")
    file_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/test_data_1/sndLHC.Genie-TGeant4_20240126_digCPP.root"
    out_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/data_examination/plots/hist_time.png"
    print("reading1")
    f = ROOT.TFile(file_path, 'read')
    print("reading2")
    tree = f.Get('cbmsim')
    
    count=0
    hist_time = ROOT.TH1F("time", "Time of Hits in one Events", 100, 0, 5)

    print("start loop")
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
    print("checking hitsTime")
    check_with_TFile()