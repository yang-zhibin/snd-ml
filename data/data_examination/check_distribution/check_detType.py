import ROOT

def check_with_TFile():
    print("reading file")
    file_path = "/eos/user/z/zhibin/sndData/converted/muons/muons_down_2025/1/sndLHC.Ntuple-TGeant4-160urad_Q4off_TCL6at1.85mm_20e6pp_digCPP_target.root"
    #out_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/data_examination/plots/hist_time.png"
    f = ROOT.TFile(file_path, 'read')
    tree = f.Get('cbmsim')
    
    count=0
    #hist_time = ROOT.TH1F("time", "Time of Hits in one Events", 100, 0, 5)

    print("start loop")
    for event in tree:
        scifi_hits =event.Digi_ScifiHits
        veto_count = 0
        us_count = 0
        ds_count = 0
        for aHit in event.Digi_MuFilterHits:
            if not aHit.isValid():
                continue
            detType = aHit.GetSystem()  # 0: scifi, 1: veto, 2: us, 3: ds
            if(detType ==1):
                veto_count+=1
            elif(detType ==2):
                us_count+=1
            elif(detType ==3):
                ds_count+=1
        print(f'veto:{veto_count}, scifi:,{len(event.Digi_ScifiHits)}, us:{us_count}, ds:{ds_count}')
        #if(count>100):
        #    break
    #canvas = ROOT.TCanvas("canvas", "Number of Hits Distribution", 1600, 1200)
    #hist_time.Draw()
    #canvas.SaveAs(out_path)

if __name__ == "__main__":
    print("checking det type")
    check_with_TFile()