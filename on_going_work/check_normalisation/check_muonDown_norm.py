import ROOT
import pandas as pd
import numpy as np
import os, uuid
import math

# read csv
# read muon down MC feature root file
# preSelect
# plot avg scifi y pos
# normolise hist

ROOT.gROOT.SetBatch(True)
ROOT.ROOT.EnableImplicitMT()


def read_rdf(metadata_df):
    feature_chain = ROOT.TChain("sndData")
    integrated_lumi = 0
    

    
    for _, row in metadata_df.iterrows():
        vetoTagged_feature_path = row['vetoTagged_feature_path']
        if os.path.exists(vetoTagged_feature_path):
            feature_chain.Add(vetoTagged_feature_path)
            integrated_lumi += 0 if math.isnan(row['lumi_per_file']) else row['lumi_per_file']
            
        
    
        
    rdf = ROOT.RDataFrame(feature_chain)
    
    ## Good DS track 
        # 1 converged fit (1 track for 1 tracking method)
        # 2 slopes in both projections below 80 mrad ('getSlopeXZ', 'getSlopeYZ')
        # 3 χ2/ndf < 5 (getChi2Ndf)
        # 4 extrapolated DS track at the Veto planes is within 3 cm of a fired Veto bar
        # 
        
        ## Good Scifi track
        # 1 converged fit (1 track for 1 tracking method) ('getTrackType()')
        # 2 χ2/ndf < 20
        
        #Good tracks
        # If SciFi track’s and DS track’s projections on the reference plane are within 3 cm distance ('extrapolateToPlaneAtZ')
        
    # rdf = rdf.Filter("(HT_DS_Chi2Ndf < 5 && HT_DS_Chi2Ndf > 0) && (HT_DS_angle_xz < 80 && HT_DS_angle_xz > 0 && HT_DS_angle_yz < 80 && HT_DS_angle_yz > 0) &&"
    #                  "(HT_Scifi_Chi2Ndf < 20 && HT_Scifi_Chi2Ndf > 0) && (HT_Scifi_angle_xz < 80 && HT_Scifi_angle_xz > 0 && HT_Scifi_angle_yz < 80 && HT_Scifi_angle_yz > 0) &&"
    #                  "(HT_DS_to_Scifi_x < 3 && HT_DS_to_Scifi_x > 0 && HT_DS_to_Scifi_y < 3 && HT_DS_to_Scifi_y > 0) &&"
    #                  "(HT_DS_vetoDy < 3 && HT_DS_vetoDy >0)"
    #                  )
    
    # rdf = rdf.Filter("(HT_DS_Chi2Ndf < 5 && HT_DS_Chi2Ndf > 0) &&"
    #                  "(HT_Scifi_Chi2Ndf < 5 && HT_Scifi_Chi2Ndf > 0)"
    #                  )
    rdf = rdf.Filter("HT_Scifi_flag == 1")
    
       
    return rdf, feature_chain, integrated_lumi
   

def plot_every_branch(chain):
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.SetDefaultSumw2(True)

    outdir = "./plots/branches/"
    os.makedirs(outdir, exist_ok=True)

    # Accept only these scalar numeric leaf types
    numeric_types = {
        "Float_t", "Double_t",
        "Int_t", "UInt_t", "Short_t", "UShort_t",
        "Long64_t", "ULong64_t",
        "Char_t", "UChar_t", "Bool_t"
    }

    leaves = list(chain.GetListOfLeaves())  # TObjArray -> Python list
    print(f"Found {len(leaves)} leaves (will skip non-numeric / complex).")

    for leaf in leaves:
        lname = leaf.GetName()
        ltype = leaf.GetTypeName() or ""

        # Skip non-numeric and complex objects (e.g., vectors/classes) — adjust if you want vector support
        if ltype not in numeric_types:
            # You can relax this by allowing vector<...> and handling element-wise later if needed.
            # if ltype.startswith("vector<"):
            #     # handle vector case here
            #     pass
            print(f"  Skipping {lname} (type {ltype})")
            continue

        print(f"→ Drawing {lname} (type {ltype}, excluding -999)")

        # Unique names per iteration to avoid clashes & deletion messages
        uid = uuid.uuid4().hex[:8]
        canvas_name = f"c_{lname}_{uid}"
        hist_name   = f"h_{lname}_{uid}"

        # Selection to drop -999 values
        selection = f"{lname} != -999"

        # Create a throwaway canvas
        c = ROOT.TCanvas(canvas_name, "", 800, 600)

        try:
            # Draw with automatic 100 bins; you can tune or compute (nbins, min, max) if desired
            draw_expr = f"{lname}>>{hist_name}(100)"
            nevt = chain.Draw(draw_expr, selection, "goff")

            h = ROOT.gDirectory.Get(hist_name)
            if not h or h.GetEntries() == 0 or nevt <= 0:
                print(f"  Skipping {lname} (no valid entries after excluding -999)")
                c.Close()
                continue

            # Detach from any directory to avoid accidental deletion
            h.SetDirectory(0)
            h.SetLineWidth(2)
            h.SetTitle(f"{lname} (values != -999);{lname};Entries")

            # Nice axes range (optional): ensure not all entries are at one bin edge
            h.Draw("HIST")
            outpath = os.path.join(outdir, f"{lname}.png")
            c.SaveAs(outpath)

        except Exception as e:
            # Catch PyROOT/C++ exceptions; true segfaults won’t be catchable, but many issues are.
            print(f"  Error drawing {lname}: {e}")
        finally:
            c.Close()

    print(f"✅ Finished. Plots in: {outdir}")
     
    
def main():
    
    beam_type = "Down"
    hist_name="HT_Scifi_startY"
    n_bins, x_min, x_max, axis_title, logy = hist_info[hist_name]
    
    muonDown_metadata = pd.read_csv("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_muon_down_metadata_recoTrks.csv")
    
    rdf, chain, integrated_lumi = read_rdf(muonDown_metadata)
    
    #plot_every_branch(chain)
    
    h_proxy_true = rdf.Histo1D(
            (f"h_muon_true_{beam_type}_{hist_name}", "", int(n_bins), float(x_min), float(x_max)),
            hist_name,
            "fluka_weight" 
        )
    h = h_proxy_true.GetValue().Clone()
    h.SetDirectory(0)
    h.GetXaxis().SetTitle(axis_title)
    h.GetYaxis().SetTitle("Expected Events")
    
    
    normalise_lumi = 0.337
    scale = float(normalise_lumi) / float(integrated_lumi) 
    
    #integral = h.Integral()
    #h.Scale(1.0/integral)
    h.Scale(scale)
    
    print(f"integrated_lumi: {integrated_lumi}")
    print(f"scale =  {normalise_lumi} * {1/integrated_lumi}")
    
    
    h.SetLineColor(ROOT.kBlue + 1)
    h.SetMarkerColor(ROOT.kBlue + 1)
    
    outdir = f"./plots/{hist_name}"
    os.makedirs(outdir, exist_ok=True)
    out_pdf=os.path.join(outdir, f"MC_muon_down_{hist_name}_noScale.pdf")
    

    
    c = ROOT.TCanvas(f"c_{axis_title}", axis_title, 900, 750)
    h.Draw("E1")
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.040)  # smaller than 0.045
    label.SetTextAlign(31)
    label.DrawLatex(0.88, 0.94, f"#int #font[12]{{L}} dt = {normalise_lumi:.3f} fb^{{-1}}")
    label.DrawLatex(0.20, 0.94, "")
    
    c.Update()
    c.Print(out_pdf)
    print(f"[OK] wrote {out_pdf}")



hist_info = {
    "signed_slope_x": (60, -3, 3, 'Shower Direction X', True),
    "signed_slope_y": (60, -3, 3, 'Shower Direction Y', True),
    "sum_hit_density": (70, 0, 7e4, 'Sum of Density Weight',True),
    'count_scifi':  (100, 0, 1000, 'SciFi Hit Total Count', True),
    'count_us1':  (13, 0, 13, 'US1 Hit Count', True),
    "start_centroid_x": (90, -70, 20, 'Start Centroid X', True),
    "start_centroid_y": (80, 0, 80, 'Start Centroid Y', True),
    "centroid_slope_y": (60, -3, 3, 'Centroid Slope y', True),
    "centroid_slope_x": (60, -3, 3, 'Centroid Slope x', True),
    "start_avgPos_x": (90, -70, 20, 'Start AvgPos X', True),
    "start_avgPos_y": (80, 0, 80, 'Start AvgPos Y', True),
    "avgPos_slope_x": (60,-3, 3, 'Centroid Slope X', True),
    "avgPos_slope_y": (60,-3, 3, 'Centroid Slope Y', True),
    "avg_scifi1_y": (70, 0, 70, 'Scifi1 AvgPos Y', False),
    "avg_scifi2_y": (70, 0, 70, 'Scifi2 AvgPos Y', False),
    "avg_scifi3_y": (70, 0, 70, 'Scifi3 AvgPos Y', False),
    "avg_scifi4_y": (70, 0, 70, 'Scifi4 AvgPos Y', False),
    "avg_scifi5_y": (70, 0, 70, 'Scifi5 AvgPos Y', False),
    
    "HT_Scifi_startY": (70, 0, 70, 'Y Pos', False),

}
    


if __name__ == "__main__":
    main()