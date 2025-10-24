import os
import ROOT
import pandas as pd

ROOT.ROOT.EnableImplicitMT()
# read metadata
# read vetoTagged feature
# apply veto cut


def load_metadata_files(file_list, root_path):
    loaded_data = {}
    for fname in file_list:
        var_name = fname.replace("_metadata.csv", "").replace("-", "_").replace(".", "_")
        full_path = os.path.join(root_path, fname)
        loaded_data[var_name] = pd.read_csv(full_path)
    return loaded_data



def read_rdf(metadata_df, file_count_max = 200):
    feature_chain = ROOT.TChain("sndData")
    int_lumi = 0

    file_count = 0
    for i, row in metadata_df.iterrows():
        sub = row['subfolder']

        # ---- vetoFree ----
        vf_feat = row['vetoTagged_feature_path']

        if os.path.exists(str(vf_feat)):
            f = ROOT.TFile.Open(vf_feat)
            t = f.Get("sndData")
            if not t or not t.GetListOfBranches().FindObject("vetoHitTime_earlist_veto1"):
                print(f'file no branch vetoHitTime_earlist_veto1')
                f.Close()
                continue
            
            feature_chain.Add(vf_feat)
            file_count += 1
            if pd.notna(row['lumi_per_file']):
                int_lumi += row['lumi_per_file']

        if file_count > file_count_max:
            break
        
    if (int_lumi==0):
        return None, None, 0
    rdf = ROOT.RDataFrame(feature_chain)
        
    return rdf, feature_chain, int_lumi

def cutflow_counts(rdf, cuts: dict):
    """
    Given an RDF and dict {cut_name: expr}, return {cut_name: count}.
    Includes 'total' as the uncut count.
    """
    counts = {}
    counts["total"] = int(rdf.Count().GetValue())
    for name, expr in cuts.items():
        counts[name] = int(rdf.Filter(expr).Count().GetValue())
    return counts

def main():
    mc_files = [
        "MC_kaon_FTFP_BERT_metadata_subset.csv",
        "MC_neutron_FTFP_BERT_metadata_subset.csv",
        "MC_muon_up_metadata.csv",
        "MC_neutrino_volTarget_100fb-1_metadata.csv",
        "real_data_2024_skim_runs_metadata.csv",
    ]
    
    root_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated'
    metadata_vars = load_metadata_files(mc_files, root_path)
    
    # read data
    print('processing read data')
    data_df = metadata_vars["real_data_2024_skim_runs"]    
    data_rdf, data_chain, data_int_lumi = read_rdf(data_df, 10)
    
    data_rdf = data_rdf.Filter("vetoHitTime_earlist>0 && vetoHitTime_latest>0")
    
    #plot these hist, set range 0,25ns, save to ./plots/
    # vetoHitTime_earlist, 
    # vetoHitTime_latest
    # vetoHitTime_earlist_veto1, 
    # vetoHitTime_latest_veto1, 
    # vetoHitTime_earlist_veto2, 
    # vetoHitTime_latest_veto2
    # vetoHitTime_earlist_veto3, 
    # vetoHitTime_latest_veto3
    
    # Ensure output directory exists
    os.makedirs("plots", exist_ok=True)

    # List of variables to plot
    vars_to_plot = [
        "vetoHitTime_earlist",
        "vetoHitTime_latest",
        "vetoHitTime_earlist_veto1",
        "vetoHitTime_latest_veto1",
        "vetoHitTime_earlist_veto2",
        "vetoHitTime_latest_veto2",
        "vetoHitTime_earlist_veto3",
        "vetoHitTime_latest_veto3",
    ]

    # Define histogram range and binning
    nbins, xmin, xmax = 100, 0, 25  # 0–25 ns

    # Loop and create histograms
    for var in vars_to_plot:
        hist = data_rdf.Histo1D(
            (f"h_{var}", f"{var}; {var} [ns]; Events", nbins, xmin, xmax), var
        )
        
        # Draw and save
        canvas = ROOT.TCanvas(f"c_{var}", var, 800, 600)
        hist.Draw()
        canvas.SaveAs(f"plots/{var}.png")

    print("✅ All histograms saved to ./plots/")
    

    
    
    
    
if __name__ == "__main__":
    main()