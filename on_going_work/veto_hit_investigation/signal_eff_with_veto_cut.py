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
    #print(metadata_vars)
    
    particle_types = ["ve", "kaon", "neutron", "data"]
    cut_names = [
        "int_lumi",
        "total",
        "vetoHitTime_earlist_veto1>2",
        "vetoHitTime_earlist>2",
        "vetoHitTime_earlist_veto1>5",
        "vetoHitTime_earlist>5",
        "vetoHitTime_earlist_veto1>10",
        "vetoHitTime_earlist>10",
    ]
    cuts = {
            "vetoHitTime_earlist_veto1>2":  "vetoHitTime_earlist_veto1 > 2",
            "vetoHitTime_earlist>2":        "vetoHitTime_earlist > 2",
            "vetoHitTime_earlist_veto1>5":  "vetoHitTime_earlist_veto1 > 5",
            "vetoHitTime_earlist>5":        "vetoHitTime_earlist > 5",
            "vetoHitTime_earlist_veto1>10": "vetoHitTime_earlist_veto1 > 10",
            "vetoHitTime_earlist>10":       "vetoHitTime_earlist > 10",
        }
    
    cuts_2024 = {
            "vetoHitTime_earlist_veto1>2":  "vetoHitTime_earlist_veto2 > 2",
            "vetoHitTime_earlist>2":        "vetoHitTime_earlist > 2",
            "vetoHitTime_earlist_veto1>5":  "vetoHitTime_earlist_veto2 > 5",
            "vetoHitTime_earlist>5":        "vetoHitTime_earlist > 5",
            "vetoHitTime_earlist_veto1>10": "vetoHitTime_earlist_veto2 > 10",
            "vetoHitTime_earlist>10":       "vetoHitTime_earlist > 10",
    }
    cutflow_df = pd.DataFrame(0, index=cut_names, columns=particle_types, dtype=float)
    print(cutflow_df)
    
    print('processing neutrino MC')
    # read neutrino MC
    neutrino_df = metadata_vars["MC_neutrino_volTarget_100fb_1"]    
    neutrino_rdf, neutrino_chain, neutrino_int_lumi = read_rdf(neutrino_df)
    neutrino_rdf = neutrino_rdf.Filter("pdgCode == 12 || pdgCode == -12 ")
    if neutrino_rdf is not None:
        counts = cutflow_counts(neutrino_rdf, cuts)
        cutflow_df.loc["total", "ve"] = counts["total"]
        for cname in cuts.keys():
            cutflow_df.loc[cname, "ve"] = counts[cname]

        cutflow_df.loc["int_lumi", "ve"] = neutrino_int_lumi
    
    
    # read data
    print('processing read data')
    data_df = metadata_vars["real_data_2024_skim_runs"]    
    data_rdf, data_chain, data_int_lumi = read_rdf(data_df, 20000)

    if data_rdf is not None:
        
        counts = cutflow_counts(data_rdf, cuts_2024)
        cutflow_df.loc["total", "data"] = counts["total"]
        for cname in cuts.keys():
            cutflow_df.loc[cname, "data"] = counts[cname]

        cutflow_df.loc["int_lumi", "data"] = data_int_lumi
    
    
    
    # read kaon
    print('processing kaon')
    kaon_df = metadata_vars['MC_kaon_FTFP_BERT_metadata_subset_csv']
    ranges = sorted(kaon_df['energy_range'].unique(), key=lambda x: (str(type(x)), x))
    cutflow_df.loc["int_lumi", "kaon"] = 1
    for erange in ranges:
        sub_df = kaon_df[kaon_df['energy_range'] == erange]
        rdf, _, int_lumi = read_rdf(sub_df)
        if not int_lumi:
            continue
        counts = cutflow_counts(rdf, cuts)
        cutflow_df.loc["total", "kaon"] += counts["total"] / int_lumi
        for cname in cuts.keys():
            cutflow_df.loc[cname, "kaon"] += counts[cname] / int_lumi

    # read neutron
    print('processing neutron')
    neutron_df = metadata_vars['MC_neutron_FTFP_BERT_metadata_subset_csv']
    ranges = sorted(neutron_df['energy_range'].unique(), key=lambda x: (str(type(x)), x))
    cutflow_df.loc["int_lumi", "neutron"] = 1
    for erange in ranges:
        sub_df = neutron_df[neutron_df['energy_range'] == erange]
        rdf, _, int_lumi = read_rdf(sub_df)
        if not int_lumi:
            continue
        counts = cutflow_counts(rdf, cuts)
        cutflow_df.loc["total", "neutron"] += counts["total"] / int_lumi
        for cname in cuts.keys():
            cutflow_df.loc[cname, "neutron"] += counts[cname] / int_lumi
        
        
    
    
    
    print(cutflow_df)
    output_dir = "./cutflow_results"
    os.makedirs(output_dir, exist_ok=True)
    cutflow_path = os.path.join(output_dir, "cutflow_summary.csv")
    cutflow_df.to_csv(cutflow_path)
    
    

    
    
    
    
if __name__ == "__main__":
    main()