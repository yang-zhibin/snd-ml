import ROOT
import pandas as pd


def read_data():
    metadata_mc_neutrino_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'
    metadata_mc_neutrino_df = pd.read_csv(metadata_mc_neutrino_path)

    feature_chain = ROOT.TChain("snddata")

    count=0
    for index, row in metadata_mc_neutrino_df.iterrows():
        
        feature_path = row['feature_path']
        print(feature_path)
        feature_chain.Add(feature_path)

    
        
        count+=1
        if count>400:
            break
    print(count)
    lumi_mc = count*100
    lumi_2022 = 3.677e+01

    normalise_factor = lumi_2022/lumi_mc

    rdf = ROOT.RDataFrame(feature_chain)
    rdf = rdf.Define("ParticleType", """
        if (pdgCode == 12 || pdgCode == -12) return std::string("ve");
        else if (pdgCode == 14 || pdgCode == -14) return std::string("vm");
        else if (pdgCode == 16 || pdgCode == -16) return std::string("vt");
        else if (pdgCode == 112 || pdgCode == -112 || pdgCode == 114 || pdgCode == -114 || pdgCode == 116 || pdgCode == -116) return std::string("NC");
        else if (pdgCode == 130 || pdgCode == 310) return std::string("kaon");
        else if (pdgCode == 2112) return std::string("neutron");
        else if (pdgCode == 13 || pdgCode == -13) return std::string("muon");
        else if (pdgCode == 0 ) return std::string("real_data");
        else return std::string("others");
        """)
        #

    branch_values = rdf.AsNumpy(["ParticleType"])["ParticleType"]
    unique_classes = set(branch_values)

    # Now count entries for each class
    print(f"MC lumi {lumi_mc} fb-1")
    for cls in unique_classes:
        count = rdf.Filter(f'ParticleType == "{cls}"').Count().GetValue()
        print(f"    '{cls}': {count} entries")

    print(f"MC lumi {lumi_mc} fb-1, after normalisation to 2022 {lumi_2022} fb-1")
    for cls in unique_classes:
        count = (rdf.Filter(f'ParticleType == "{cls}"').Count().GetValue())*normalise_factor
        print(f"    '{cls}': {count} entries")



if __name__ == "__main__":
    read_data()