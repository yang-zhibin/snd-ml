import ROOT
import pandas as pd


# my method to calculte rate
def get_int_rate():
    # from paper CERN-SND@LHC-NOTE-2023-002
    # Neutron yield data
    neutron_yield_FTFP_BERT = {
        "Energy [GeV]": [
            "(5,10)", "(10,20)", "(20,30)", "(30,40)", "(40,50)", "(50,60)", "(60,70)",
            "(70,80)", "(80,90)", "(90,100)", "(100,150)", "(150,200)"
        ],
        "Int. Rate": [
            "4.62e+04", "7.59e+03", "1.18e+03", "5.30e+02", "4.66e+02", "2.60e+01",
            "1.80e+01", "8.48", "8.48", "0", "0", "0"
        ]
    }
    neutron_yield_FTFP_BERT_2022 = pd.DataFrame(neutron_yield_FTFP_BERT)

    # Kaon yield data
    kaon_yield_FTFP_BERT = {
        "Energy [GeV]": [
            "(5,10)", "(10,20)", "(20,30)", "(30,40)", "(40,50)", "(50,60)", "(60,70)",
            "(70,80)", "(80,90)", "(90,100)", "(100,150)", "(150,200)"
        ],
        "Int. Rate": [
            "2.51e+04", "5.72e+03", "8.53e+02", "1.10e+02", "9.38e+01", "6.48e+01",
            "9.90e+00", "2.32e+01", "1.15e+01", "1.15e+01", "0", "0"
        ]
    }
    kaon_yield_FTFP_BERT_2022 = pd.DataFrame(kaon_yield_FTFP_BERT)

    # Luminosity in fb^-1
    lumi_2022 = 36.77

    # Convert to float
    neutron_yield_FTFP_BERT_2022["Int. Rate"] = neutron_yield_FTFP_BERT_2022["Int. Rate"].astype(float)
    kaon_yield_FTFP_BERT_2022["Int. Rate"] = kaon_yield_FTFP_BERT_2022["Int. Rate"].astype(float)

    # Create normalized DataFrames
    neutron_yield_FTFP_BERT_per_fb = pd.DataFrame({
        "Energy [GeV]": neutron_yield_FTFP_BERT_2022["Energy [GeV]"],
        "Rate per fb^-1": neutron_yield_FTFP_BERT_2022["Int. Rate"] / lumi_2022
    })

    kaon_yield_FTFP_BERT_per_fb = pd.DataFrame({
        "Energy [GeV]": kaon_yield_FTFP_BERT_2022["Energy [GeV]"],
        "Rate per fb^-1": kaon_yield_FTFP_BERT_2022["Int. Rate"] / lumi_2022
    })

    return neutron_yield_FTFP_BERT_per_fb, kaon_yield_FTFP_BERT_per_fb

# cris's method
def get_int_rate_from_cris():
    bins = [[0, 10],
        [10, 20],
        [20, 30],
        [30, 40],
        [40, 50],
        [50, 60],
        [60, 70],
        [70, 80],
        [80, 90],
        [90, 100]]
    calculated_lumi = 1e-5*4
    f = ROOT.TFile("/eos/experiment/sndlhc/users/dancc/MuonDIS/NeutralsExpect/120423/7977358.MC-DISneutrals_.root")
    denominator_bin = 2
    samples = ["Kaons", "neutrons"]
    interaction_rates = {}
    for i_sample in samples :
        interaction_rates[i_sample] = []
    include_KS = True
    for bin_low, bin_high in bins :
        
        if bin_low == 0. :
            bin_low = 5.
        
        # Kaons
        low_bin = f.Energy_neutrals_noVeto_maxentrack_K_L0.GetXaxis().FindBin(bin_low)
        up_bin = f.Energy_neutrals_noVeto_maxentrack_K_L0.GetXaxis().FindBin(bin_high)
        
    
        interaction_rates["Kaons"].append(f.Energy_neutrals_noVeto_maxentrack_K_L0.Integral(low_bin, up_bin)/calculated_lumi)
        print(bin_low, bin_high, low_bin, up_bin, f.Energy_neutrals_noVeto_maxentrack_K_L0.Integral(low_bin, up_bin)/calculated_lumi)
#        interaction_rates["Kaons_FTFP_BERT"].append(f.Energy_neutrals_noVeto_maxentrack_K_L0.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)

        if include_KS :
            low_bin = f.Energy_neutrals_noVeto_maxentrack_K_S0.GetXaxis().FindBin(bin_low)
            up_bin = f.Energy_neutrals_noVeto_maxentrack_K_S0.GetXaxis().FindBin(bin_high)
            interaction_rates["Kaons"][-1] += (f.Energy_neutrals_noVeto_maxentrack_K_S0.Integral(low_bin, up_bin)/calculated_lumi)
#            interaction_rates["Kaons_FTFP_BERT"][-1] += (f.Energy_neutrals_noVeto_maxentrack_K_S0.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)
            
        # neutrons
        low_bin = f.Energy_neutrals_noVeto_maxentrack_neutron.GetXaxis().FindBin(bin_low)
        up_bin = f.Energy_neutrals_noVeto_maxentrack_neutron.GetXaxis().FindBin(bin_high)
        interaction_rates["neutrons"].append(f.Energy_neutrals_noVeto_maxentrack_neutron.Integral(low_bin, up_bin)/calculated_lumi)
#        interaction_rates["neutrons_FTFP_BERT"].append(f.Energy_neutrals_noVeto_maxentrack_neutron.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)
 
        low_bin = f.Energy_neutrals_noVeto_maxentrack_antineutron.GetXaxis().FindBin(bin_low)
        up_bin = f.Energy_neutrals_noVeto_maxentrack_antineutron.GetXaxis().FindBin(bin_high)
        interaction_rates["neutrons"][-1] += (f.Energy_neutrals_noVeto_maxentrack_antineutron.Integral(low_bin, up_bin)/calculated_lumi)
#        interaction_rates["neutrons_FTFP_BERT"][-1] += (f.Energy_neutrals_noVeto_maxentrack_antineutron.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)

    return interaction_rates
def main():
    neutron_rate, kaon_rate = get_int_rate()
    interaction_rates = get_int_rate_from_cris()
    
    print(neutron_rate)
    print(kaon_rate)
    print(interaction_rates)
    
    df_neutron = pd.DataFrame(neutron_rate)
    df_kaon = pd.DataFrame(kaon_rate)

    # Merge for comparison
    comparison = pd.DataFrame({
        "Energy [GeV]": df_neutron["Energy [GeV]"],
        "Neutron (from paper)": df_neutron["Rate per fb^-1"],
        "Neutron (from CRIS)": interaction_rates["neutrons"] + [None] * (len(df_neutron) - len(interaction_rates["neutrons"])),
        "Kaon (from paper)": df_kaon["Rate per fb^-1"],
        "Kaon (from CRIS)": interaction_rates["Kaons"] + [None] * (len(df_kaon) - len(interaction_rates["Kaons"]))
    })
    
    print(comparison)
    
    
    
    
if __name__ == "__main__":
    main()
