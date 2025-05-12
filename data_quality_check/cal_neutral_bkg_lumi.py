import pandas as pd
import re


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

# check total events accorss energy
def check_generated_events(kaon_metadata_path, neutron_metadata_path):
    # Read CSVs into DataFrames
    kaon_df = pd.read_csv(kaon_metadata_path)
    neutron_df = pd.read_csv(neutron_metadata_path)

    # Group by 'subfolder' and sum 'n_event' for each
    kaon_grouped = kaon_df.groupby('subfolder')['n_event'].sum().reset_index(name='kaon_total_events')
    neutron_grouped = neutron_df.groupby('subfolder')['n_event'].sum().reset_index(name='neutron_total_events')

    kaon_grouped['kaon_total_events'] = kaon_grouped['kaon_total_events'].apply(lambda x: f"{x:.2e}")
    neutron_grouped['neutron_total_events'] = neutron_grouped['neutron_total_events'].apply(lambda x: f"{x:.2e}")

    # Print the results
    print("Kaon Events by Subfolder:")
    print(kaon_grouped.to_string(index=False))

    print("\nNeutron Events by Subfolder:")
    print(neutron_grouped.to_string(index=False))

def extract_energy(subfolder):
    match = re.search(r'[_/]([a-zA-Z]+)_(\d+)_?(\d+)?', subfolder)
    if match:
        low = int(match.group(2))
        high = int(match.group(3)) if match.group(3) else None
        return (low, high)
    return None

def cal_lumi_for_neutral_bkg(int_rate_df, metadata_path):
    metadata_df = pd.read_csv(metadata_path)
    metadata_df["energy_range"] = metadata_df["subfolder"].apply(extract_energy)

    print(metadata_df)

    rate_dict = dict(zip(int_rate_df["Energy [GeV]"], int_rate_df["Rate per fb^-1"]))
    print(rate_dict)

    def compute_lumi(row):
        energy_range = row["energy_range"]
        if energy_range is None:
            return None
        energy_str = f"({energy_range[0]},{energy_range[1]})"
        rate = rate_dict.get(energy_str)
        n_event = row.get("n_event")
        print("energy_range",energy_str,"rate",rate, "n_event", n_event)
        if rate == 0 or rate is None:
            return None
        return n_event / rate

    metadata_df["lumi_fb^-1"] = metadata_df.apply(compute_lumi, axis=1)

    print(metadata_df)
    metadata_df.to_csv("test_data/calculated_luminosity.csv", index=False)
    return metadata_df
    # loop over metadata_df
        # get Rate per fb^-1 for the energy range of the row
        # if rate = 0, lumi =None
        #define lumi = n_event/rate 


if __name__ == "__main__":
    #main()
    kaon_metadata_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_kaon_FTFP_BERT_metadata.csv'
    neutron_metadata_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutron_FTFP_BERT_metadata.csv'
    #check_generated_events(kaon_metadata_path, neutron_metadata_path)
    neutron_rates, kaon_rates = get_int_rate()
    print("neutron_rates")
    print(neutron_rates)
    print("kaon_rates")
    print(kaon_rates)

    cal_lumi_for_neutral_bkg(neutron_rates, neutron_metadata_path)
