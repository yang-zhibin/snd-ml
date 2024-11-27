import pandas as pd
import glob
import ROOT

def count_event():
    data_neutron = {
    "partition": ["neutrons_5_10", "neutrons_10_20", "neutrons_20_30", "neutrons_30_40", "neutrons_40_50", "neutrons_50_60", "neutrons_60_70", "neutrons_70_80", "neutrons_80_90",'neutrons_90_100'],
    "intRate": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48, 0],
    "generated": [2.12e6, 2.05e6, 7.54e5, 7.41e5, 7.37e4, 3.35e5, 3.33e5, 3.24e5, 1.17e5, 3.22e5]
    }   
    data_kaons = {
        "partition": ["kaons_5_10", "kaons_10_20", "kaons_20_30", "kaons_30_40", "kaons_40_50", "kaons_50_60", "kaons_60_70", "kaons_70_80", "kaons_80_90", "kaons_90_100"],
        "intRate": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1, 1.15e1],
        "generated": [2.14e6, 2.09e6, 7.49e5, 7.53e5, 7.43e5, 3.41e5, 3.34e5, 3.35e5, 3.17e5, 3.30e5]
    }
    data_neutrino = {
        "partition": ["neutrino"],
        "intRate": [157],
        "generated": [1.0e9]
    }
    # Convert dictionaries to pandas DataFrames
    df_neutron = pd.DataFrame(data_neutron)
    df_kaons = pd.DataFrame(data_kaons)
    df_neutrino = pd.DataFrame(data_neutrino)

    weight_df = pd.concat([df_neutron, df_kaons, df_neutrino], ignore_index=True)

    print("Combined Data:")
    print(weight_df)
    file_list_dir = '/eos/user/z/zhibin/sndData/converted/'
    csv_files = glob.glob(f"{file_list_dir}/val*.csv")
    # Loop through the list of files and read each into a DataFrame
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file,header=None,names=['path', 'n_event'] )
        dataframes.append(df)
    
    # Concatenate all the DataFrames into one
    file_list_df = pd.concat(dataframes, ignore_index=True)
    print(file_list_df)
    def extract_keyword(path):
        keywords = weight_df['partition'].str.lower().tolist()
        for keyword in keywords:
            if keyword in path.lower():
                return keyword
        return None

    file_list_df['keyword'] = file_list_df['path'].apply(extract_keyword)
    weight_df['sum_n_event'] = weight_df['partition'].apply(
        lambda x: file_list_df[file_list_df['keyword'] == x.lower()]['n_event'].sum()
        )
    print(weight_df['sum_n_event'])


def main():
    data_neutron = {
    "partition": ["neutrons_5_10", "neutrons_10_20", "neutrons_20_30", "neutrons_30_40", "neutrons_40_50", "neutrons_50_60", "neutrons_60_70", "neutrons_70_80", "neutrons_80_90"],
    "intRate": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48],
    "generated": [2.12e6, 2.05e6, 7.54e5, 7.41e5, 7.37e4, 3.35e5, 3.33e5, 3.24e5, 1.17e5]
    }   
    data_kaons = {
        "partition": ["kaons_5_10", "kaons_10_20", "kaons_20_30", "kaons_30_40", "kaons_40_50", "kaons_50_60", "kaons_60_70", "kaons_70_80", "kaons_80_90"],
        "intRate": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1],
        "generated": [2.14e6, 2.09e6, 7.49e5, 7.53e5, 7.43e5, 3.41e5, 3.34e5, 3.35e5, 3.17e5]
    }
    data_neutrino = {
        "partition": ["neutrino"],
        "intRate": [157],
        "generated": [1.0e9]
    }
    # Convert dictionaries to pandas DataFrames
    df_neutron = pd.DataFrame(data_neutron)
    df_kaons = pd.DataFrame(data_kaons)
    df_neutrino = pd.DataFrame(data_neutrino)

    weight_df = pd.concat([df_neutron, df_kaons, df_neutrino], ignore_index=True)

    print("Combined Data:")
    print(weight_df)


    file_list_dir = '/eos/user/z/zhibin/sndData/converted/'
    csv_files = glob.glob(f"{file_list_dir}/train*.csv")
    # Loop through the list of files and read each into a DataFrame
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file,header=None,names=['path', 'n_event'] )
        dataframes.append(df)
    
    # Concatenate all the DataFrames into one
    file_list_df = pd.concat(dataframes, ignore_index=True)
    print(file_list_df)

    def extract_keyword(path):
        keywords = weight_df['partition'].str.lower().tolist()
        for keyword in keywords:
            if keyword in path.lower():
                return keyword
        return None

    file_list_df['keyword'] = file_list_df['path'].apply(extract_keyword)
    weight_df['sum_n_event'] = weight_df['partition'].apply(
        lambda x: file_list_df[file_list_df['keyword'] == x.lower()]['n_event'].sum()
        )
    weight_df['scale'] = 1000
    #print(weight_df.loc[weight_df[weight_df['partition'] == 'partition'].index]['scale'])
    weight_df.iloc[-1,-1] *= 100

    weight_df['weight'] = weight_df['intRate']/weight_df['sum_n_event'] * weight_df['scale']
    
    print(weight_df)   
    print("sum of weight:")
    print("neutron:{}".format(weight_df['weight'][0:9].sum()))
    print("kaon:{}".format(weight_df['weight'][9:18].sum()))
    print("neutrino:{}".format(weight_df['weight'][18:].sum()))

    weight_df.to_csv('/afs/cern.ch/user/z/zhibin/work/snd-ml/src/dataset/trianing_weight.csv')

def read_and_sum_stage2(file_path):
    """Function to read a file and sum the 'stage2' variable using ROOT."""
    df = ROOT.RDataFrame("cbmsim", file_path)
    stage2_count = df.Sum("stage2").GetValue()
    print('reading', file_path)
    return stage2_count

def val_eff():
    data_neutron = {
    "partition": ["neutrons_5_10", "neutrons_10_20", "neutrons_20_30", "neutrons_30_40", "neutrons_40_50", "neutrons_50_60", "neutrons_60_70", "neutrons_70_80", "neutrons_80_90",'neutrons_90_100'],
    "intRate": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48, 1],
    "generated": [2.12e6, 2.05e6, 7.54e5, 7.41e5, 7.37e4, 3.35e5, 3.33e5, 3.24e5, 1.17e5, 1]
    }   
    data_kaons = {
        "partition": ["kaons_5_10", "kaons_10_20", "kaons_20_30", "kaons_30_40", "kaons_40_50", "kaons_50_60", "kaons_60_70", "kaons_70_80", "kaons_80_90", "kaons_90_100"],
        "intRate": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1,1],
        "generated": [2.14e6, 2.09e6, 7.49e5, 7.53e5, 7.43e5, 3.41e5, 3.34e5, 3.35e5, 3.17e5,1]
    }
    data_neutrino = {
        "partition": ["neutrino"],
        "intRate": [1],
        "generated": [1.0e9]
    }
    # Convert dictionaries to pandas DataFrames
    df_neutron = pd.DataFrame(data_neutron)
    df_kaons = pd.DataFrame(data_kaons)
    df_neutrino = pd.DataFrame(data_neutrino)

    weight_df = pd.concat([df_neutron, df_kaons, df_neutrino], ignore_index=True)

    print("Combined Data:")
    print(weight_df)


    file_list_dir = '/eos/user/z/zhibin/sndData/converted/'
    csv_files = glob.glob(f"{file_list_dir}/pred*.csv")
    # Loop through the list of files and read each into a DataFrame
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file,header=None,names=['path', 'n_event'] )
        dataframes.append(df)
    
    # Concatenate all the DataFrames into one
    file_list_df = pd.concat(dataframes, ignore_index=True)
    file_list_df['stage2count'] = file_list_df['path'].apply(read_and_sum_stage2)

    print(file_list_df)

    def extract_keyword(path):
        keywords = weight_df['partition'].str.lower().tolist()
        for keyword in keywords:
            if keyword in path.lower():
                return keyword
        return None

    file_list_df['keyword'] = file_list_df['path'].apply(extract_keyword)
    weight_df['sum_n_event'] = weight_df['partition'].apply(
        lambda x: file_list_df[file_list_df['keyword'] == x.lower()]['n_event'].sum()
        )
    weight_df['stage2'] = weight_df['partition'].apply(
        lambda x: file_list_df[file_list_df['keyword'] == x.lower()]['stage2count'].sum()
        )
    weight_df['cut_eff'] = weight_df['stage2']/weight_df['sum_n_event']
    weight_df['scale'] = 1000
    #print(weight_df.loc[weight_df[weight_df['partition'] == 'partition'].index]['scale'])


    weight_df['weight'] = weight_df['intRate']/weight_df['sum_n_event'] * weight_df['scale']

    weight_df['particle_type'] = weight_df['partition'].apply(lambda x: x.split('_')[0])
    total_weights = weight_df.groupby('particle_type')['weight'].sum()
    weight_df['normalized_weight'] = weight_df.apply(lambda x: x['weight'] / total_weights[x['particle_type']], axis=1)

    print(weight_df)   
    print("sum of weight:")
    print("neutron:{}".format(weight_df['normalized_weight'][0:9].sum()))
    print("kaon:{}".format(weight_df['normalized_weight'][9:18].sum()))
    print("neutrino:{}".format(weight_df['normalized_weight'][18:].sum()))

    weight_df.to_csv('/afs/cern.ch/user/z/zhibin/work/snd-ml/src/dataset/trianing_weight_v2.csv')

if __name__ == "__main__":
    #main()
    #count_event()
    val_eff()