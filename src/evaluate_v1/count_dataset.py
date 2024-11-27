
import pandas as pd
import glob
import ROOT
from tqdm import tqdm
def read_and_sum_stage2(file_path):
    """Function to read a file and sum the 'stage2' variable using ROOT."""
    print('reading', file_path)
    df = ROOT.RDataFrame("cbmsim", file_path)
    stage2_count = df.Sum("stage2").GetValue()
    return stage2_count
def read_stage2():
    output_file = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/converted_dataset_v2.csv'
    file_list_dir = '/eos/user/z/zhibin/sndData/converted/'
    dataset_name = 'prediction'
    file = f'{file_list_dir}/{dataset_name}_files.csv'
    file_list_df = pd.read_csv(file, header=None, names=['path', 'n_event'])
    with open(output_file, 'w') as f:
        # Write the header of the output CSV file
        f.write('path,n_event,stage2count\n')
        # Process each row in DataFrame
        for index, row in tqdm(file_list_df.iterrows(), total=file_list_df.shape[0], desc="Processing files"):
            stage2_count = read_and_sum_stage2(row['path'])
            # Write the result immediately to the output CSV file
            f.write(f"{row['path']},{row['n_event']},{stage2_count}\n")



def count_event():
    data_neutron = {
    "partition": ["neutrons_5_10", "neutrons_10_20", "neutrons_20_30", "neutrons_30_40", "neutrons_40_50", "neutrons_50_60", "neutrons_60_70", "neutrons_70_80", "neutrons_80_90",'neutrons_90_100'],
    "intRate": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48, 0],
    "paper_generated": [2.12e6, 2.05e6, 7.54e5, 7.41e5, 7.37e4, 3.35e5, 3.33e5, 3.24e5, 1.17e5, 3.22e5]
    }   
    data_kaons = {
        "partition": ["kaons_5_10", "kaons_10_20", "kaons_20_30", "kaons_30_40", "kaons_40_50", "kaons_50_60", "kaons_60_70", "kaons_70_80", "kaons_80_90", "kaons_90_100"],
        "intRate": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1, 1.15e1],
        "paper_generated": [2.14e6, 2.09e6, 7.49e5, 7.53e5, 7.43e5, 3.41e5, 3.34e5, 3.35e5, 3.17e5, 3.30e5]
    }
    data_neutrino = {
        "partition": ["neutrino"],
        "intRate": [157],
        "paper_generated": [1.0e9]
    }
    # Convert dictionaries to pandas DataFrames
    df_neutron = pd.DataFrame(data_neutron)
    df_kaons = pd.DataFrame(data_kaons)
    df_neutrino = pd.DataFrame(data_neutrino)

    weight_df = pd.concat([df_neutron, df_kaons, df_neutrino], ignore_index=True)

    print("Combined Data:")
    print(weight_df)
    file_list_dir = '/eos/user/z/zhibin/sndData/converted/'
    # Map these patterns to dataset names
    #dataset_names = ['train', 'validation', 'prediction']
    dataset_names = ['prediction']
    for dataset_name in dataset_names:
        file_list_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/stage2'
        csv_files = glob.glob(f"{file_list_dir}/*.csv")
        # Loop through the list of files and read each into a DataFrame
        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            dataframes.append(df)
        #print(dataframes)
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
        weight_df[dataset_name] = weight_df['partition'].apply(
            lambda x: file_list_df[file_list_df['keyword'] == x.lower()]['n_event'].sum()
            )
        weight_df[f'stage2_{dataset_name}'] = weight_df['partition'].apply(
        lambda x: file_list_df[file_list_df['keyword'] == x.lower()]['stage2count'].sum()
        )
        weight_df[f'cutEff_{dataset_name}'] = weight_df[f'stage2_{dataset_name}']/weight_df[dataset_name]

    #weight_df['converted'] = weight_df['train'] + weight_df['validation'] + weight_df['prediction']

    output_file_name = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/converted_dataset_v2.csv'
    weight_df.to_csv(output_file_name, index=False)
    print(f"Saved DataFrame to {output_file_name}")

if __name__ == "__main__":
    count_event()

