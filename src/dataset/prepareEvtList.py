import uproot
import os
import csv
from itertools import islice
import pandas as pd
from tqdm import tqdm

def get_file_list(root_path):
    partition_list = []
    event_list = []
    file_list = []


    for dirpath, dirnames, filenames in tqdm(os.walk(root_path)):
        for filename in filenames:
            if filename.endswith('.root') and 'converted' in filename:
                
                file_path = os.path.join(dirpath, filename)
                path_parts = file_path.split(os.sep)
                partition = path_parts[-3]
                try:
                    with uproot.open(file_path) as Rfile:
                        tree = Rfile["cbmsim"]
                        n_evt = tree.num_entries
                    event_list.append(n_evt)
                    file_list.append(file_path)
                    partition_list.append(partition)
                except KeyError:
                    print(f"Error: 'cbmsim' tree not found in {file_path}.")
                except Exception as e:
                    print(f"Error opening {filename}: {e}")


    return partition_list, event_list, file_list

def prepareEvtList(root_path):

    partitions = [
       'real_muon' 
       # "muons",
       # "Neutrinos",
       # "kaons",
       # "neutrons",
    ]

    total_counts = {}

    for particle in partitions:
        particle_path = os.path.join(root_path, particle)
        print(f"preparing {particle}")
        
        partition_list, event_list, file_list = get_file_list(particle_path)
        data = {
            'partition': partition_list,
            'event': event_list,
            'file': file_list
        }

        df = pd.DataFrame(data)
        csv_outpath = f"{root_path}/{particle}/{particle}_evt_list.csv"
        print(csv_outpath)
        print("{} total events: {}".format(particle, df['event'].sum()))
        df.to_csv(csv_outpath)

# current dataset
'''preparing muons
    12it [00:01, 11.72it/s]
    /eos/user/z/zhibin/sndData/converted//muons/muons_evt_list.csv
    muons total events: 68536
    preparing Neutrinos
    401it [01:01,  6.55it/s]
    /eos/user/z/zhibin/sndData/converted//Neutrinos/Neutrinos_evt_list.csv
    Neutrinos total events: 317831
    preparing kaons
    10223it [18:35,  9.16it/s]
    /eos/user/z/zhibin/sndData/converted//kaons/kaons_evt_list.csv
    kaons total events: 22348488
    preparing neutrons
    10510it [21:51,  8.01it/s]
    /eos/user/z/zhibin/sndData/converted//neutrons/neutrons_evt_list.csv
    neutrons total events: 24315013'''


def gen_split_list(root_path):
    particles_split = {
        "Neutrinos": (100000, 150000),
        "kaons": (10000, 15000),  #will be 10 times larger since this is the split for 10GeV interval
        "neutrons": (10000, 15000) #will be 10 times larger since this is the split for 10GeV interval
    }

    combined_train_set = pd.DataFrame()
    combined_val_set = pd.DataFrame()
    combined_test_set = pd.DataFrame()

    for particle, (train_num, val_num) in particles_split.items():

        csv_path = f"{root_path}/{particle}/{particle}_evt_list.csv"
        event_list = pd.read_csv(csv_path)

        if ((particle == 'kaons') | (particle =='neutrons')):
            event_list['cumulative_event'] = event_list.groupby('partition')['event'].cumsum()
        else:
            event_list['cumulative_event'] = event_list['event'].cumsum()
        train_set = event_list[event_list['cumulative_event'] <= train_num].drop(columns=['cumulative_event'])
        val_set = event_list[(event_list['cumulative_event'] > train_num) & (event_list['cumulative_event'] <= val_num)].drop(columns=['cumulative_event'])
        test_set = event_list[event_list['cumulative_event'] > val_num].drop(columns=['cumulative_event'])

        # Combine the splits into the combined sets
        combined_train_set = pd.concat([combined_train_set, train_set], ignore_index=True)
        combined_val_set = pd.concat([combined_val_set, val_set], ignore_index=True)
        combined_test_set = pd.concat([combined_test_set, test_set], ignore_index=True)

        print(f"Splits for {particle} added: train ({len(train_set)}), val ({len(val_set)}), test ({len(test_set)})")

    # Save the combined sets
    combined_train_set.to_csv(f"{root_path}/combined_train.csv", index=False)
    combined_val_set.to_csv(f"{root_path}/combined_val.csv", index=False)
    combined_test_set.to_csv(f"{root_path}/combined_test.csv", index=False)

    print(f"Combined splits saved: train ({len(combined_train_set)}), val ({len(combined_val_set)}), test ({len(combined_test_set)})")

    train_counts = combined_train_set['partition'].value_counts()
    val_counts = combined_val_set['partition'].value_counts()
    test_counts = combined_test_set['partition'].value_counts()
    
    train_sum = combined_train_set.groupby('partition')['event'].sum()
    val_sum = combined_val_set.groupby('partition')['event'].sum()
    test_sum = combined_test_set.groupby('partition')['event'].sum()

    print("\nTrain set:")
    for particle, count in train_counts.items():
        print(f"{particle}: count = {count}, sum = {train_sum[particle]}")
    
    print("\nValidation set:")
    for particle, count in val_counts.items():
        print(f"{particle}: count = {count}, sum = {val_sum[particle]}")
    
    print("\nTest set:")
    for particle, count in test_counts.items():
        print(f"{particle}: count = {count}, sum = {test_sum[particle]}")

'''
    train set:
    Neutrinos: count = 125, sum = 99541
    Filterv4_kaons_80_90_tgtarea: count = 10, sum = 10000
    Filterv4_neutrons_80_90_tgtarea: count = 10, sum = 10000
    Filterv4_neutrons_70_80_tgtarea: count = 10, sum = 10000
    Filterv4_neutrons_60_70_tgtarea: count = 10, sum = 10000
    Filterv4_neutrons_50_60_tgtarea: count = 10, sum = 10000
    Filterv4_neutrons_40_50_tgtarea: count = 10, sum = 10000
    Filterv4_neutrons_30_40_tgtarea: count = 10, sum = 10000
    Filterv4_neutrons_20_30_tgtarea: count = 10, sum = 10000
    Filterv4_kaons_90_100_tgtarea: count = 10, sum = 9700
    Filterv4_kaons_70_80_tgtarea: count = 10, sum = 10000
    Filterv4_kaons_60_70_tgtarea: count = 10, sum = 10000
    Filterv4_kaons_50_60_tgtarea: count = 10, sum = 10000
    Filterv4_kaons_40_50_tgtarea: count = 10, sum = 10000
    Filterv4_kaons_30_40_tgtarea: count = 10, sum = 10000
    Filterv4_kaons_20_30_tgtarea: count = 10, sum = 10000
    Filterv4_neutrons_90_100_tgtarea: count = 10, sum = 9986
    Filterv4_kaons_10_20_tgtarea: count = 5, sum = 10000
    Filterv4_neutrons_10_20_tgtarea: count = 5, sum = 10000
    Filterv4_kaons_5_10_tgtarea: count = 5, sum = 10000
    Filterv4_neutrons_5_10_tgtarea: count = 5, sum = 10000
    Filterv4_kaons_5_10_tgtarea_highstat: count = 2, sum = 10000
    Filterv4_neutrons_5_10_tgtarea_highstat: count = 2, sum = 10000

    Validation set:
    Neutrinos: count = 63, sum = 50099
    Filterv4_kaons_80_90_tgtarea: count = 5, sum = 5000
    Filterv4_neutrons_80_90_tgtarea: count = 5, sum = 5000
    Filterv4_neutrons_70_80_tgtarea: count = 5, sum = 5000
    Filterv4_neutrons_60_70_tgtarea: count = 5, sum = 5000
    Filterv4_neutrons_50_60_tgtarea: count = 5, sum = 5000
    Filterv4_neutrons_40_50_tgtarea: count = 5, sum = 5000
    Filterv4_neutrons_30_40_tgtarea: count = 5, sum = 5000
    Filterv4_neutrons_20_30_tgtarea: count = 5, sum = 5000
    Filterv4_kaons_90_100_tgtarea: count = 5, sum = 4922
    Filterv4_kaons_70_80_tgtarea: count = 5, sum = 5000
    Filterv4_kaons_60_70_tgtarea: count = 5, sum = 5000
    Filterv4_kaons_50_60_tgtarea: count = 5, sum = 5000
    Filterv4_kaons_40_50_tgtarea: count = 5, sum = 5000
    Filterv4_kaons_30_40_tgtarea: count = 5, sum = 5000
    Filterv4_kaons_20_30_tgtarea: count = 5, sum = 5000
    Filterv4_neutrons_90_100_tgtarea: count = 5, sum = 4974
    Filterv4_kaons_10_20_tgtarea: count = 2, sum = 4000
    Filterv4_neutrons_10_20_tgtarea: count = 2, sum = 4000
    Filterv4_kaons_5_10_tgtarea: count = 2, sum = 4000
    Filterv4_neutrons_5_10_tgtarea: count = 2, sum = 4000
    Filterv4_kaons_5_10_tgtarea_highstat: count = 1, sum = 5000
    Filterv4_neutrons_5_10_tgtarea_highstat: count = 1, sum = 5000

    Test set:
    Filterv4_neutrons_5_10_tgtarea_highstat: count = 2979, sum = 14,895,000
    Filterv4_kaons_5_10_tgtarea_highstat: count = 2885, sum = 14425000
    Filterv4_neutrons_5_10_tgtarea: count = 989, sum = 1978000
    Filterv4_kaons_5_10_tgtarea: count = 988, sum = 1976000
    Filterv4_neutrons_10_20_tgtarea: count = 984, sum = 1968000
    Filterv4_neutrons_20_30_tgtarea: count = 981, sum = 981000
    Filterv4_neutrons_40_50_tgtarea: count = 976, sum = 976000
    Filterv4_kaons_40_50_tgtarea: count = 972, sum = 972000
    Filterv4_neutrons_30_40_tgtarea: count = 964, sum = 964000
    Filterv4_kaons_20_30_tgtarea: count = 783, sum = 783000
    Filterv4_kaons_30_40_tgtarea: count = 782, sum = 782000
    Filterv4_kaons_10_20_tgtarea: count = 590, sum = 1180000
    Filterv4_kaons_90_100_tgtarea: count = 481, sum = 476551
    Filterv4_neutrons_90_100_tgtarea: count = 480, sum = 478914
    Filterv4_neutrons_60_70_tgtarea: count = 479, sum = 478506
    Filterv4_neutrons_80_90_tgtarea: count = 479, sum = 479000
    Filterv4_neutrons_50_60_tgtarea: count = 477, sum = 477000
    Filterv4_neutrons_70_80_tgtarea: count = 477, sum = 476633
    Filterv4_kaons_60_70_tgtarea: count = 464, sum = 461369
    Filterv4_kaons_50_60_tgtarea: count = 384, sum = 384000
    Filterv4_kaons_80_90_tgtarea: count = 383, sum = 382946
    Filterv4_kaons_70_80_tgtarea: count = 363, sum = 363000
    Neutrinos: count = 211, sum = 168191'''


def cal_weight(event_list):
    data_neutron = {
    "partition": ["neutrons_5_10", "neutrons_10_20", "neutrons_20_30", "neutrons_30_40", "neutrons_40_50", "neutrons_50_60", "neutrons_60_70", "neutrons_70_80", "neutrons_80_90",'neutrons_90_100'],
    "intRate": [4.62e4, 7.59e3, 1.18e3, 5.30e2, 4.66e2, 2.60e1, 1.80e1, 8.48, 8.48, 1]
    }   
    data_kaons = {
        "partition": ["kaons_5_10", "kaons_10_20", "kaons_20_30", "kaons_30_40", "kaons_40_50", "kaons_50_60", "kaons_60_70", "kaons_70_80", "kaons_80_90", "kaons_90_100"],
        "intRate": [2.51e4, 5.72e3, 8.53e2, 1.10e2, 9.38e1, 6.48e1, 9.90, 2.32e1, 1.15e1,1]
    }
    data_neutrino = {
        "partition": ["neutrino"],
        "intRate": [157]
    }

    # Create dataframes
    df_neutron = pd.DataFrame(data_neutron)
    df_kaons = pd.DataFrame(data_kaons)
    df_neutrino = pd.DataFrame(data_neutrino)

    # Combine the dataframes into one
    intRate_df = pd.concat([df_neutron, df_kaons, df_neutrino], ignore_index=True)

    for idx, row in intRate_df.iterrows():
        partition = row["partition"]
        train_events = event_list[event_list["partition"].str.contains(partition, case=False, na=False)]["event"].sum()
        if train_events > 0:
            intRate_df.at[idx, "train_events"] = train_events
    

    intRate_df['weight'] = intRate_df['intRate'] / intRate_df['train_events'] * 100

    intRate_df['intRate_weight'] = intRate_df['intRate'] / intRate_df['intRate'].sum() *10

    # Extracting neutron and kaon data separately for normalizing
    neutron_data = intRate_df[intRate_df['partition'].str.contains("neutrons")]
    kaon_data = intRate_df[intRate_df['partition'].str.contains("kaons")]
    neutrino_data = intRate_df[intRate_df['partition'] == 'neutrino']



    # Normalizing weights
    normalized_neutron_weights = neutron_data['weight'] / neutron_data['weight'].sum()
    normalized_kaon_weights = kaon_data['weight'] / kaon_data['weight'].sum() 
    normalized_neutrino_weights = neutrino_data['weight'] / neutrino_data['weight'].sum() 

    # Assigning the normalized weights back to the DataFrame
    intRate_df.loc[intRate_df['partition'].str.contains("neutrons"), 'normalized_weight'] = normalized_neutron_weights.values
    intRate_df.loc[intRate_df['partition'].str.contains("kaons"), 'normalized_weight'] = normalized_kaon_weights.values
    intRate_df.loc[intRate_df['partition'] == 'neutrino', 'normalized_weight'] = normalized_neutrino_weights.values


    print(intRate_df)

    
    event_list["normalized_weight"] = None
    event_list["weight"] = None
    event_list["intRate_weight"] = None
    for idx, row in intRate_df.iterrows():
        partition = row["partition"]
        weight = row["weight"]
        normalized_weight = row["normalized_weight"]
        intRate_weight = row["intRate_weight"]

        # Updating the weight and normalized_weight in event_list
        mask = event_list["partition"].str.contains(partition, case=False, na=False)
        event_list.loc[mask, "weight"] = weight
        event_list.loc[mask, "normalized_weight"] = normalized_weight
        event_list.loc[mask, "intRate_weight"] = intRate_weight

    print("None value count:", event_list['weight'].isnull().sum())
    return event_list 

def allocate_weight(root_path):
    partitions = {
        'train',
        'val',
        'test'
    }

    for partition in partitions:
        csv_path = f"{root_path}/combined_{partition}.csv"
        event_list = pd.read_csv(csv_path)
        if partition == 'train':
            event_list = cal_weight(event_list)
        else:
            event_list['weight'] = 1
            event_list['normalized_weight'] = 1
        
        print(event_list)

        event_list.to_csv(csv_path)

def allocate_weight_to_particle(root_path):
    partitions = {
        'neutrinos',
        'kaons',
        'neutrons'
    }

    for partition in partitions:
        csv_path = f"{root_path}/{partition}/{partition}_evt_list.csv"
        event_list = pd.read_csv(csv_path)
        event_list = cal_weight(event_list)
        print(event_list)

        event_list[['partition', 'event', 'file', 'normalized_weight', 'weight', 'intRate_weight']].to_csv(csv_path, index=False)


def kaon_neuton_evt_list(root_path):
    # kaon and neutron
    kaon_list = pd.read_csv('/afs/cern.ch/user/z/zhibin/eos/sndData/converted/kaons/kaons_evt_list.csv')
    neutron_list = pd.read_csv('/afs/cern.ch/user/z/zhibin/eos/sndData/converted/neutrons/neutrons_evt_list.csv')

    combine_list = pd.concat([kaon_list,neutron_list])

    mixed_list = combine_list.sample(frac=1, random_state=1).reset_index(drop=True)
    print(mixed_list)

    mixed_list.to_csv('/afs/cern.ch/user/z/zhibin/eos/sndData/converted/mixed_evt_list.csv')


if __name__ == "__main__":
    root_path = '/eos/user/z/zhibin/sndData/converted/'
    prepareEvtList(root_path)
    #gen_split_list(root_path)

    #allocate_weight(root_path)
    #allocate_weight_to_particle(root_path)
    #kaon_neuton_evt_list(root_path)
