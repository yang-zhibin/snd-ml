import pandas as pd


def set_3d_split():
    mc_neutrino_metadata_path ='./updated/MC_neutrino_volTarget_100fb-1_metadata.csv'
    mc_neutrino_metadata = pd.read_csv(mc_neutrino_metadata_path)

    # set 40% as train, 20% as validation, 40% test, add a column "split_3d_1" store train/valid/test
    total = len(mc_neutrino_metadata)
    train_end = int(0.4 * total)
    valid_end = train_end + int(0.2 * total)

    # Assign splits
    mc_neutrino_metadata.loc[:train_end - 1, 'split_3d_1'] = 'train'
    mc_neutrino_metadata.loc[train_end:valid_end - 1, 'split_3d_1'] = 'valid'
    mc_neutrino_metadata.loc[valid_end:, 'split_3d_1'] = 'test'

    # Print the result
    print(mc_neutrino_metadata)
    mc_neutrino_metadata.to_csv(mc_neutrino_metadata_path, index=False)

def main():
    set_3d_split()

if __name__ == "__main__":
    main()

