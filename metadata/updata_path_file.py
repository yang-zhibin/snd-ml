import pandas as pd
import os

def check_subfolder(csv_file):
    # read csv 
    df = pd.read_csv(csv_file)

    # Check for duplicates based on 'subfolder' and 'partition'
    duplicate_mask = df.duplicated(subset=['subfolder', 'partition'], keep=False)
    duplicates = df[duplicate_mask]
    
    if duplicate_mask.any():
        # Update 'subfolder' with unique IDs for duplicates
        df['partition'] = df.apply(
            lambda row: f"{row['partition']}/{row.name}" if duplicate_mask[row.name] else row['partition'],
            axis=1
        )
        
        # Save the updated DataFrame back to the original file
        df.to_csv(csv_file, index=False)
        print(f"Duplicates found and partition updated. Changes saved to {csv_file}.")
    else:
        print(f"No duplicates found in {csv_file}.")


def add_hit_path(csv_file, eos_root_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Create the `hit_path` column
    df['subfolder'] = df['subfolder'].astype(str)
    if 'muon' in csv_file:
        df['hit_path'] = df.apply(
            lambda row: f"hit_{row['data_type']}_{row['subfolder'].replace('/', '_')}_{row['partition']}_{row['n_event']}.root", axis=1
        )
    else:
        df['hit_path'] = df.apply(
            lambda row: f"hit_{row['data_type']}_{row['subfolder'].replace('/', '_')}_{row['partition']}.root", axis=1
        )

    # Add directory path to hit_path
    df['hit_path'] = df.apply(
        lambda row: f"{eos_root_path}/{row['data_type']}/{row['subfolder']}/{row['partition']}/{row['hit_path']}", axis=1
    )


    # Save the updated DataFrame back to the original file
    df.to_csv(csv_file, index=False)

def add_job_number(csv_file, eos_root_path):
    pass
    #add job number, process 1e6 events per job

def add_feature_path(csv_file, eos_root_path):
    pass


def main():
    folder_path = './'
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

    eos_root_path = '/eos/user/z/zhibin/sndData'
    for csv_file in csv_files:
        #check_subfolder(csv_file)
        add_hit_path(csv_file, eos_root_path)


if __name__ == "__main__":
    main()

