import pandas as pd
import os

def check_subfolder(csv_file):
    # read csv 
    df = pd.read_csv(csv_file)

    # Check for duplicates based on 'subfolder' and 'partition'
    duplicates = df[df.duplicated(subset=['subfolder', 'partition'], keep=False)]
    
    if not duplicates.empty:
        print(f"Found duplicates in {csv_file}.")
        print(duplicates)
    else:
        print(f"No duplicates found in {csv_file}.")

    # generate hits file

def add_hit_path(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Create the `hit_path` column
    df['hit_path'] = df.apply(
        lambda row: f"hit_{row['data_type']}_{row['subfolder'].replace('/', '_')}_{row['partition']}.root", axis=1
    )

    # Save the updated DataFrame back to the original file
    df.to_csv(csv_file, index=False)

def main():
    folder_path = './'
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

    for csv_file in csv_files:
        check_subfolder(csv_file)
        add_hit_path(csv_file)


if __name__ == "__main__":
    main()

