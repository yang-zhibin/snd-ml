import pandas as pd
import glob
import ROOT
from tqdm import tqdm
import argparse
def read_and_sum_stage2(file_path):
    """Function to read a file and sum the 'stage2' variable using ROOT."""
    print('reading', file_path)
    df = ROOT.RDataFrame("cbmsim", file_path)
    stage2_count = df.Sum("stage2").GetValue()
    return stage2_count
def read_stage2(df, job_number):
    output_file = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/stage2/stage2count_{job_number}.csv'


    with open(output_file, 'w') as f:
        # Write the header of the output CSV file
        f.write('path,n_event,stage2count\n')
        # Process each row in DataFrame
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing files"):
            stage2_count = read_and_sum_stage2(row['path'])
            # Write the result immediately to the output CSV file
            f.write(f"{row['path']},{row['n_event']},{stage2_count}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-n", "--num", dest="num", type=int, required=True)
    args = parser.parse_args()
    job_number = args.num

    file_list_dir = '/eos/user/z/zhibin/sndData/converted/'
    dataset_name = 'prediction'
    file = f'{file_list_dir}/{dataset_name}_files.csv'
    file_list_df = pd.read_csv(file, header=None, names=['path', 'n_event'])

    df = file_list_df[job_number*1000:job_number*1000+1000]
    read_stage2(df,job_number)