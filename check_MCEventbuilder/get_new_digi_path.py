import pandas as pd

# Path to the input metadata CSV
input_csv = "/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv"

# Columns to keep
columns_to_keep = [
    "data_type",
    "digi_path",
    "geo_path",
    "raw_path",
    "newRaw_path",
    "newDigi_path"
]

# Read the CSV and keep only specified columns
df = pd.read_csv(input_csv, usecols=columns_to_keep)

# Save the filtered DataFrame to a new CSV
output_csv = "./newDigi_withMCEventBuilder.csv"
df.to_csv(output_csv, index=False)

print(f"Filtered metadata saved to: {output_csv}")