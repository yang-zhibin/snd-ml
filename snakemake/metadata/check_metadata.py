import pandas as pd
import os
import glob

def main():
    metadata_rootpath = './updated'
    metadata_files = glob.glob(os.path.join(metadata_rootpath, "*_metadata.csv"))
    target_paths = []
    for file in metadata_files:
        metadata = pd.read_csv(file)
        target_paths.extend(metadata[column_name].tolist())
    return target_paths

    # 

if __name__ == "__main__":
    main()