# Machine Learning for SND


## How to run snakemake
### Local
- `snakemake -j 1` (means 1 core will be used)
### On HTCondor
- setup snakemake for htcondor (https://github.com/Snakemake-Profiles/htcondor)
- `snakemake --profile .config/snakemake/htcondor`
## Generate Metadata
- generate raw Metadata
get digitized file and geo file, check if they are corrupted. (It takes a while if there are so many files since it will need to open it)
    - in snakemake file
        1. uncomment the input in `rule finish`:
            `expand("/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/raw/{metadata_csv}", metadata_csv=metadata_csv_list)`
        1. uncomment tthe metadata csv you want in `metadata_csv_list` 
    - run snakemake 
- update metadata
update the input/output file path name for this workflow (e.g. GNN input/output)
    - in snakemake file
        1. uncomment the input in `rule finish`:
            `expand("/afs/cern.ch/work/z/zhibin/snd-ml/snakemake/metadata/updated/{metadata_csv}", metadata_csv=metadata_csv_list)`
        1. uncomment the metadata csv you want in `metadata_csv_list` 
    - run snakemake 

There reason that it seperate into two processes is one can easily add new steps input/output into the metadata files without checking the raw files

## Training


## Evaluation