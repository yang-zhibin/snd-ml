import csv
import re

def extract_digi_paths(log_file, out_csv):
    digi_paths = []

    # Regex to capture the first input file after "input:"
    input_re = re.compile(r'^\s*input:\s*([^,\s]+)', re.MULTILINE)

    with open(log_file, "r") as f:
        content = f.read()

    for match in input_re.finditer(content):
        digi_paths.append(match.group(1))

    # Write to CSV
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["digi_path"])
        writer.writeheader()
        for p in digi_paths:
            writer.writerow({"digi_path": p})


if __name__ == "__main__":
    LOG_FILE = "/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/.snakemake/log/2026-01-23T194752.926805.snakemake.log"
    LOG_CSV  = "/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/metadata/raw/log_outputs.csv"

    extract_digi_paths(LOG_FILE, LOG_CSV)
