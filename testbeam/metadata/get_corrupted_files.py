import re
import csv
import pandas as pd
from pathlib import Path


def extract_log_outputs(log_file: str, out_csv: str = "log_outputs.csv"):
    """
    Parse a Snakemake log and collect all unique output paths.
    
    Parameters
    ----------
    log_file : str
        Path to Snakemake log file.
    out_csv : str, optional
        Path for output CSV file.
    
    Returns
    -------
    list[str]
        Unique list of output paths found in the log.
    """
    outputs = []
    re_output = re.compile(r"^\s*output:\s*(.+)")

    with open(log_file) as f:
        for line in f:
            m = re_output.match(line)
            if m:
                parts = [p.strip() for p in m.group(1).split(",")]
                outputs.extend(parts)

    # unique while preserving order
    seen = set()
    unique = []
    for p in outputs:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    # write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_path"])
        for path in unique:
            writer.writerow([path])

    print(f"Extracted {len(unique)} output paths → {out_csv}")
    return unique


def clean_metadata_folder(
    metadata_folder: str,
    log_csv: str,
    key: str = "feature_path",
    backup_suffix: str | None = ".bak"
):
    """
    Cross-check metadata CSV files with log outputs and delete matching rows.

    Parameters
    ----------
    metadata_folder : str
        Directory containing metadata CSVs.
    log_csv : str
        CSV containing a column of feature paths to remove.
    key : str, optional
        Name of the column used to match (default: 'feature_path').
    backup_suffix : str | None, optional
        If not None, original CSVs are kept with this suffix.
        If None, metadata files are overwritten directly.

    Returns
    -------
    dict
        Summary: {filename: number_of_rows_removed}
    """
    meta_dir = Path(metadata_folder)
    paths_to_remove = set(pd.read_csv(log_csv)[key].dropna().astype(str))
    
    summary = {}
    for csv_file in meta_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        if key not in df.columns:
            print(f"[skip] {csv_file.name}: no column '{key}'")
            continue
        
        before = len(df)
        df = df[~df[key].astype(str).isin(paths_to_remove)]
        removed = before - len(df)

        # backup original if requested
        if backup_suffix:
            csv_file.rename(csv_file.with_suffix(csv_file.suffix + backup_suffix))

        # save cleaned metadata
        df.to_csv(csv_file, index=False)
        summary[csv_file.name] = removed
        print(f"[ok] {csv_file.name}: removed {removed} rows")

    return summary


if __name__ == "__main__":
    LOG_FILE = '/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/.snakemake/log/2025-12-18T110345.765005.snakemake.log'
    LOG_CSV = "/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/metadata/raw/log_outputs.csv"
    # paths = extract_log_outputs(LOG_FILE, LOG_CSV)
    
    metadata_folder = "/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/metadata/updated"
    summary = clean_metadata_folder(
        metadata_folder=metadata_folder,
        log_csv=LOG_CSV
    )
        
    print(summary)
    