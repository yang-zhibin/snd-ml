
import pandas as pd

# ---- runs you want to keep --------------------------------------------------
RUNS_TO_KEEP = {
    8285, 8315, 8320, 8323, 8638, 8724, 9015, 9094, 9258, 9262, 9288,
    9361, 9411, 9436, 9562, 9569, 9622, 9685, 9715, 9880, 9885, 9913
}
# -----------------------------------------------------------------------------

def main() -> None:
    # 1) read the CSV
    df = pd.read_csv("/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2024_metadata.csv")

    # 2) pull the numeric run id from the “partition” string:  run_00##### ➜ #####
    #       - str.extract gives the first capture group (digits)
    #       - astype(int) turns it into an integer so membership tests are cheap
    df["run_number"] = (
        df["partition"]
        .str.extract(r"run_00(\d+)", expand=False)
    )

    # drop rows where regex didn’t match (i.e. run_number is NaN)
    df = df[df["run_number"].notna()]
    df["run_number"] = df["run_number"].astype(int)

    # 3) filter to just the runs you care about
    df_filtered = df[df["run_number"].isin(RUNS_TO_KEEP)]

    total_lumi = df_filtered["lumi_per_file"].sum()
    print(f"Total lumi_per_file for selected runs: {total_lumi:.2f}")
    # 4) write result
    df_filtered.to_csv("./updated/real_data_2024_skim_runs.csv", index=False)

if __name__ == "__main__":
    main()