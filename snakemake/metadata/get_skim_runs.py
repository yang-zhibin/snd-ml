
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple




def process_real_data() -> None:
    # ---- runs you want to keep --------------------------------------------------
    RUNS_TO_KEEP = {
        8285, 8315, 8320, 8323, 8638, 8724, 9015, 9094, 9258, 9262, 9288,
        9361, 9411, 9436, 9562, 9569, 9622, 9685, 9715, 9880, 9885,9913
    }

    # -----------------------------------------------------------------------------

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
    df_filtered.to_csv("./updated/real_data_2024_skim_runs_metadata.csv", index=False)

def process_neutral_bkg(
    csv_path: str,
    targets: Optional[Dict[str, int]] = None,
    default_target: int = 300_000,
    group_col: str = "energy_range",
    event_col: str = "n_event",
    shuffle: bool = False,
    random_state: int = 42,
    save_subset: bool = True,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """
    Reads `csv_path`, samples rows per bin up to targets, and (optionally) saves to
    {original_path}/{original_name}_subset.csv
    """
    if targets is None:
        targets = {"(5-10)": 600_000, "(10-20)": 400_000}

    df = pd.read_csv(csv_path)
    sampled_groups = []

    for energy_range, group in df.groupby(group_col, sort=False):
        target_events = targets.get(energy_range, default_target)
        if shuffle:
            group = group.sample(frac=1, random_state=random_state)

        # Include a row if the *previous* cumulative sum < target (matches your loop)
        prev_cum = group[event_col].cumsum().shift(fill_value=0)
        selected = group.loc[prev_cum < target_events]
        sampled_groups.append(selected)

    sampled_df = pd.concat(sampled_groups, ignore_index=True)

    out_path = None
    if save_subset:
        p = Path(csv_path)
        out_path = p.parent / f"{p.stem}_subset.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(out_path, index=False)
        print(f"Saved subset to: {out_path}")

    print(sampled_df)
    return sampled_df, out_path
    

if __name__ == "__main__":
    # 
    #process_real_data()
    #process_kaon()
    kaon_df,kaon_path = process_neutral_bkg("./updated/MC_kaon_FTFP_BERT_metadata.csv")
    neutron_df,neutron_path = process_neutral_bkg("./updated/MC_neutron_FTFP_BERT_metadata.csv")
    #process_neutron()