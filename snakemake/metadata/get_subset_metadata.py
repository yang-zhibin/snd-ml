import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


RUNS_TO_KEEP = {
    8285, 8315, 8320, 8323, 8638, 8724, 9015, 9094, 9258, 9262, 9288,
    9361, 9411, 9436, 9562, 9569, 9622, 9685, 9715, 9880, 9885, 9913
}


def process_real_data(
    input_csv: str,
    output_full_csv: str,
    output_subset_csv: str,
    runs_to_keep: Optional[set] = None,
    subset_nrows: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if runs_to_keep is None:
        runs_to_keep = RUNS_TO_KEEP

    input_path = Path(input_csv)
    output_full_path = Path(output_full_csv)
    output_subset_path = Path(output_subset_csv)

    output_full_path.parent.mkdir(parents=True, exist_ok=True)
    output_subset_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    df["run_number"] = df["partition"].str.extract(r"run_00(\d+)", expand=False)

    df = df[df["run_number"].notna()].copy()
    df["run_number"] = df["run_number"].astype(int)

    df_filtered = df[df["run_number"].isin(runs_to_keep)].copy()
    df_subset = df_filtered.head(subset_nrows).copy()

    total_lumi = df_filtered["lumi_per_file"].sum()
    print(f"Selected rows: {len(df_filtered)}")
    print(f"Total lumi_per_file for selected runs: {total_lumi:.2f}")

    df_filtered.to_csv(output_full_path, index=False)
    df_subset.to_csv(output_subset_path, index=False)

    print(f"Saved skimmed metadata to: {output_full_path}")
    print(f"Saved first {len(df_subset)} rows to: {output_subset_path}")

    return df_filtered, df_subset


def process_neutral_bkg(
    csv_path: str,
    output_csv: Optional[str] = None,
    targets: Optional[Dict[str, int]] = None,
    default_target: int = 300_000,
    group_col: str = "energy_range",
    event_col: str = "n_event",
    shuffle: bool = False,
    random_state: int = 42,
    save_subset: bool = True,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """
    Reads `csv_path`, samples rows per bin up to targets, and optionally saves to output_csv.
    The selection rule keeps a row if the previous cumulative sum is < target.
    """
    if targets is None:
        targets = {"(5-10)": 600_000, "(10-20)": 400_000}

    input_path = Path(csv_path)
    df = pd.read_csv(input_path)

    sampled_groups = []

    for energy_range, group in df.groupby(group_col, sort=False):
        target_events = targets.get(energy_range, default_target)

        if shuffle:
            group = group.sample(frac=1, random_state=random_state).reset_index(drop=True)

        prev_cum = group[event_col].cumsum().shift(fill_value=0)
        selected = group.loc[prev_cum < target_events].copy()
        sampled_groups.append(selected)

        print(
            f"{energy_range}: selected {len(selected)} rows, "
            f"total {selected[event_col].sum()} {event_col} "
            f"(target {target_events})"
        )

    sampled_df = pd.concat(sampled_groups, ignore_index=True)

    out_path = None
    if save_subset:
        if output_csv is None:
            out_path = input_path.parent / f"{input_path.stem}_subset.csv"
        else:
            out_path = Path(output_csv)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(out_path, index=False)
        print(f"Saved subset to: {out_path}")

    return sampled_df, out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate skim/subset metadata files.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_real = subparsers.add_parser("real-data", help="Create skimmed real-data metadata.")
    parser_real.add_argument("--input", required=True, help="Input real-data metadata CSV")
    parser_real.add_argument("--output-full", required=True, help="Output skimmed metadata CSV")
    parser_real.add_argument("--output-subset", required=True, help="Output skimmed subset CSV")
    parser_real.add_argument("--subset-nrows", type=int, default=500, help="Number of rows for subset output")

    parser_bkg = subparsers.add_parser("neutral-bkg", help="Create neutral background subset metadata.")
    parser_bkg.add_argument("--input", required=True, help="Input neutral background metadata CSV")
    parser_bkg.add_argument("--output", required=True, help="Output subset CSV")
    parser_bkg.add_argument("--default-target", type=int, default=300_000)
    parser_bkg.add_argument("--shuffle", action="store_true")
    parser_bkg.add_argument("--random-state", type=int, default=42)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "real-data":
        process_real_data(
            input_csv=args.input,
            output_full_csv=args.output_full,
            output_subset_csv=args.output_subset,
            subset_nrows=args.subset_nrows,
        )
    elif args.mode == "neutral-bkg":
        process_neutral_bkg(
            csv_path=args.input,
            output_csv=args.output,
            default_target=args.default_target,
            shuffle=args.shuffle,
            random_state=args.random_state,
            save_subset=True,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()