import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT
from tqdm import tqdm

import io
import contextlib


ROOT.EnableImplicitMT()

PRESELECT_CUT = """
(
    AvgSFChan == 1
    && NoVetoHits == 0
    && At_least_two_consecutive_SciFi_planes == 1
    && SciFiContinuity == 1
)
""".strip()


def extract_sample_name(file_name: str) -> str:
    stem = Path(file_name).stem

    if not stem.startswith("MC_") or not stem.endswith("_metadata"):
        raise ValueError(f"Unexpected filename format: {file_name}")

    return stem[len("MC_"):-len("_metadata")]


def extract_particle_name(sample_name: str) -> str:
    return sample_name.split("_")[0]


def build_split_group(df: pd.DataFrame, sample_name: str) -> pd.Series:
    particle_name = extract_particle_name(sample_name)

    if particle_name in ["kaon", "neutron"]:
        if "energy_range" not in df.columns:
            raise ValueError(f"'energy_range' column not found for {sample_name}")
        return sample_name + "__" + df["energy_range"].astype(str)

    return pd.Series([sample_name] * len(df), index=df.index)


def build_train_cap_group(df: pd.DataFrame) -> pd.Series:
    special_mask = df["particle_name"].isin(["kaon", "neutron"])

    result = df["sample_name"].copy()
    result.loc[special_mask] = (
        df.loc[special_mask, "sample_name"]
        + "__"
        + df.loc[special_mask, "energy_range"].astype(str)
    )
    return result


def count_preselected_events(root_file_path: Path) -> int:
    root_file_path = str(Path(root_file_path))

    if not Path(root_file_path).exists():
        raise FileNotFoundError(f"ROOT file not found: {root_file_path}")

    rdf = ROOT.RDataFrame("cutFlowSummary", root_file_path)
    count = rdf.Filter(PRESELECT_CUT).Count()

    return int(count.GetValue())


def add_preselect_count_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    preselect_counts = []

    print("\nComputing preSelect_count from ROOT files...")
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Processing ROOT files",
        unit="file",
    ):
        root_path = row.nueAnalysisFilter_path
        base_path = row.output_base_path
        full_root_path = f"{base_path}/{root_path}"
        count = count_preselected_events(full_root_path)
        preselect_counts.append(count)

    df["preSelect_count"] = preselect_counts
    return df


def load_and_merge_metadata(metadata_dir, metadata_file_list):
    metadata_dir = Path(metadata_dir)
    df_list = []

    for file_name in metadata_file_list:
        file_path = metadata_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        sample_name = extract_sample_name(file_name)
        particle_name = extract_particle_name(sample_name)

        df = pd.read_csv(file_path)

        required_columns = ["n_event", "lumi_per_file", "nueAnalysis_path"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"'{col}' column not found in {file_name}")

        if particle_name in ["kaon", "neutron"] and "energy_range" not in df.columns:
            raise ValueError(f"'energy_range' column not found in {file_name}")

        df = df.copy()
        df["source_metadata_file"] = file_name
        df["sample_name"] = sample_name
        df["particle_name"] = particle_name
        df["split_group"] = build_split_group(df, sample_name)

        df_list.append(df)

        print(f"Loaded {file_name:<40} -> {sample_name}")

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df["train_cap_group"] = build_train_cap_group(merged_df)

    return merged_df


def split_one_group(
    df_group: pd.DataFrame,
    train_frac: float = 0.4,
    test_frac: float = 0.1,
    val_frac: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    if not np.isclose(train_frac + test_frac + val_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.")

    df_group = df_group.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df_group)

    if n == 1:
        split_labels = ["train"]
    elif n == 2:
        split_labels = ["train", "val"]
    else:
        n_train = int(np.floor(n * train_frac))
        n_test = int(np.floor(n * test_frac))
        n_val = n - n_train - n_test

        if n_train == 0:
            n_train = 1
            if n_val > 1:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

        split_labels = ["train"] * n_train + ["test"] * n_test + ["val"] * n_val

        if len(split_labels) != n:
            raise RuntimeError(
                f"Split label length mismatch: len(split_labels)={len(split_labels)}, n={n}"
            )

    df_group = df_group.copy()
    df_group["split"] = split_labels
    return df_group


def stratified_split(
    merged_df: pd.DataFrame,
    train_frac: float = 0.4,
    test_frac: float = 0.1,
    val_frac: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    split_df_list = []

    for split_group, df_group in merged_df.groupby("split_group", sort=True):
        group_seed = seed + (abs(hash(str(split_group))) % 100000)
        split_df = split_one_group(
            df_group=df_group,
            train_frac=train_frac,
            test_frac=test_frac,
            val_frac=val_frac,
            seed=group_seed,
        )
        split_df_list.append(split_df)

    return pd.concat(split_df_list, ignore_index=True)


def select_rows_by_event_budget(
    df_group: pd.DataFrame,
    target_events: float,
    event_column: str,
    seed: int = 42,
) -> pd.Index:
    if len(df_group) == 0 or target_events <= 0:
        return pd.Index([])

    shuffled = df_group.sample(frac=1, random_state=seed).copy()
    shuffled["cum_event_budget"] = shuffled[event_column].cumsum()

    keep_mask = shuffled["cum_event_budget"] <= target_events

    if keep_mask.any():
        selected = shuffled.loc[keep_mask]
        selected_sum = selected[event_column].sum()

        if len(selected) < len(shuffled):
            next_row = shuffled.iloc[[len(selected)]]
            next_sum = selected_sum + next_row[event_column].iloc[0]

            if abs(next_sum - target_events) < abs(selected_sum - target_events):
                selected = pd.concat([selected, next_row], axis=0)
    else:
        selected = shuffled.iloc[[0]]

    return selected.index


def apply_event_count_based_train_cap(
    df: pd.DataFrame,
    event_column: str = "preSelect_count",
    background_to_signal_ratio: float = 5.0,
    alpha: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    df = df.copy()
    df["selected_for_use"] = False

    train_mask = df["split"] == "train"
    signal_mask = df["particle_name"] == "neutrino"
    background_mask = df["particle_name"].isin(["muon", "kaon", "neutron"])

    df.loc[train_mask & signal_mask, "selected_for_use"] = True

    signal_train_events = df.loc[train_mask & signal_mask, event_column].sum()
    background_train_budget = background_to_signal_ratio * signal_train_events

    print("\nTraining event-count cap")
    print("-" * 120)
    print(f"train signal {event_column:<15}: {int(signal_train_events):,}")
    print(f"background/signal ratio      : {background_to_signal_ratio:.3f}")
    print(f"target background train evts : {int(background_train_budget):,}")
    print(f"allocation alpha             : {alpha:.3f}")
    print("-" * 120)

    train_bkg_df = df.loc[train_mask & background_mask].copy()

    if train_bkg_df.empty:
        print("No background rows found in training split.")
        return df

    available_by_group = (
        train_bkg_df.groupby("train_cap_group", dropna=False)[event_column]
        .sum()
        .sort_index()
    )

    n_groups = len(available_by_group)
    total_available = available_by_group.sum()

    if n_groups == 0 or total_available <= 0:
        print("No available background events for capping.")
        return df

    equal_share = background_train_budget / n_groups

    target_by_group = {}
    for group_name, available in available_by_group.items():
        proportional_share = background_train_budget * (available / total_available)
        target = alpha * equal_share + (1.0 - alpha) * proportional_share
        target = min(target, available)
        target_by_group[group_name] = target

    selected_indices = []

    print(
        f"{'group':<40} {'available':>15} {'target':>15} "
        f"{'selected':>15} {'selected_files':>15}"
    )
    print("-" * 120)

    total_selected_background_events = 0
    total_selected_background_files = 0

    for i, (group_name, target) in enumerate(sorted(target_by_group.items())):
        group_df = train_bkg_df[train_bkg_df["train_cap_group"] == group_name].copy()
        group_seed = seed + i

        selected_idx = select_rows_by_event_budget(
            df_group=group_df,
            target_events=target,
            event_column=event_column,
            seed=group_seed,
        )

        selected_group_df = group_df.loc[selected_idx]
        selected_events = selected_group_df[event_column].sum()
        selected_files = len(selected_group_df)
        available_events = group_df[event_column].sum()

        selected_indices.extend(selected_idx.tolist())

        total_selected_background_events += selected_events
        total_selected_background_files += selected_files

        print(
            f"{str(group_name):<40} "
            f"{int(available_events):>15,} "
            f"{int(target):>15,} "
            f"{int(selected_events):>15,} "
            f"{selected_files:>15}"
        )

    df.loc[selected_indices, "selected_for_use"] = True

    print("-" * 120)
    print(
        f"{'TOTAL_SELECTED_BACKGROUND':<40} "
        f"{int(total_available):>15,} "
        f"{int(sum(target_by_group.values())):>15,} "
        f"{int(total_selected_background_events):>15,} "
        f"{total_selected_background_files:>15}"
    )
    print("-" * 120)

    return df


def apply_split_selection(
    df: pd.DataFrame,
    split_name: str,
    use_frac: float,
    equal_size: bool,
    event_column: str = "n_event",
    seed: int = 42,
) -> pd.DataFrame:
    if not 0.0 <= use_frac <= 1.0:
        raise ValueError(f"{split_name} use fraction must be between 0 and 1.")

    df = df.copy()

    split_mask = df["split"] == split_name
    split_df = df[split_mask].copy()

    df.loc[split_mask, "selected_for_use"] = False

    if split_df.empty:
        print(f"No {split_name} rows found.")
        return df

    available_by_group = (
        split_df.groupby("train_cap_group", dropna=False)[event_column]
        .sum()
        .sort_index()
    )

    if available_by_group.empty:
        print(f"No groups found for {split_name}.")
        return df

    equal_target = available_by_group.min()

    print(f"\nSelecting {split_name} split")
    print("-" * 120)
    print(f"use fraction : {use_frac:.3f}")
    print(f"equal size   : {equal_size}")
    if equal_size:
        print(f"equal target : {int(equal_target):,}")
    print("-" * 120)

    selected_indices = []

    print(
        f"{'group':<40} {'available':>15} {'base_target':>15} "
        f"{'target':>15} {'selected':>15} {'selected_files':>15}"
    )
    print("-" * 120)

    for i, (group_name, available_events) in enumerate(available_by_group.items()):
        group_df = split_df[split_df["train_cap_group"] == group_name].copy()

        if equal_size:
            base_target = equal_target
        else:
            base_target = available_events

        target_events = base_target * use_frac

        selected_idx = select_rows_by_event_budget(
            df_group=group_df,
            target_events=target_events,
            event_column=event_column,
            seed=seed + i,
        )

        selected_group_df = group_df.loc[selected_idx]
        selected_events = selected_group_df[event_column].sum()
        selected_files = len(selected_group_df)

        selected_indices.extend(selected_idx.tolist())

        print(
            f"{str(group_name):<40} "
            f"{int(available_events):>15,} "
            f"{int(base_target):>15,} "
            f"{int(target_events):>15,} "
            f"{int(selected_events):>15,} "
            f"{selected_files:>15}"
        )

    df.loc[selected_indices, "selected_for_use"] = True
    return df


def filter_by_split_dataset(df: pd.DataFrame, split_dataset: str) -> pd.DataFrame:
    if split_dataset == "neutral_hadron":
        return df[df["particle_name"].isin(["kaon", "neutron"])].copy()

    if split_dataset == "neutrino":
        mask = (
            df["sample_name"].str.contains("ve", case=False, regex=False)
            | df["sample_name"].str.contains("vm", case=False, regex=False)
        )
        return df[mask].copy()

    if split_dataset == "all":
        return df.copy()

    raise ValueError(f"Unknown split_dataset: {split_dataset}")


def format_int(value):
    return f"{int(value):,}"


def format_float(value):
    return f"{float(value):,.6f}"


def print_summary_header(event_column: str):
    print(
        f"{'class':<40} "
        f"{event_column:>18} "
        f"{'total_event':>18} "
        f"{'fraction':>12} "
        f"{'lumi_per_file_sum':>20}"
    )


def print_summary_row(name, event_sum, lumi_sum, total_events, indent=0):
    prefix = " " * indent
    width = max(8, 40 - indent)

    frac = 100.0 * event_sum / total_events if total_events else 0.0

    print(
        f"{prefix}{name:<{width}} "
        f"{format_int(event_sum):>18} "
        f"{format_int(total_events):>18} "
        f"{frac:>11.2f}% "
        f"{format_float(lumi_sum):>20}"
    )


def print_event_summary(
    df: pd.DataFrame,
    title: str,
    event_column: str,
    reference_df: pd.DataFrame,
):
    print(f"\n{title}")
    print_summary_header(event_column)
    print("-" * 120)

    particle_total_dict = {}

    for sample_name, sample_df in df.groupby("sample_name", sort=True):
        particle_name = sample_df["particle_name"].iloc[0]

        sample_event_sum = sample_df[event_column].sum()
        sample_lumi_sum = sample_df["lumi_per_file"].sum()

        ref_sample_df = reference_df[reference_df["sample_name"] == sample_name]
        ref_sample_total = ref_sample_df[event_column].sum()

        print_summary_row(
            name=sample_name,
            event_sum=sample_event_sum,
            lumi_sum=sample_lumi_sum,
            total_events=ref_sample_total,
        )

        particle_total_dict.setdefault(
            particle_name,
            {"event": 0, "lumi": 0.0, "ref_event": 0},
        )
        particle_total_dict[particle_name]["event"] += sample_event_sum
        particle_total_dict[particle_name]["lumi"] += sample_lumi_sum
        particle_total_dict[particle_name]["ref_event"] += ref_sample_total

        if particle_name in ["kaon", "neutron"]:
            grouped = (
                sample_df.groupby("energy_range", dropna=False)[
                    [event_column, "lumi_per_file"]
                ]
                .sum()
                .reset_index()
            )

            for _, row in grouped.iterrows():
                energy = row["energy_range"]

                ref_energy_total = ref_sample_df[
                    ref_sample_df["energy_range"] == energy
                ][event_column].sum()

                print_summary_row(
                    name=f"energy={energy}",
                    event_sum=row[event_column],
                    lumi_sum=row["lumi_per_file"],
                    total_events=ref_energy_total,
                    indent=4,
                )

    print("\nEvent summary by particle:")
    print_summary_header(event_column)
    print("-" * 120)

    for particle_name, totals in sorted(particle_total_dict.items()):
        print_summary_row(
            name=particle_name,
            event_sum=totals["event"],
            lumi_sum=totals["lumi"],
            total_events=totals["ref_event"],
        )


def print_selection_overview(
    df: pd.DataFrame,
    event_column: str,
):
    print("\nSelection overview")
    print("-" * 100)
    print(
        f"{'split':<15} "
        f"{'assigned_events':>18} "
        f"{'selected_events':>18} "
        f"{'selected/assigned':>18} "
        f"{'selected_files':>15} "
        f"{'assigned_files':>15}"
    )
    print("-" * 100)

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        selected_df = split_df[split_df["selected_for_use"]]

        assigned_events = split_df[event_column].sum()
        selected_events = selected_df[event_column].sum()
        frac = 100.0 * selected_events / assigned_events if assigned_events else 0.0

        print(
            f"{split_name:<15} "
            f"{format_int(assigned_events):>18} "
            f"{format_int(selected_events):>18} "
            f"{frac:>17.2f}% "
            f"{len(selected_df):>15} "
            f"{len(split_df):>15}"
        )


def print_actual_subset_summaries(
    df: pd.DataFrame,
    event_column: str,
    reference_df: pd.DataFrame,
):
    train_df = df[(df["split"] == "train") & (df["selected_for_use"])]
    val_df = df[(df["split"] == "val") & (df["selected_for_use"])]
    test_df = df[(df["split"] == "test") & (df["selected_for_use"])]

    print_selection_overview(df, event_column=event_column)

    print_event_summary(
        train_df,
        title="Actual training subset summary",
        event_column=event_column,
        reference_df=reference_df,
    )

    print_event_summary(
        val_df,
        title="Actual validation subset summary",
        event_column=event_column,
        reference_df=reference_df,
    )

    print_event_summary(
        test_df,
        title="Actual test subset summary",
        event_column=event_column,
        reference_df=reference_df,
    )


def main(args):
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        merged_df = load_and_merge_metadata(
            metadata_dir=args.metadata_dir,
            metadata_file_list=args.metadata,
        )

        merged_df = filter_by_split_dataset(
            merged_df,
            split_dataset=args.split_dataset,
        )

        if merged_df.empty:
            raise ValueError(
                f"No rows left after applying split_dataset={args.split_dataset}"
            )

        print_event_summary(
            merged_df,
            title=f"Full metadata summary after split_dataset={args.split_dataset}",
            event_column="n_event",
            reference_df=merged_df,
        )

        split_df = stratified_split(
            merged_df=merged_df,
            train_frac=0.4,
            test_frac=0.4,
            val_frac=0.2,
            seed=args.seed,
        )

        split_df["selected_for_use"] = False

        split_df = apply_split_selection(
            df=split_df,
            split_name="train",
            use_frac=args.train_use_frac,
            equal_size=args.equal_size_train,
            event_column="n_event",
            seed=args.seed,
        )

        split_df = apply_split_selection(
            df=split_df,
            split_name="val",
            use_frac=args.val_use_frac,
            equal_size=args.equal_size_val,
            event_column="n_event",
            seed=args.seed + 1000,
        )

        split_df = apply_split_selection(
            df=split_df,
            split_name="test",
            use_frac=args.test_use_frac,
            equal_size=args.equal_size_test,
            event_column="n_event",
            seed=args.seed + 2000,
        )

        print_actual_subset_summaries(
            split_df,
            event_column="n_event",
            reference_df=merged_df,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # split_df.to_csv(output_path, index=False)

        # print(f"\nSaved split metadata to: {output_path}")
        selected_df = split_df[split_df["selected_for_use"]].copy()

        selected_df.to_csv(output_path, index=False)

        print(
            f"\nSaved selected metadata to: {output_path} "
            f"({len(selected_df):,} rows / {len(split_df):,} total rows"
        )

    report_text = buffer.getvalue()

    print(report_text)

    txt_path = Path(args.output).with_suffix(".txt")
    txt_path.write_text(report_text)

    print(f"Saved summary report to: {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate merged metadata CSV with split and selected-for-use flags"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output merged CSV file",
    )

    parser.add_argument(
        "-m", "--metadata",
        nargs="+",
        required=True,
        help="Metadata CSV filenames",
    )

    parser.add_argument(
        "-i", "--metadata-dir",
        required=True,
        help="Directory containing metadata CSV files",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting and selection",
    )

    parser.add_argument(
        "--split-dataset",
        choices=["neutral_hadron", "neutrino", "all"],
        default="all",
        help=(
            "Dataset subset to process: "
            "'neutral_hadron' = kaon + neutron, "
            "'neutrino' = ve + vm neutrino samples, "
            "'all' = all samples"
        ),
    )

    parser.add_argument(
        "--train-use-frac",
        type=float,
        default=0.1,
        help="Fraction of selected train target to use.",
    )

    parser.add_argument(
        "--val-use-frac",
        type=float,
        default=0.1,
        help="Fraction of selected validation target to use.",
    )

    parser.add_argument(
        "--test-use-frac",
        type=float,
        default=0.1,
        help="Fraction of selected test target to use.",
    )

    parser.add_argument(
        "--no-equal-size-train",
        action="store_false",
        dest="equal_size_train",
        default=True,
        help="Disable equal-size balancing for train split.",
    )

    parser.add_argument(
        "--no-equal-size-val",
        action="store_false",
        dest="equal_size_val",
        default=True,
        help="Disable equal-size balancing for validation split.",
    )

    parser.add_argument(
        "--no-equal-size-test",
        action="store_false",
        dest="equal_size_test",
        default=True,
        help="Disable equal-size balancing for test split.",
    )

    parser.add_argument(
        "--background-to-signal-ratio",
        type=float,
        default=5.0,
        help="Total background train event budget = ratio * train signal events",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Budget allocation mix for background train groups: "
            "alpha * equal_share + (1-alpha) * proportional_share"
        ),
    )

    args = parser.parse_args()
    main(args)