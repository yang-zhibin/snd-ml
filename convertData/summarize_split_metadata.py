import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "partition_id",
    "fold",
    "split",
    "usage",
    "is_train_eligible",
    "is_real_data",
    "class_name",
    "region",
    "particle_group",
    "n_events",
    "lumi_per_partition",
}


def bool_series(series):
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def load_split_metadata(path):
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Split metadata is missing required columns: {missing}")

    df = df.copy()
    df["n_events"] = pd.to_numeric(df["n_events"], errors="coerce").fillna(0).astype(int)
    df["lumi_per_partition"] = pd.to_numeric(
        df["lumi_per_partition"],
        errors="coerce",
    ).fillna(0.0)
    df["is_train_eligible"] = bool_series(df["is_train_eligible"])
    df["is_real_data"] = bool_series(df["is_real_data"])
    df["class_name"] = df["class_name"].fillna("").replace("", "unclassified")
    return df


def add_group_summary(rows, df, summary_type, group_columns):
    if group_columns:
        grouped = df.groupby(group_columns, dropna=False, sort=True)
    else:
        grouped = [((), df)]

    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {
            "summary_type": summary_type,
            "fold": "ALL",
            "split": "ALL",
            "usage": "ALL",
            "class_name": "ALL",
            "region": "ALL",
            "particle_group": "ALL",
            "is_train_eligible": "ALL",
            "is_real_data": "ALL",
            "n_rows": int(len(group)),
            "n_partitions": int(group["partition_id"].nunique()),
            "n_events": int(group["n_events"].sum()),
            "lumi_per_partition": float(group["lumi_per_partition"].sum()),
        }
        for column, value in zip(group_columns, key):
            row[column] = value
        rows.append(row)


def train_eligible_invariant_rows(df):
    rows = []
    train_eligible = df[df["is_train_eligible"]].copy()
    if train_eligible.empty:
        return rows

    invariant = train_eligible.groupby("partition_id").agg(
        n_fold_rows=("fold", "count"),
        n_test_folds=("split", lambda values: int((values == "test").sum())),
        n_train_val_folds=("split", lambda values: int(values.isin(["train", "val"]).sum())),
        n_train_folds=("split", lambda values: int((values == "train").sum())),
        n_val_folds=("split", lambda values: int((values == "val").sum())),
        region=("region", "first"),
        particle_group=("particle_group", "first"),
        class_name=("class_name", "first"),
        n_events=("n_events", "first"),
        lumi_per_partition=("lumi_per_partition", "first"),
    ).reset_index()
    invariant["passes_oof_invariant"] = (
        (invariant["n_fold_rows"] == 2)
        & (invariant["n_test_folds"] == 1)
        & (invariant["n_train_val_folds"] == 1)
    )

    for _, item in invariant.iterrows():
        rows.append(
            {
                "summary_type": "train_eligible_partition_invariant",
                "fold": "ALL",
                "split": "ALL",
                "usage": "ALL",
                "class_name": item["class_name"],
                "region": item["region"],
                "particle_group": item["particle_group"],
                "is_train_eligible": True,
                "is_real_data": "ALL",
                "n_rows": int(item["n_fold_rows"]),
                "n_partitions": 1,
                "n_events": int(item["n_events"]),
                "lumi_per_partition": float(item["lumi_per_partition"]),
                "partition_id": item["partition_id"],
                "n_test_folds": int(item["n_test_folds"]),
                "n_train_val_folds": int(item["n_train_val_folds"]),
                "n_train_folds": int(item["n_train_folds"]),
                "n_val_folds": int(item["n_val_folds"]),
                "passes_oof_invariant": bool(item["passes_oof_invariant"]),
            }
        )
    return rows


def build_summary(df):
    rows = []
    add_group_summary(rows, df, "overall", [])
    add_group_summary(rows, df, "by_fold_split_usage", ["fold", "split", "usage"])
    add_group_summary(rows, df, "by_fold_class_split_usage", ["fold", "class_name", "split", "usage"])
    add_group_summary(
        rows,
        df,
        "by_fold_region_particle_split_usage",
        ["fold", "region", "particle_group", "split", "usage"],
    )
    add_group_summary(rows, df, "by_class_split_usage", ["class_name", "split", "usage"])
    add_group_summary(rows, df, "by_real_data_usage", ["is_real_data", "usage"])
    add_group_summary(rows, df, "by_train_eligible_split_usage", ["is_train_eligible", "split", "usage"])
    rows.extend(train_eligible_invariant_rows(df))

    columns = [
        "summary_type",
        "fold",
        "split",
        "usage",
        "class_name",
        "region",
        "particle_group",
        "is_train_eligible",
        "is_real_data",
        "n_rows",
        "n_partitions",
        "n_events",
        "lumi_per_partition",
        "partition_id",
        "n_test_folds",
        "n_train_val_folds",
        "n_train_folds",
        "n_val_folds",
        "passes_oof_invariant",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def format_table(df, columns, max_rows=40):
    if df.empty:
        return "(empty)"
    view = df.loc[:, columns].head(max_rows)
    text = view.to_string(index=False)
    if len(df) > max_rows:
        text += f"\n... {len(df) - max_rows} more rows"
    return text


def build_report(df, summary_df):
    lines = []
    train_eligible = df[df["is_train_eligible"]]
    invariant = summary_df[summary_df["summary_type"] == "train_eligible_partition_invariant"]
    n_failed = int((invariant["passes_oof_invariant"] == False).sum()) if not invariant.empty else 0

    lines.append("Split Metadata Summary")
    lines.append("======================")
    lines.append("")
    lines.append(f"Total fold rows: {len(df):,}")
    lines.append(f"Unique partitions: {df['partition_id'].nunique():,}")
    lines.append(f"Train-eligible partitions: {train_eligible['partition_id'].nunique():,}")
    lines.append(f"Real-data fold rows: {int(df['is_real_data'].sum()):,}")
    lines.append(f"OOF invariant failures: {n_failed:,}")
    lines.append("")

    lines.append("By fold / split / usage")
    lines.append("-----------------------")
    fold_summary = summary_df[summary_df["summary_type"] == "by_fold_split_usage"]
    lines.append(format_table(fold_summary, ["fold", "split", "usage", "n_rows", "n_partitions", "n_events"]))
    lines.append("")

    lines.append("By fold / class / split / usage")
    lines.append("-------------------------------")
    class_summary = summary_df[
        (summary_df["summary_type"] == "by_fold_class_split_usage")
        & (summary_df["class_name"] != "unclassified")
    ]
    lines.append(
        format_table(
            class_summary,
            ["fold", "class_name", "split", "usage", "n_rows", "n_partitions", "n_events"],
        )
    )
    lines.append("")

    if n_failed:
        lines.append("Failed train-eligible partition invariants")
        lines.append("------------------------------------------")
        failed = invariant[invariant["passes_oof_invariant"] == False]
        lines.append(
            format_table(
                failed,
                [
                    "partition_id",
                    "class_name",
                    "n_test_folds",
                    "n_train_val_folds",
                    "n_train_folds",
                    "n_val_folds",
                ],
            )
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize fold-aware split metadata.")
    parser.add_argument("-i", "--input", required=True, help="Input split metadata CSV")
    parser.add_argument("-o", "--output", required=True, help="Output detailed summary CSV")
    parser.add_argument("-r", "--report", required=True, help="Output human-readable text report")
    return parser.parse_args()


def main():
    args = parse_args()
    split_df = load_split_metadata(args.input)
    summary_df = build_summary(split_df)
    report_text = build_report(split_df, summary_df)

    output = Path(args.output)
    report = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output, index=False)
    report.write_text(report_text, encoding="utf-8")

    print(f"Wrote split summary CSV: {output} ({len(summary_df)} rows)")
    print(f"Wrote split summary report: {report}")


if __name__ == "__main__":
    main()
