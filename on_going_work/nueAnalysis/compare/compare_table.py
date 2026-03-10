#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_nue_table(path: str, year: str) -> pd.DataFrame:
    """
    Load the first CSV (nueAnalysis_table.csv).

    Expected columns:
      Step, cut, Events_2022, Eff_2022_percent, Events_2024, Eff_2024_percent

    year must be '2022' or '2024'.
    """
    df = pd.read_csv(path)

    events_col = f"Events_{year}"
    cumeff_col = f"Eff_{year}_percent"

    required = {"Step", "cut", events_col, cumeff_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    out = df[["Step", "cut", events_col, cumeff_col]].copy()
    out = out.rename(
        columns={
            "cut": "cut_first",
            events_col: "count_first",
            cumeff_col: "cumeff_first_percent",
        }
    )

    out["Step"] = out["Step"].astype(int)
    out["count_first"] = pd.to_numeric(out["count_first"], errors="coerce")
    out["cumeff_first_percent"] = pd.to_numeric(out["cumeff_first_percent"], errors="coerce")

    # Step-wise / relative efficiency in percent from counts
    out = out.sort_values("Step").reset_index(drop=True)
    out["releff_first_percent"] = 100.0
    for i in range(1, len(out)):
        prev = out.loc[i - 1, "count_first"]
        curr = out.loc[i, "count_first"]
        out.loc[i, "releff_first_percent"] = 100.0 * curr / prev if prev else float("nan")

    return out


def combine_first_rows(df_first: pd.DataFrame, steps, new_step: int, new_name: str) -> pd.Series:
    """
    Combine several consecutive rows from the first table into one comparison row.

    Example:
      steps=[6,7]  -> relative efficiency = count(step7)/count(step5)
                   -> cumulative efficiency = cumulative efficiency at step7
                   -> count = count(step7)
    """
    sub = df_first[df_first["Step"].isin(steps)].sort_values("Step").copy()
    if sub.empty:
        raise ValueError(f"No rows found for steps {steps} in first table")

    first_step = min(steps)
    prev_row = df_first[df_first["Step"] == first_step - 1]
    if prev_row.empty:
        raise ValueError(f"Cannot combine steps {steps}: missing previous step {first_step - 1}")

    prev_count = prev_row.iloc[0]["count_first"]
    final_row = sub.iloc[-1]

    releff = 100.0 * final_row["count_first"] / prev_count if prev_count else float("nan")

    return pd.Series(
        {
            "compare_key": new_name,
            "step_first": f"{steps[0]}-{steps[-1]}",
            "cut_first": " + ".join(sub["cut_first"].tolist()),
            "count_first": final_row["count_first"],
            "cumeff_first_percent": final_row["cumeff_first_percent"],
            "releff_first_percent": releff,
        }
    )


def load_my_cut_table(path: str) -> pd.DataFrame:
    """
    Load the second CSV (my_cut_table.csv).

    Expected columns:
      step, cut, count, cumulative_efficiency, relative_efficiency

    The efficiencies are fractions, not percent.
    They are converted here to percent.
    """
    df = pd.read_csv(path)

    required = {"step", "cut", "count", "cumulative_efficiency", "relative_efficiency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    out = df.copy()
    out["step"] = out["step"].astype(int)
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out["cumulative_efficiency"] = pd.to_numeric(out["cumulative_efficiency"], errors="coerce")
    out["relative_efficiency"] = pd.to_numeric(out["relative_efficiency"], errors="coerce")

    # Convert fraction -> percent
    out["cumeff_second_percent"] = 100.0 * out["cumulative_efficiency"]
    out["releff_second_percent"] = 100.0 * out["relative_efficiency"]

    out = out.rename(
        columns={
            "step": "step_second",
            "cut": "cut_second",
            "count": "count_second",
        }
    )

    return out[["step_second", "cut_second", "count_second", "cumeff_second_percent", "releff_second_percent"]]


def build_first_comparison_table(df_first: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comparison-ready version of the first table.

    Mapping used:
      Total                  <- step 1
      stableBeam             <- step 2
      IP1BunchCrossing       <- step 3
      PreEvtClockCycle100    <- step 4
      avgScifiFiducial       <- step 5
      USBarsVeto             <- steps 6+7 combined
      noVetoHit              <- step 8
      consecutiveSciFiHits   <- step 9
      SciFiContinuity        <- step 10
      USPlaneHit_0_1         <- step 11
      SciFiHit35             <- step 12
      TotalUSQDC600          <- step 13
      NoHitLastDS            <- step 14

    Note:
      The second table has no explicit counterpart of first-table step 13
      ("Total US QDC > 600"), so it is left out here.
    """
    rows = []

    direct_map = {
        "Total": 1,
        "stableBeam": 2,
        "IP1BunchCrossing": 3,
        "PreEvtClockCycle100": 4,
        "avgScifiFiducial": 5,
        "noVetoHit": 8,
        "consecutiveSciFiHits": 9,
        "SciFiContinuity": 10,
        "USPlaneHit_0_1": 11,
        "SciFiHit35": 12,
        "TotalUSQDC600":13,
        "NoHitLastDS": 14,
    }

    for key, step in direct_map.items():
        row = df_first[df_first["Step"] == step]
        if row.empty:
            raise ValueError(f"Missing step {step} in first table")
        r = row.iloc[0]
        rows.append(
            {
                "compare_key": key,
                "step_first": str(step),
                "cut_first": r["cut_first"],
                "count_first": r["count_first"],
                "cumeff_first_percent": r["cumeff_first_percent"],
                "releff_first_percent": r["releff_first_percent"],
            }
        )

    # Combined row for USBarsVeto = first-table steps 6 and 7 together
    rows.append(
        combine_first_rows(
            df_first,
            steps=[6, 7],
            new_step=67,
            new_name="USBarsVeto",
        ).to_dict()
    )

    out = pd.DataFrame(rows)

    # order like the second table
    order = [
        "Total",
        "stableBeam",
        "IP1BunchCrossing",
        "PreEvtClockCycle100",
        "avgScifiFiducial",
        "USBarsVeto",
        "noVetoHit",
        "consecutiveSciFiHits",
        "SciFiContinuity",
        "USPlaneHit_0_1",
        "SciFiHit35",
        "TotalUSQDC600",
        "NoHitLastDS",
    ]
    out["order"] = out["compare_key"].map({k: i for i, k in enumerate(order)})
    out = out.sort_values("order").drop(columns="order").reset_index(drop=True)

    return out


def compare_tables(first_csv: str, second_csv: str, year: str, output_csv: str) -> pd.DataFrame:
    df_first = load_nue_table(first_csv, year)
    df_first_cmp = build_first_comparison_table(df_first)

    df_second = load_my_cut_table(second_csv)
    df_second["compare_key"] = df_second["cut_second"]

    merged = pd.merge(df_first_cmp, df_second, on="compare_key", how="inner")

    merged["count_diff"] = merged["count_second"] - merged["count_first"]
    merged["count_ratio_second_over_first"] = merged["count_second"] / merged["count_first"]

    merged["cumeff_diff_percent_points"] = (
        merged["cumeff_second_percent"] - merged["cumeff_first_percent"]
    )
    merged["releff_diff_percent_points"] = (
        merged["releff_second_percent"] - merged["releff_first_percent"]
    )

    cols = [
        "compare_key",
        "step_first",
        "cut_first",
        "count_first",
        "cumeff_first_percent",
        "releff_first_percent",
        "step_second",
        "cut_second",
        "count_second",
        "cumeff_second_percent",
        "releff_second_percent",
        "count_diff",
        "count_ratio_second_over_first",
        "cumeff_diff_percent_points",
        "releff_diff_percent_points",
    ]
    merged = merged[cols]

    merged.to_csv(output_csv, index=False)
    return merged



def fmt(x):
    if pd.isna(x):
        return ""
    if isinstance(x, str):
        return x
    x = float(x)
    if abs(x) < 1e-2 and x != 0:
        return f"{x:.2e}"
    if abs(x - round(x)) < 1e-12 and abs(x) >= 1:
        return f"{int(round(x))}"
    return f"{x:.6f}".rstrip("0").rstrip(".")
def make_table_plot(df: pd.DataFrame, output_png: str, title: str, columns_to_keep: list[str]):
    
    display_df = df[[c for c in columns_to_keep if c in df.columns]].copy()

    # Rename columns
    display_df = display_df.rename(columns={
        "count_first": "nueAnalysis count",
        "cumeff_first_percent": "nueAnalysis cumulative eff (%)",
        "releff_first_percent": "nueAnalysis relative eff (%)",

        "count_second": "my_cut_flow count",
        "cumeff_second_percent": "my_cut_flow cumulative eff (%)",
        "releff_second_percent": "my_cut_flow relative eff (%)",

        "count_ratio_second_over_first": "my_cut_flow / nueAnalysis",
        "cumeff_diff_percent_points": "Δ cumulative eff (%)",
        "releff_diff_percent_points": "Δ relative eff (%)",
        "compare_key": "cut"
    })

    for col in display_df.columns:
        display_df[col] = display_df[col].map(fmt)

    nrows, ncols = display_df.shape

    # Larger figure for long headers
    fig_width = max(20, 2.5 * ncols)
    fig_height = max(4, 0.55 * (nrows + 2))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=14)

    # Wider columns
    col_widths = []
    for c in display_df.columns:
        if c == "cut":
            col_widths.append(0.25)
        else:
            col_widths.append(0.18)

    # normalize
    s = sum(col_widths)
    col_widths = [w / s for w in col_widths]

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
        if col == 0:
            cell._loc = "left"

    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Compare two cutflow CSV tables.")
    parser.add_argument(
        "--first",
        default="nueAnalysis_table.csv",
        help="Path to first CSV (default: nueAnalysis_table.csv)",
    )
    parser.add_argument(
        "--second",
        default="my_cut_table.csv",
        help="Path to second CSV (default: my_cut_table.csv)",
    )
    parser.add_argument(
        "--year",
        choices=["2022", "2024"],
        default="2024",
        help="Which year from the first table to compare against (default: 2024)",
    )
    parser.add_argument(
        "--output",
        default="cutflow_comparison.csv",
        help="Output comparison CSV (default: cutflow_comparison.csv)",
    )
    
    parser.add_argument("--png-output", default="cutflow_comparison_table.png")

    args = parser.parse_args()

    df = compare_tables(args.first, args.second, args.year, args.output)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print("\nComparison done.")
    print(f"Saved: {args.output}\n")
    print(df)
    
    cum_cols = [
        "compare_key",
        "count_first",
        "cumeff_first_percent",
        "count_second",
        "cumeff_second_percent",
        "count_ratio_second_over_first",
        "cumeff_diff_percent_points",
    ]
    make_table_plot(
        df,
        "cutflow_comparison_cumeff_table.png",
        f"Cutflow comparison cumulative efficiency ({args.year})",
        cum_cols,
    )

    # relative efficiency table
    rel_cols = [
        "compare_key",
        "count_first",
        "releff_first_percent",
        "count_second",
        "releff_second_percent",
        "count_ratio_second_over_first",
        "releff_diff_percent_points",
    ]
    make_table_plot(
        df,
        "cutflow_comparison_releff_table.png",
        f"Cutflow comparison relative efficiency ({args.year})",
        rel_cols,
    )
    
    
    


if __name__ == "__main__":
    main()