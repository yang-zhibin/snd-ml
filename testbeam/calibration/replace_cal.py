import pandas as pd
from pathlib import Path

KEY = ["board_id", "fe_id", "channel", "tac"]
PARAMS_DEFAULT_qdc = ["a", "b", "c", "chi2", "d", "dof", "e"]
PARAMS_DEFAULT_tdc = ["tdc","a", "b", "c", "chi2", "d", "dof", "e"]


def dedup_by_key(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """
    Returns df with at most 1 row per KEY (globally).
    policy:
      - "keep_first"
      - "keep_last"
      - "min_chi2"   (requires column chi2)
      - "max_dof"    (requires column dof)
    """
    if policy in ("keep_first", "keep_last"):
        return df.drop_duplicates(KEY, keep=policy.split("_")[1])

    if policy == "min_chi2":
        if "chi2" not in df.columns:
            raise ValueError("Policy min_chi2 requires 'chi2' column")
        tmp = df.copy()
        tmp["chi2"] = pd.to_numeric(tmp["chi2"], errors="coerce")
        tmp = tmp.sort_values(KEY + ["chi2"], ascending=True)  # smallest chi2 first
        return tmp.drop_duplicates(KEY, keep="first")

    if policy == "max_dof":
        if "dof" not in df.columns:
            raise ValueError("Policy max_dof requires 'dof' column")
        tmp = df.copy()
        tmp["dof"] = pd.to_numeric(tmp["dof"], errors="coerce")
        tmp = tmp.sort_values(KEY + ["dof"], ascending=[True, True, True, True, False])  # largest dof first
        return tmp.drop_duplicates(KEY, keep="first")

    raise ValueError(f"Unknown dedup policy: {policy}")


def replace_params_for_board(
    lab_csv: str,
    tb_csv: str,
    out_csv: str,
    board_id: int = 14,
    params: list[str] | None = None,
    dedup_policy: str = "min_chi2",   # <- change to keep_last if you prefer
):
    lab = pd.read_csv(lab_csv)
    tb  = pd.read_csv(tb_csv)
    if "tdc" in lab_csv:
        PARAMS_DEFAULT = PARAMS_DEFAULT_tdc
    else:
        PARAMS_DEFAULT = PARAMS_DEFAULT_qdc
    # checks
    for name, df, path in [("LAB", lab, lab_csv), ("TESTBEAM", tb, tb_csv)]:
        missing = [c for c in KEY if c not in df.columns]
        if missing:
            raise ValueError(f"{name} file {path} missing key cols {missing}")

    if params is None:
        params = [c for c in PARAMS_DEFAULT if c in lab.columns and c in tb.columns]
    else:
        for c in params:
            if c not in lab.columns or c not in tb.columns:
                raise ValueError(f"Param '{c}' missing in lab or testbeam")

    # --- Deduplicate only board_id == 14 area (safer & faster) ---
    lab_b = lab[lab["board_id"] == board_id].copy()
    tb_b  = tb[tb["board_id"] == board_id].copy()

    # Save raw duplicates for inspection (optional but useful)
    outdir = Path(out_csv).parent
    outdir.mkdir(parents=True, exist_ok=True)

    lab_dups = lab_b[lab_b.duplicated(KEY, keep=False)]
    tb_dups  = tb_b[tb_b.duplicated(KEY, keep=False)]

    if not lab_dups.empty:
        lab_dups.sort_values(KEY).to_csv(outdir / f"duplicates_lab_board{board_id}.csv", index=False)
    if not tb_dups.empty:
        tb_dups.sort_values(KEY).to_csv(outdir / f"duplicates_testbeam_board{board_id}.csv", index=False)

    # Dedup according to policy
    lab_b_d = dedup_by_key(lab_b, dedup_policy)
    tb_b_d  = dedup_by_key(tb_b, dedup_policy)

    # Align lab params onto testbeam keys for this board
    lab_keep = lab_b_d[KEY + params].copy()
    merged = tb_b_d.merge(lab_keep, on=KEY, how="left", suffixes=("", "_lab"))

    # rows where lab exists
    has_match = merged[params[0] + "_lab"].notna() if params else pd.Series(False, index=merged.index)

    # replace params
    for p in params:
        merged.loc[has_match, p] = merged.loc[has_match, p + "_lab"]
        merged.drop(columns=[p + "_lab"], inplace=True)

    # Write the modified board back into the full testbeam DF:
    # easiest: remove old board rows, append merged, then sort if you like.
    tb_out = pd.concat([tb[tb["board_id"] != board_id], merged], ignore_index=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    tb_out.to_csv(out_csv, index=False)

    # Save unmatched keys
    unmatched = merged.loc[~has_match, KEY].copy()
    if not unmatched.empty:
        unmatched_path = outdir / f"unmatched_board{board_id}.csv"
        unmatched.to_csv(unmatched_path, index=False)

    print(f"\n[{Path(out_csv).name}] board_id == {board_id}")
    print(f"  dedup_policy:           {dedup_policy}")
    print(f"  TB rows (raw):          {len(tb_b)}")
    print(f"  TB rows (deduped):      {len(tb_b_d)}")
    print(f"  LAB rows (raw):         {len(lab_b)}")
    print(f"  LAB rows (deduped):     {len(lab_b_d)}")
    print(f"  Matched & replaced:     {int(has_match.sum())}")
    print(f"  Unmatched kept as-is:   {len(unmatched)}")
    print(f"Saved -> {out_csv}")
    if not unmatched.empty:
        print(f"Unmatched IDs -> {unmatched_path}")
    if not lab_dups.empty or not tb_dups.empty:
        print(f"Duplicates saved next to output (board {board_id}).")


def main():
    # QDC
    replace_params_for_board(
        lab_csv="/eos/user/z/zhibin/TestBeam/run_000041/qdc_cal.csv",
        tb_csv="/eos/experiment/sndlhc/raw_data/testbeam_24/run_100890/qdc_cal.csv",
        out_csv="./cal_csv/qdc_cal.csv",
        board_id=14,
    )

    # TAC
    replace_params_for_board(
        lab_csv="/eos/user/z/zhibin/TestBeam/run_000041/tdc_cal.csv",
        tb_csv="/eos/experiment/sndlhc/raw_data/testbeam_24/run_100890/tdc_cal.csv",
        out_csv="./cal_csv/tdc_cal.csv",
        board_id=14,
    )


if __name__ == "__main__":
    main()

    # run_comparison("/eos/user/z/zhibin/TestBeam/run_000041/qdc_cal.csv",
    #                "/eos/experiment/sndlhc/raw_data/testbeam_24/run_100890/qdc_cal.csv",
    #                "qdc")

    # run_comparison("/eos/user/z/zhibin/TestBeam/run_000041/tdc_cal.csv",
    #                "/eos/experiment/sndlhc/raw_data/testbeam_24/run_100890/tdc_cal.csv",
    #                "tac")
