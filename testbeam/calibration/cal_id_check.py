import pandas as pd
import numpy as np
from pathlib import Path

KEY_COLS = ["board_id", "fe_id", "channel", "tac"]
DEFAULT_VAL_COLS = ["a", "b", "c", "chi2", "d", "dof", "e"]


def _require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}. Present: {list(df.columns)}")


def _normalize_key_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Force integers for IDs when possible (keeps comparisons stable)
    for c in ["board_id", "fe_id", "channel", "tac"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def check_pair(lab_csv: str, tb_csv: str, tag: str, outdir: str = "id_checks",
               value_cols: list[str] | None = None):
    outdir = Path(outdir) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    lab = _normalize_key_types(pd.read_csv(lab_csv))
    tb = _normalize_key_types(pd.read_csv(tb_csv))

    _require_cols(lab, KEY_COLS, f"{tag} LAB")
    _require_cols(tb, KEY_COLS, f"{tag} TESTBEAM")

    # Duplicate key detection
    dup_lab = lab[lab.duplicated(KEY_COLS, keep=False)].sort_values(KEY_COLS)
    dup_tb = tb[tb.duplicated(KEY_COLS, keep=False)].sort_values(KEY_COLS)

    if len(dup_lab):
        dup_lab.to_csv(outdir / "duplicates_lab.csv", index=False)
        print(f"[{tag}] LAB duplicate keys: {len(dup_lab)} rows -> {outdir/'duplicates_lab.csv'}")
    else:
        print(f"[{tag}] LAB duplicate keys: none")

    if len(dup_tb):
        dup_tb.to_csv(outdir / "duplicates_testbeam.csv", index=False)
        print(f"[{tag}] TESTBEAM duplicate keys: {len(dup_tb)} rows -> {outdir/'duplicates_testbeam.csv'}")
    else:
        print(f"[{tag}] TESTBEAM duplicate keys: none")

    # Key sets
    lab_keys = set(map(tuple, lab[KEY_COLS].dropna().astype(int).to_numpy()))
    tb_keys  = set(map(tuple, tb[KEY_COLS].dropna().astype(int).to_numpy()))

    only_lab = sorted(lab_keys - tb_keys)
    only_tb  = sorted(tb_keys - lab_keys)

    # Save missing keys as CSV
    if only_lab:
        pd.DataFrame(only_lab, columns=KEY_COLS).to_csv(outdir / "keys_only_in_lab.csv", index=False)
        print(f"[{tag}] Keys only in LAB: {len(only_lab)} -> {outdir/'keys_only_in_lab.csv'}")
    else:
        print(f"[{tag}] Keys only in LAB: none")

    if only_tb:
        pd.DataFrame(only_tb, columns=KEY_COLS).to_csv(outdir / "keys_only_in_testbeam.csv", index=False)
        print(f"[{tag}] Keys only in TESTBEAM: {len(only_tb)} -> {outdir/'keys_only_in_testbeam.csv'}")
    else:
        print(f"[{tag}] Keys only in TESTBEAM: none")

    # Inner-join on key for value diffs (optional)
    if value_cols is None:
        value_cols = [c for c in DEFAULT_VAL_COLS if c in lab.columns and c in tb.columns]

    if value_cols:
        lab_sel = lab[KEY_COLS + value_cols].copy()
        tb_sel  = tb[KEY_COLS + value_cols].copy()

        merged = lab_sel.merge(tb_sel, on=KEY_COLS, suffixes=("_lab", "_tb"), how="inner")
        merged.to_csv(outdir / "matched_rows.csv", index=False)

        # Compute diffs
        diff = merged[KEY_COLS].copy()
        for c in value_cols:
            diff[c + "_diff"] = merged[c + "_lab"] - merged[c + "_tb"]

        # Flag any non-zero diffs (or NaN mismatch)
        mask = np.zeros(len(diff), dtype=bool)
        for c in value_cols:
            dl = merged[c + "_lab"]
            dt = merged[c + "_tb"]
            mask |= (dl.isna() ^ dt.isna())  # one side NaN
            both = ~(dl.isna() | dt.isna())
            mask |= (both & (dl != dt))

        diff_all_path = outdir / "value_differences_all.csv"
        diff.to_csv(diff_all_path, index=False)

        diff_changed = diff[mask]
        diff_changed_path = outdir / "value_differences_changed_only.csv"
        diff_changed.to_csv(diff_changed_path, index=False)

        print(f"[{tag}] Matched keys: {len(merged)} -> {outdir/'matched_rows.csv'}")
        print(f"[{tag}] Value diffs (all): {len(diff)} -> {diff_all_path}")
        print(f"[{tag}] Value diffs (changed only): {len(diff_changed)} -> {diff_changed_path}")
    else:
        print(f"[{tag}] No shared value columns to diff; key-check only.")



def count_ids_per_board(csv_path, tag):
    df = pd.read_csv(csv_path)

    # Check required columns
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    # Keep only key columns and drop duplicates
    unique_ids = df[KEY_COLS].dropna().drop_duplicates()

    # Count per board
    counts = (
        unique_ids
        .groupby("board_id")
        .size()
        .reset_index(name="n_unique_ids")
        .sort_values("board_id")
    )

    print(f"\n=== {tag} ===")
    print(counts)

    # Save
    outdir = Path("board_id_counts")
    outdir.mkdir(exist_ok=True)
    counts.to_csv(outdir / f"{tag}_counts.csv", index=False)



def main():
    # Edit filenames as needed
    # check_pair("/eos/user/z/zhibin/TestBeam/run_000041/qdc_cal.csv", "/eos/experiment/sndlhc/raw_data/testbeam_24/run_100890/qdc_cal.csv", tag="qdc", outdir="id_checks")
    #check_pair("/eos/user/z/zhibin/TestBeam/run_000041/tdc_cal.csv", "/eos/experiment/sndlhc/raw_data/testbeam_24/run_100890/tdc_cal.csv", tag="tac", outdir="id_checks")

    
    count_ids_per_board("/eos/user/z/zhibin/TestBeam/run_000041/qdc_cal.csv", "lab_qdc")
    count_ids_per_board("/eos/experiment/sndlhc/raw_data/testbeam_24/run_100890/qdc_cal.csv", "testbeam_qdc")


if __name__ == "__main__":
    main()
