import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import ROOT

ROOT.ROOT.EnableImplicitMT()  # parallelize RDataFrame where possible

REQUIRED_BRANCHES = ("preSelect_vetoFree", "preSelect_vetoTagged")
TREE_NAME = "sndData"


def count_flags(root_path: str, tree: str = TREE_NAME):
    """
    Return counts of entries where each REQUIRED_BRANCH is true/non-zero.
    If a branch or file is missing, raise RuntimeError with a helpful message.
    """
    # Construct an RDataFrame on the tree/file
    try:
        rdf = ROOT.RDataFrame(tree, root_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open '{root_path}' with tree '{tree}': {e}")

    # Verify required branches exist
    cols = {str(c) for c in rdf.GetColumnNames()}
    missing = [b for b in REQUIRED_BRANCHES if b not in cols]
    if missing:
        raise RuntimeError(
            f"Missing branch(es) {missing} in tree '{tree}' of file '{root_path}'"
        )

    # Count entries where the branch evaluates to true (or != 0 for ints)
    # Using explicit != 0 guards both bool and integer branches
    counts = {}
    for b in REQUIRED_BRANCHES:
        cnt = int(rdf.Filter(f"{b} != 0").Count().GetValue())
        counts[b] = cnt

    return counts["preSelect_vetoFree"], counts["preSelect_vetoTagged"]


def main(args):
    csv_path = Path(args.csv_file)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if "preSelect_path" not in df.columns:
        print("CSV must contain a 'preSelect_path' column.", file=sys.stderr)
        sys.exit(1)

    # Prepare output columns
    if "vetoFree_count" not in df.columns:
        df["vetoFree_count"] = pd.NA
    if "vetoTagged_count" not in df.columns:
        df["vetoTagged_count"] = pd.NA

    # Iterate rows and fill counts
    for idx, row in df.iterrows():
        root_path = row["preSelect_path"]
        if pd.isna(root_path) or str(root_path).strip() == "":
            continue

        try:
            vf, vt = count_flags(str(root_path), TREE_NAME)
            df.at[idx, "vetoFree_count"] = vf
            df.at[idx, "vetoTagged_count"] = vt
            
            # If vetoFree_count < 2 → set all "vetoFree*" columns to NA
            if vf < 2:
                veto_free_cols = [c for c in df.columns if c.startswith("vetoFree")]
                for col in veto_free_cols:
                    df.at[idx, col] = pd.NA
                if "digi_path" in df.columns:
                    print(f"vetoFree drop: {row['digi_path']}")

            # If vetoTagged_count < 2 → set all "vetoTagged*" columns to NA
            if vt < 2:
                veto_tagged_cols = [c for c in df.columns if c.startswith("vetoTagged")]
                for col in veto_tagged_cols:
                    df.at[idx, col] = pd.NA
                if "digi_path" in df.columns:
                    print(f"vetoTagged drop: {row['digi_path']}")
                    
                    
        except Exception as e:
            # keep going; log per-row error
            print(f'error raised when processing {row["preSelect_path"]}')

        
    # Overwrite the CSV in place
    
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV written to: {csv_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--csv_file", dest="csv_file", help="csv file", required=True)
    args = parser.parse_args()
    main(args)


