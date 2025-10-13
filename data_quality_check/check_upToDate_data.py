import pandas as pd
import os
from tqdm import tqdm
import ROOT

def check_preSelection(metadata, max_rows=None):
    """
    For each row in metadata:
      - open the CSV at 'preCutEff_path'
      - check if the 'cut' column contains '1_scifi>200'
    Optionally limit processing to first `max_rows` entries.
    """
    results = []
    stats = {
        "passed": 0,
        "not_found": 0,
        "missing_file": 0,
        "no_cut_column": 0,
        "read_error": 0,
    }

    # tqdm progress bar
    for i, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Checking pre-selection"):
        if max_rows is not None and i >= max_rows:
            break

        preCutEff_path = row.get("preCutEff_path", None)

        if not preCutEff_path or not os.path.exists(preCutEff_path):
            results.append(False)
            stats["missing_file"] += 1
            continue

        try:
            df = pd.read_csv(preCutEff_path)
        except Exception:
            results.append(False)
            stats["read_error"] += 1
            continue

        if "cut" not in df.columns:
            results.append(False)
            stats["no_cut_column"] += 1
            continue

        passed = any(df["cut"].astype(str).str.contains(r"\b1_scifi>200\b"))
        if passed:
            stats["passed"] += 1
        else:
            stats["not_found"] += 1

        results.append(passed)

    # Print summary
    total = len(results)
    print("\n📊 === Pre-selection Summary ===")
    print(f"Checked files:         {total}")
    print(f"✅ Passed:              {stats['passed']}")
    print(f"⚠️  Not found:          {stats['not_found']}")
    print(f"❌ Missing files:       {stats['missing_file']}")
    print(f"🧩 No 'cut' column:     {stats['no_cut_column']}")
    print(f"💥 Read errors:         {stats['read_error']}")
    print(f"──────────────────────────────")
    print(f"✔️  Success rate:       {stats['passed'] / total * 100:.1f}%")

    return stats





def check_afterSelection(metadata, veto_mode, model_name, max_rows=1e6):
    flag = f"preSelect_{veto_mode}"
    pred_col = f"{veto_mode}_prediction_{model_name}_output_path"

    ok, mismatch, error = 0, 0, 0

    for i, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Checking after-selection"):
        if max_rows is not None and i >= max_rows:
            break

        pre_path = row.get("preSelect_path", None)
        pred_path = row.get(pred_col, None)
        if not pre_path or not pred_path:
            error += 1
            continue

        try:
            f_pre = ROOT.TFile.Open(pre_path)
            tree_pre = f_pre.Get("sndData")
            rdf_pre = ROOT.RDataFrame(tree_pre)
            n_pre_pass = int(rdf_pre.Filter(f"{flag} == 1").Count().GetValue())
            f_pre.Close()

            f_pred = ROOT.TFile.Open(pred_path)
            tree_pred = f_pred.Get("sndData")
            rdf_pred = ROOT.RDataFrame(tree_pred)
            n_pred_total = int(rdf_pred.Count().GetValue())
            f_pred.Close()

            if n_pre_pass == n_pred_total:
                ok += 1
            else:
                mismatch += 1
        except Exception:
            error += 1

    total = ok + mismatch + error
    print("\n=== File-level summary ===")
    print(f"Total files checked : {total}")
    print(f"OK (match)           : {ok}")
    print(f"MISMATCH             : {mismatch}")
    print(f"ERROR                : {error}")

    return {
        "total": total,
        "ok": ok,
        "mismatch": mismatch,
        "error": error
    }

def main():
    #csv_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/real_data_2024_skim_runs_metadata.csv'
    csv_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_kaon_FTFP_BERT_metadata_subset.csv'
    metadata = pd.read_csv(csv_path)

    # Only check first 100 rows (set max_rows=None to check all)
    #check_preSelection(metadata, max_rows=10000)
    check_afterSelection(metadata, veto_mode="vetoTagged", model_name="GravNet_v4")
    


if __name__ == "__main__":
    main()
