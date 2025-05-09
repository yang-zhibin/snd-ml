import pandas as pd

def main():
    df = pd.read_csv("/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/csv_v2/ve_baseline_eff_yield_df.csv")
    filtered_rows = []
    last_yield = -float('inf') 
    for _, row in df.iterrows():
        current_yield = row["ve_yield"]
        if current_yield - last_yield >= 2:
            filtered_rows.append(row)
            last_yield = current_yield

    # Convert to DataFrame
    filtered_df = pd.DataFrame(filtered_rows)

    # Optionally save to a new CSV
    filtered_df.to_csv("ve_cut_scores.csv", index=False)

if __name__ == "__main__":
    main()