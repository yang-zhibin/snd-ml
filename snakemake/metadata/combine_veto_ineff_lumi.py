import pandas as pd

def main():
    # Read lumi DataFrame
    df_lumi = pd.read_csv("./SND_lumi.csv")
    df_lumi['From'] = pd.to_datetime(df_lumi['From'])
    df_lumi['To'] = pd.to_datetime(df_lumi['To'])

    # Read inefficiency periods DataFrame
    df_ineff = pd.read_csv("./SND_veto_ineff.csv")
    df_ineff['From'] = pd.to_datetime(df_ineff['From'])
    df_ineff['To'] = pd.to_datetime(df_ineff['To'])

    # Add ineff column, defaulting to None
    df_lumi['ineff'] = None

    # Assign inefficiency values based on containment
    for i, run_row in df_lumi.iterrows():
        for _, ineff_row in df_ineff.iterrows():
            if run_row['From'] >= ineff_row['From'] and run_row['To'] <= ineff_row['To']:
                df_lumi.at[i, 'ineff'] = ineff_row['ineff']
                break  # Stop after finding the first matching period

    # Save or return the result
    df_lumi.to_csv("lumi_with_ineff.csv", index=False)
    print("Saved combined DataFrame to lumi_with_ineff.csv")

if __name__ == "__main__":
    main()
