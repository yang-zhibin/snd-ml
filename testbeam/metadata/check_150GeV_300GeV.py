import pandas as pd
import ROOT

ROOT.gROOT.SetBatch(True)

# --- config ---
METADATA_CSV = "/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/metadata/updated/MC_data_testbeam2024_metadata.csv"
TREE_NAME = "sndData"

Z_BRANCH = "z"
SCIFI_BRANCH = "count_scifi"
DENSITY_BRANCH = "density_scifi"


def means_from_root(path):
    df = ROOT.RDataFrame(TREE_NAME, path)
    return (
        float(df.Mean(Z_BRANCH).GetValue()),
        float(df.Mean(SCIFI_BRANCH).GetValue()),
        float(df.Mean(DENSITY_BRANCH).GetValue()),
    )


def plot_distributions(df_means, energy_gev):
    # Handle degenerate ranges (all values equal)
    def make_hist(name, title, series):
        vmin, vmax = float(series.min()), float(series.max())
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        return ROOT.TH1F(name, title, 40, vmin, vmax)

    h_z = make_hist(
        f"h_z_{energy_gev}",
        f"{energy_gev} GeV e- (MC_data);z_mean;files",
        df_means["z_mean"],
    )
    h_scifi = make_hist(
        f"h_scifi_{energy_gev}",
        f"{energy_gev} GeV e- (MC_data);count_scifi_mean;files",
        df_means["scifi_count_mean"],
    )
    h_dens = make_hist(
        f"h_dens_{energy_gev}",
        f"{energy_gev} GeV e- (MC_data);density_scifi_mean;files",
        df_means["density_mean"],
    )

    for _, r in df_means.iterrows():
        h_z.Fill(r["z_mean"])
        h_scifi.Fill(r["scifi_count_mean"])
        h_dens.Fill(r["density_mean"])

    c = ROOT.TCanvas(f"c_{energy_gev}", "", 1200, 400)
    c.Divide(3, 1)

    c.cd(1); h_z.Draw("HIST")
    c.cd(2); h_scifi.Draw("HIST")
    c.cd(3); h_dens.Draw("HIST")

    c.SaveAs(f"means_{energy_gev}GeV_MC_data.png")


meta = pd.read_csv(METADATA_CSV)


def process_energy(energy_gev):
    energy_str = f"{energy_gev}GeV"
    sel = meta[
        (meta["beam_energy"] == energy_str) &
        (meta["beam_type"] == "e-") &
        (meta["data_type"] == "MC_data")
    ]

    print(sel)

    rows = []
    for _, r in sel.iterrows():
        z_mean, scifi_mean, dens_mean = means_from_root(r["feature_path"])
        rows.append({
            "feature_path": r["feature_path"],
            "z_mean": z_mean,
            "scifi_count_mean": scifi_mean,
            "density_mean": dens_mean,
        })

    out = pd.DataFrame(rows)
    out_cut = out[out["z_mean"] < 327.5]
    
    print(f"\n=== {energy_gev} GeV electrons (MC_data) ===")
    print(out.to_string(index=False))

    if len(out) > 0:
        plot_distributions(out, energy_gev)

    return out_cut

# --- run ---
out_cut_150 = process_energy(150)
out_cut_300 = process_energy(300)

#combine out_cut, save to ./raw/first_peak_150GeV_300GeV.csv
out_all = pd.concat([out_cut_150, out_cut_300], ignore_index=True)
out_path = "./raw/first_peak_150GeV_300GeV.csv"
out_all.to_csv(out_path, index=False)

print(f"\nSaved combined table to: {out_path}")