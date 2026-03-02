from pathlib import Path
import pandas as pd
import ROOT
import random
import os
from tqdm import tqdm



def load_all_metadata(metadata_dir):
    """
    Reads all metadata CSV files in a directory and groups them by:
        MC_2024, DATA_2024, MC_2023, DATA_2023

    Returns
    -------
    dict[str, pd.DataFrame]
        {
          "MC_2024": df,
          "DATA_2024": df,
          "MC_2023": df,
          "DATA_2023": df
        }
    """

    metadata_dir = Path(metadata_dir)

    groups = {
        "MC_2024": [],
        "DATA_2024": [],
        "MC_2023": [],
        "DATA_2023": []
    }

    for csv_file in metadata_dir.glob("*.csv"):
        name = csv_file.name.lower()

        # ---- Classify file ----
        if "mc" in name and "2024" in name:
            key = "MC_2024"
        elif "mc" in name and "2023" in name:
            key = "MC_2023"
        elif "real" in name:
            if "2024" in name or "testbeam_24" in name:
                key = "DATA_2024"
            elif "june2023" in name:
                key = "DATA_2023"
            else:
                continue
        else:
            continue

        print(f"📄 Loading {csv_file.name} → {key}")
        df = pd.read_csv(csv_file)
        df["source_file"] = csv_file.name  # keep provenance
        groups[key].append(df)

    # ---- Merge per group ----
    for key in groups:
        if groups[key]:
            groups[key] = pd.concat(groups[key], ignore_index=True)
        else:
            groups[key] = pd.DataFrame()

    return groups


def chains_by_group(metadata, *,
                    energy_col="beam_energy",
                    particle_col="beam_type",
                    file_col="input_path",
                    tree_name="cbmsim",
                    unique_files=True,
                    max_files = None):
    """
    Build a ROOT.TChain per (beam_energy, beam_type) group.

    Returns
    -------
    dict[(energy, particle), ROOT.TChain]
    """
    chains = {}

    grouped = metadata.groupby([energy_col, particle_col], dropna=False)
    geo_file = metadata['geo_path'][0]
    for (E, p), df in grouped:
        chain = ROOT.TChain(tree_name)

        files = df[file_col].dropna().astype(str)
        if unique_files:
            files = files.drop_duplicates()

        n_added = 0
        for f in files:
            # TChain.Add returns number of files added (0 if failed)
            ret = chain.Add(f)
            if ret:
                n_added += 1
            if max_files<=n_added:
                break
                

        chains[(E, p)] = chain
        print(f"[{tree_name}] group (E={E}, type={p}): added {n_added} files, entries={chain.GetEntries()}")

    return chains, geo_file
# -----------------------------------------------------------------------------
# Variable configuration: choose between "qdc" and "hitTime"
# -----------------------------------------------------------------------------
def value_spec(mode: str):
    """
    Returns:
      getter(hit) -> float
      var_label (str) : used in axis titles and histogram names
      ranges (dict)   : default binning/ranges for 1D and 2D
    """
    mode = mode.lower()

    if mode in ("qdc", "signal"):
        getter = lambda hit: float(hit.GetSignal(0))
        var_label = "QDC"
        ranges = dict(
            y_nbins_2d=200, y_min_2d=-50,  y_max_2d=150,
            x_nbins_1d=250, x_min_1d=-100, x_max_1d=150,
        )
        return getter, var_label, ranges

    if mode in ("hittime", "time"):
        getter = lambda hit: float(hit.GetTime())
        var_label = "hitTime"
        # NOTE: adjust to your detector's time units/range if needed
        ranges = dict(
            y_nbins_2d=150, y_min_2d=-5,    y_max_2d=10,
            x_nbins_1d=350, x_min_1d=-10,    x_max_1d=25,
        )
        return getter, var_label, ranges

    raise ValueError(f"Unknown mode='{mode}'. Use 'qdc' or 'hitTime'.")


# -----------------------------------------------------------------------------
# Histogram factories (lazy creation)
# -----------------------------------------------------------------------------
def get_hist2d_vs_chan(hmap, key, E, p, *, var_label,
                       nbins_chan=512, nbins_y=200, y_min=-50, y_max=150):
    """
    key = (station, orientation, mat)
    Creates/returns TH2F: var vs channel for this key.
    """
    if key not in hmap:
        st, ori, mat = key
        ori_i = int(ori)
        name  = f"h_{var_label}_vs_chan_st{st}_o{ori_i}_m{mat}_E{E}_{p}"
        title = f"{var_label} vs channel (st={st}, o={ori_i}, mat={mat}) [E={E}, p={p}];channel;{var_label}"
        h = ROOT.TH2F(name, title,
                      nbins_chan, -0.5, nbins_chan - 0.5,
                      nbins_y, y_min, y_max)
        h.Sumw2()
        hmap[key] = h
    return hmap[key]


def get_hist1d_for_channel(hmap, ch, E, p, *, var_label, prefix,
                           nbins=250, xmin=-100, xmax=150):
    """
    Creates/returns TH1F: var distribution for a specific channel.
    """
    if ch not in hmap:
        name  = f"h_{prefix}_ch{ch}_{var_label}_E{E}_{p}"
        title = f"{var_label} {prefix} ch={ch} [E={E}, p={p}];{var_label};Counts"
        h = ROOT.TH1F(name, title, nbins, xmin, xmax)
        h.Sumw2()
        hmap[ch] = h
    return hmap[ch]

def fill_raw_data(chain, E, p, *,
                     mode="qdc",
                     tree_branch="Digi_ScifiHits",
                     max_hits=None,
                     nbins_chan_2d=200,
                     st2_mat0_make_per_channel=True):
    getter, var_label, R = value_spec(mode)

    total = int(chain.GetEntries())
    n_hits = total if max_hits is None else min(total, int(max_hits))

    # 2D: var vs station
    h_vs_station = ROOT.TH2F(
        f"h_{var_label}_vs_station_E{E}_{p}",
        f"{var_label} vs station [E={E}, p={p}];station;{var_label}",
        6, -0.5, 5.5,
        R["y_nbins_2d"], R["y_min_2d"], R["y_max_2d"],
    )
    h_vs_station.Sumw2()

    # 2D: var vs channel per (st,ori,mat)
    h_vs_chan_by_key = {}  # (station, orientation, mat) -> TH2F

    # 1D: st=2, mat=0, split by orientation
    h_st2_o0_m0 = ROOT.TH1F(
        f"h_{var_label}_st2_o0_m0_E{E}_{p}",
        f"{var_label} st=2, ori=0, mat=0 [E={E}, p={p}];{var_label};Counts",
        R["x_nbins_1d"], R["x_min_1d"], R["x_max_1d"],
    )
    h_st2_o0_m0.Sumw2()

    h_st2_o1_m0 = ROOT.TH1F(
        f"h_{var_label}_st2_o1_m0_E{E}_{p}",
        f"{var_label} st=2, ori=1, mat=0 [E={E}, p={p}];{var_label};Counts",
        R["x_nbins_1d"], R["x_min_1d"], R["x_max_1d"],
    )
    h_st2_o1_m0.Sumw2()

    # Per-channel 1D (keep separate maps for ori=0 and ori=1 to avoid mixups)
    by_ch_o0 = {}
    by_ch_o1 = {}

    for i in tqdm(range(n_hits), desc=f"fill {mode} E={E} p={p}"):
        chain.GetEntry(i)
        print(dir(chain))
        #get goard id for station, orientation
        # 
        break
        
        

# -----------------------------------------------------------------------------
# Fill function (works for QDC or hitTime)
# -----------------------------------------------------------------------------
def fill_scifi_hists(chain, E, p, *,
                     mode="qdc",
                     tree_branch="Digi_ScifiHits",
                     max_events=None,
                     nbins_chan_2d=200,
                     st2_mat0_make_per_channel=True):
    getter, var_label, R = value_spec(mode)

    total = int(chain.GetEntries())
    n_events = total if max_events is None else min(total, int(max_events))

    # 2D: var vs station
    h_vs_station = ROOT.TH2F(
        f"h_{var_label}_vs_station_E{E}_{p}",
        f"{var_label} vs station [E={E}, p={p}];station;{var_label}",
        6, -0.5, 5.5,
        R["y_nbins_2d"], R["y_min_2d"], R["y_max_2d"],
    )
    h_vs_station.Sumw2()

    # 2D: var vs channel per (st,ori,mat)
    h_vs_chan_by_key = {}  # (station, orientation, mat) -> TH2F

    # 1D: st=2, mat=0, split by orientation
    h_st2_o0_m0 = ROOT.TH1F(
        f"h_{var_label}_st2_o0_m0_E{E}_{p}",
        f"{var_label} st=2, ori=0, mat=0 [E={E}, p={p}];{var_label};Counts",
        R["x_nbins_1d"], R["x_min_1d"], R["x_max_1d"],
    )
    h_st2_o0_m0.Sumw2()

    h_st2_o1_m0 = ROOT.TH1F(
        f"h_{var_label}_st2_o1_m0_E{E}_{p}",
        f"{var_label} st=2, ori=1, mat=0 [E={E}, p={p}];{var_label};Counts",
        R["x_nbins_1d"], R["x_min_1d"], R["x_max_1d"],
    )
    h_st2_o1_m0.Sumw2()

    # Per-channel 1D (keep separate maps for ori=0 and ori=1 to avoid mixups)
    by_ch_o0 = {}
    by_ch_o1 = {}

    for i in tqdm(range(n_events), desc=f"fill {mode} E={E} p={p}"):
        chain.GetEntry(i)
        hits = getattr(chain, tree_branch)

        for ih in range(hits.GetEntriesFast()):
            aHit  = hits.At(ih)
            detID = int(aHit.GetDetectorID())
            val   = float(getter(aHit))

            st  = detID // 1000000
            ori = int(aHit.isVertical())           # 0/1
            mat = (detID % 100000) // 10000
            ch  = detID % 1000

            h_vs_station.Fill(st, val)

            h2 = get_hist2d_vs_chan(
                h_vs_chan_by_key, (st, ori, mat), E, p,
                var_label=var_label,
                nbins_chan=nbins_chan_2d,
                nbins_y=R["y_nbins_2d"], y_min=R["y_min_2d"], y_max=R["y_max_2d"],
            )
            h2.Fill(ch, val)

            # st=2, mat=0 special selections
            if st == 2 and mat == 0:
                if ori == 0:
                    h_st2_o0_m0.Fill(val)
                    if st2_mat0_make_per_channel:
                        hch = get_hist1d_for_channel(
                            by_ch_o0, ch, E, p,
                            var_label=var_label,
                            prefix="st2_o0_m0",
                            nbins=R["x_nbins_1d"], xmin=R["x_min_1d"], xmax=R["x_max_1d"],
                        )
                        hch.Fill(val)
                else:
                    h_st2_o1_m0.Fill(val)
                    if st2_mat0_make_per_channel:
                        hch = get_hist1d_for_channel(
                            by_ch_o1, ch, E, p,
                            var_label=var_label,
                            prefix="st2_o1_m0",
                            nbins=R["x_nbins_1d"], xmin=R["x_min_1d"], xmax=R["x_max_1d"],
                        )
                        hch.Fill(val)

    return h_vs_station, h_vs_chan_by_key, h_st2_o0_m0, h_st2_o1_m0, by_ch_o0, by_ch_o1


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def _draw_2d_compare(h_dt, h_mc, title_dt="Data", title_mc="MC"):
    c = ROOT.TCanvas("c", "c", 1400, 600)
    c.Divide(2, 1)

    zmax = max(h_dt.GetMaximum(), h_mc.GetMaximum())
    if zmax > 0:
        h_dt.SetMaximum(zmax)
        h_mc.SetMaximum(zmax)

    c.cd(1)
    h_dt.SetTitle(f"{h_dt.GetTitle()} ({title_dt})")
    h_dt.Draw("COLZ")

    c.cd(2)
    h_mc.SetTitle(f"{h_mc.GetTitle()} ({title_mc})")
    h_mc.Draw("COLZ")

    c.Update()
    return c


def _draw_4_1d_compare(h_dt_o0, h_dt_o1, h_mc_o0, h_mc_o1, fout, *,
                      normalize=True, logy=False):
    ROOT.gStyle.SetOptStat(0)

    # Clone defensively
    h_dt_o0 = h_dt_o0.Clone(h_dt_o0.GetName() + "_c")
    h_dt_o1 = h_dt_o1.Clone(h_dt_o1.GetName() + "_c")
    h_mc_o0 = h_mc_o0.Clone(h_mc_o0.GetName() + "_c")
    h_mc_o1 = h_mc_o1.Clone(h_mc_o1.GetName() + "_c")

    if normalize:
        for h in (h_dt_o0, h_dt_o1, h_mc_o0, h_mc_o1):
            integral = h.Integral(0, h.GetNbinsX() + 1)
            if integral > 0:
                h.Scale(1.0 / integral)

    # Style: ori=0 blue, ori=1 red
    col_dt_o0 = ROOT.kBlue + 2
    col_mc_o0 = ROOT.kBlue - 4
    col_dt_o1 = ROOT.kRed + 2
    col_mc_o1 = ROOT.kRed - 4

    for h, col in [(h_dt_o0, col_dt_o0), (h_dt_o1, col_dt_o1)]:
        h.SetMarkerStyle(20)
        h.SetMarkerSize(0.9)
        h.SetMarkerColor(col)
        h.SetLineColor(col)
        h.SetLineWidth(2)

    for h, col in [(h_mc_o0, col_mc_o0), (h_mc_o1, col_mc_o1)]:
        h.SetLineColorAlpha(col, 0.65)
        h.SetLineWidth(3)
        h.SetLineStyle(2)
        h.SetFillStyle(0)
        h.SetMarkerSize(0)

    ymax = max(h.GetMaximum() for h in (h_dt_o0, h_dt_o1, h_mc_o0, h_mc_o1))
    if ymax > 0:
        h_dt_o0.SetMaximum(ymax * (20 if logy else 1.35))

    c = ROOT.TCanvas("c_1d4", "c_1d4", 900, 700)
    if logy:
        c.SetLogy()

    h_mc_o0.Draw("HIST")
    h_mc_o1.Draw("HIST SAME")
    h_dt_o0.Draw("E1 SAME")
    h_dt_o1.Draw("E1 SAME")

    leg = ROOT.TLegend(0.55, 0.68, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_dt_o0, "Data ori=0", "lep")
    leg.AddEntry(h_mc_o0, "MC ori=0", "l")
    leg.AddEntry(h_dt_o1, "Data ori=1", "lep")
    leg.AddEntry(h_mc_o1, "MC ori=1", "l")
    leg.Draw()

    c.Update()
    c.SaveAs(fout)
    print(f"[saved] {fout}")
    c.Close()


def save_by_channel_pdf(h_dt_by_ch, h_mc_by_ch, fout, *, normalize=True, logy=False):
    ROOT.gStyle.SetOptStat(0)

    chans = sorted(set(h_dt_by_ch.keys()) | set(h_mc_by_ch.keys()))
    if not chans:
        print(f"[skip] no channel histograms to save for {fout}")
        return

    c = ROOT.TCanvas("c_by_ch", "c_by_ch", 900, 700)
    if logy:
        c.SetLogy()

    c.Print(fout + "[")

    for ch in chans:
        c.Clear()
        c.SetLogy(1 if logy else 0)

        hdt = h_dt_by_ch.get(ch)
        hmc = h_mc_by_ch.get(ch)

        if hdt:
            hdt = hdt.Clone(hdt.GetName() + f"_page{ch}_dt")
        if hmc:
            hmc = hmc.Clone(hmc.GetName() + f"_page{ch}_mc")

        if normalize:
            if hdt:
                i = hdt.Integral(0, hdt.GetNbinsX() + 1)
                if i > 0:
                    hdt.Scale(1.0 / i)
            if hmc:
                i = hmc.Integral(0, hmc.GetNbinsX() + 1)
                if i > 0:
                    hmc.Scale(1.0 / i)

        if hdt:
            hdt.SetMarkerStyle(20)
            hdt.SetMarkerSize(0.9)
            hdt.SetLineWidth(2)
        if hmc:
            hmc.SetLineWidth(3)
            hmc.SetLineStyle(2)

        ymax = 0.0
        if hdt: ymax = max(ymax, hdt.GetMaximum())
        if hmc: ymax = max(ymax, hmc.GetMaximum())
        if ymax > 0:
            if hdt:
                hdt.SetMaximum(ymax * (20 if logy else 1.35))
            else:
                hmc.SetMaximum(ymax * (20 if logy else 1.35))

        if hmc and not hdt:
            hmc.Draw("HIST")
        elif hdt and not hmc:
            hdt.Draw("E1")
        else:
            hmc.Draw("HIST")
            hdt.Draw("E1 SAME")

        leg = ROOT.TLegend(0.60, 0.74, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        if hdt: leg.AddEntry(hdt, f"Data ch={ch}", "lep")
        if hmc: leg.AddEntry(hmc, f"MC   ch={ch}", "l")
        leg.Draw()

        c.Update()
        c.Print(fout)

    c.Print(fout + "]")
    c.Close()
    print(f"[saved] {fout}")


# -----------------------------------------------------------------------------
# Main driver (same interface, plus mode)
# -----------------------------------------------------------------------------
def plot_comparisons_by_group(
    mc_chains, data_chains, mc_geo_file, data_geo_file, *,
    out_dir="plots",
    year=None,
    tree_branch="Digi_ScifiHits",
    max_events_MC=None,
    max_events_Data=None,
    logz=False,
    normalize=False,
    logy=False,
    mode="qdc",
):
    """
    Produces (per E,p):
      - var vs station 2D compare PDF
      - var vs channel 2D compare PDFs for each (st,ori,mat)
      - 4x 1D compare overlay for st=2 mat=0 ori=0/1 (Data+MC)
      - per-channel multipage PDFs (st=2 mat=0) for ori=0 and ori=1 separately
    """
    _, var_label, _ = value_spec(mode)

    out_dir = f"{out_dir}/{year}" if year is not None else out_dir
    os.makedirs(out_dir, exist_ok=True)

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    all_keys = sorted(set(mc_chains.keys()) | set(data_chains.keys()))

    for key in all_keys:
        E, p = key
        mc_chain = mc_chains.get(key)
        dt_chain = data_chains.get(key)

        if mc_chain is None or dt_chain is None:
            print(f"[skip] missing side for group (E={E}, type={p})  "
                  f"MC={'ok' if mc_chain else 'missing'} Data={'ok' if dt_chain else 'missing'}")
            continue

        # Fill
        h_station_dt, hmap_dt, h_dt_o0, h_dt_o1, h_dt_by_ch_o0, h_dt_by_ch_o1 = fill_raw_data(
            dt_chain, E, p, mode=mode, tree_branch=tree_branch, max_events=max_events_Data
        )
        h_station_mc, hmap_mc, h_mc_o0, h_mc_o1, h_mc_by_ch_o0, h_mc_by_ch_o1 = fill_scifi_hists(
            mc_chain, E, p, mode=mode, tree_branch=tree_branch, max_events=max_events_MC
        )

        # Per-channel PDFs (ori separated)
        fout_ch_o0 = f"{out_dir}/{var_label}_by_channel_E{E}_{p}_st2_mat0_ori0.pdf"
        save_by_channel_pdf(h_dt_by_ch_o0, h_mc_by_ch_o0, fout_ch_o0,
                            normalize=normalize, logy=logy)

        fout_ch_o1 = f"{out_dir}/{var_label}_by_channel_E{E}_{p}_st2_mat0_ori1.pdf"
        save_by_channel_pdf(h_dt_by_ch_o1, h_mc_by_ch_o1, fout_ch_o1,
                            normalize=normalize, logy=logy)

        # Station compare
        h_station_dt = h_station_dt.Clone(f"h_{var_label}_vs_station_dt_E{E}_{p}")
        h_station_mc = h_station_mc.Clone(f"h_{var_label}_vs_station_mc_E{E}_{p}")

        c = _draw_2d_compare(h_station_dt, h_station_mc, "Data", "MC")
        if logz:
            c.cd(1).SetLogz()
            c.cd(2).SetLogz()
            c.Update()

        fout = f"{out_dir}/{var_label}_vs_station_E{E}_{p}.pdf"
        c.SaveAs(fout)
        print(f"[saved] {fout}")
        c.Close()

        # Channel comparisons per (st,ori,mat)
        keys_2d = sorted(set(hmap_dt.keys()) | set(hmap_mc.keys()))
        for (st, ori, mat) in keys_2d:
            if (st, ori, mat) in hmap_dt:
                hdt = hmap_dt[(st, ori, mat)].Clone(f"h_dt_{var_label}_E{E}_{p}_st{st}_o{int(ori)}_m{mat}")
            else:
                htmp = hmap_mc[(st, ori, mat)]
                hdt = ROOT.TH2F(
                    f"h_dt_{var_label}_E{E}_{p}_st{st}_o{int(ori)}_m{mat}",
                    htmp.GetTitle(),
                    htmp.GetNbinsX(), htmp.GetXaxis().GetXmin(), htmp.GetXaxis().GetXmax(),
                    htmp.GetNbinsY(), htmp.GetYaxis().GetXmin(), htmp.GetYaxis().GetXmax(),
                )
                hdt.Sumw2()

            if (st, ori, mat) in hmap_mc:
                hmc = hmap_mc[(st, ori, mat)].Clone(f"h_mc_{var_label}_E{E}_{p}_st{st}_o{int(ori)}_m{mat}")
            else:
                htmp = hmap_dt[(st, ori, mat)]
                hmc = ROOT.TH2F(
                    f"h_mc_{var_label}_E{E}_{p}_st{st}_o{int(ori)}_m{mat}",
                    htmp.GetTitle(),
                    htmp.GetNbinsX(), htmp.GetXaxis().GetXmin(), htmp.GetXaxis().GetXmax(),
                    htmp.GetNbinsY(), htmp.GetYaxis().GetXmin(), htmp.GetYaxis().GetXmax(),
                )
                hmc.Sumw2()

            c2 = _draw_2d_compare(hdt, hmc, "Data", "MC")
            if logz:
                c2.cd(1).SetLogz()
                c2.cd(2).SetLogz()
                c2.Update()

            fout2 = f"{out_dir}/{var_label}_vs_chan_E{E}_{p}_station{st}_orientation{int(ori)}_mat{mat}.pdf"
            c2.SaveAs(fout2)
            print(f"[saved] {fout2}")
            c2.Close()

        # 4x overlay 1D compare (st=2 mat=0)
        h_dt_o0 = h_dt_o0.Clone(f"h_dt_{var_label}_st2_o0_m0_E{E}_{p}")
        h_dt_o1 = h_dt_o1.Clone(f"h_dt_{var_label}_st2_o1_m0_E{E}_{p}")
        h_mc_o0 = h_mc_o0.Clone(f"h_mc_{var_label}_st2_o0_m0_E{E}_{p}")
        h_mc_o1 = h_mc_o1.Clone(f"h_mc_{var_label}_st2_o1_m0_E{E}_{p}")

        fout4 = f"{out_dir}/{var_label}_1d4_E{E}_{p}_st2_mat0.pdf"
        _draw_4_1d_compare(h_dt_o0, h_dt_o1, h_mc_o0, h_mc_o1, fout4,
                           normalize=normalize, logy=logy)

        break

def main():
    metadata = load_all_metadata("../metadata/updated")

    mc24   = metadata["MC_2024"]
    data24 = metadata["DATA_2024"]

    mc24_chains, mc24_geo_file   = chains_by_group(mc24,   file_col="digi_path", tree_name="cbmsim", max_files=10)
    data24_chains, data24_geo_file= chains_by_group(data24, file_col="raw_path", tree_name="data", max_files=10)
    
    
    #print(mc24_geo_file, data24_geo_file)
    # plot_qdc_comparisons_by_group(
    #     mc24_chains,
    #     data24_chains,
    #     mc24_geo_file,
    #     data24_geo_file,
    #     out_dir="qdc_plot",
    #     year = "2024",
    #     tree_branch="Digi_ScifiHits",
    #     nbins=200, xmin=-50, xmax=150,
    #     max_events=20,     
    #     normalize=True,
    #     logy=True
    # )
    plot_comparisons_by_group(
        mc24_chains,
        data24_chains,
        mc24_geo_file,
        data24_geo_file,
        out_dir="TestbeamPlot",
        year = "2024",
        tree_branch="Digi_ScifiHits",
        max_events_MC=50,    
        max_events_Data=200,    
        logz=True,
        normalize=True,
        logy=True,      
        mode="hitTime"  
    )
    
    
if __name__ == "__main__":
    main()
