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


def qdc_hist_from_chain(chain, hist_name, *,
                        tree_branch="Digi_ScifiHits",
                        nbins=150, xmin=0, xmax=150,
                        max_events=None):
    h = ROOT.TH1F(hist_name, hist_name, nbins, xmin, xmax)
    h.Sumw2()

    total = int(chain.GetEntries())
    if max_events is not None:
        n_events = min(total, int(max_events))


    random_indices = random.sample(range(total), n_events)
    for i in random_indices:
    #for i in range(n_events):
        chain.GetEntry(i)
        hits = getattr(chain, tree_branch)  # TClonesArray

        n_hits = hits.GetEntriesFast()
        for ih in range(n_hits):
            aHit = hits.At(ih)
            qdc = aHit.GetSignal(0)
            h.Fill(qdc)

    return h


def get_hist(hmap, key, E, p, nbins_chan=512, nbins_qdc=200, qdc_min=-50, qdc_max=150):
    """
    key = (station, orientation, mat)
      - station: int
      - orientation: int/bool (0/1)
      - mat: int
    """
    if key not in hmap:
        st, ori, mat = key
        ori_i = int(ori)  # bool -> 0/1
        name  = f"h_qdc_vs_chan_st{st}_o{ori_i}_m{mat}"
        title = f"QDC vs channel (st={st}, o={ori_i}, mat={mat}) [E={E}, p={p}];channel;QDC"
        h = ROOT.TH2F(name, title,
                      nbins_chan, -0.5, nbins_chan - 0.5,
                      nbins_qdc, qdc_min, qdc_max)
        h.Sumw2()
        hmap[key] = h
    return hmap[key]


def get_qdc_hist_for_channel(hmap, ch, E, p, *, nbins=250, xmin=-100, xmax=150):
    if ch not in hmap:
        name  = f"h_qdc_st2_o0_m0_ch{ch}_E{E}_{p}"
        title = f"QDC st=2 mat=0 ori=0 ch={ch} [E={E}, p={p}];QDC;Counts"
        h = ROOT.TH1F(name, title, nbins, xmin, xmax)
        h.Sumw2()
        hmap[ch] = h
    return hmap[ch]

def fill_scifi_hists(chain, E, p, tree_branch="Digi_ScifiHits", max_events=None):
    total = int(chain.GetEntries())
    n_events = total if max_events is None else min(total, int(max_events))

    # --- existing outputs ---
    h_qdc_vs_station = ROOT.TH2F("h_qdc_vs_station", f"QDC vs station [E={E}, p={p}];station;QDC",
                                6, -0.5, 5.5,
                                200, -50, 150)
    h_qdc_vs_station.Sumw2()

    h_qdc_vs_chan_by_key = {}  # (station, orientation, mat) -> TH2F

    # --- NEW: two 1D QDC hists for st=2, mat=0, ori=0/1 ---
    h_qdc_st2_o0_m0 = ROOT.TH1F("h_qdc_st2_o0_m0", f"QDC st=2, ori=0, mat=0 [E={E}, p={p}];QDC;Counts",
                               250, -100, 150)
    h_qdc_st2_o0_m0.Sumw2()

    h_qdc_st2_o1_m0 = ROOT.TH1F("h_qdc_st2_o1_m0", f"QDC st=2, ori=1, mat=0 [E={E}, p={p}];QDC;Counts",
                               250, -100, 150)
    h_qdc_st2_o1_m0.Sumw2()
    
    h_qdc_st2_o0_m0_by_ch = {}

    for i in tqdm(range(n_events)):
        chain.GetEntry(i)
        hits = getattr(chain, tree_branch)

        for ih in range(hits.GetEntriesFast()):
            aHit  = hits.At(ih)
            detID = int(aHit.GetDetectorID())
            qdc   = float(aHit.GetSignal(0))

            st  = detID // 1000000
            ori = int(aHit.isVertical())        # 0/1
            mat = (detID % 100000) // 10000
            ch  = detID % 1000

            # existing fills
            h_qdc_vs_station.Fill(st, qdc)

            h2 = get_hist(h_qdc_vs_chan_by_key, (st, ori, mat), E, p,
                          nbins_chan=200, nbins_qdc=200, qdc_min=-50, qdc_max=150)
            h2.Fill(ch, qdc)

            # NEW fills (the two selections)
            
            if st == 2 and mat == 0:
                if ori == 0:
                    h_qdc_st2_o0_m0.Fill(qdc)
                    # hch = get_qdc_hist_for_channel(h_qdc_st2_o0_m0_by_ch, ch, E, p,
                    #                                nbins=250, xmin=-100, xmax=150)
                    # hch.Fill(qdc)
                elif ori == 1:
                    h_qdc_st2_o1_m0.Fill(qdc)
                    hch = get_qdc_hist_for_channel(h_qdc_st2_o0_m0_by_ch, ch, E, p,
                                                   nbins=250, xmin=-100, xmax=150)
                    hch.Fill(qdc)

    return h_qdc_vs_station, h_qdc_vs_chan_by_key, h_qdc_st2_o0_m0, h_qdc_st2_o1_m0, h_qdc_st2_o0_m0_by_ch



def _draw_2d_compare(h_dt, h_mc, title_dt="Data", title_mc="MC"):
    """
    Draw side-by-side comparison of two TH2s.
    Returns a ROOT.TCanvas (caller owns it).
    """
    c = ROOT.TCanvas("c", "c", 1400, 600)
    c.Divide(2, 1)

    # Make sure both use comparable z-range
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

def _draw_4_qdc_1d(h_dt_o0, h_dt_o1, h_mc_o0, h_mc_o1, fout4, *, normalize=True, logy=False):
    ROOT.gStyle.SetOptStat(0)

    # Clone defensively
    h_dt_o0 = h_dt_o0.Clone(h_dt_o0.GetName() + "_c")
    h_dt_o1 = h_dt_o1.Clone(h_dt_o1.GetName() + "_c")
    h_mc_o0 = h_mc_o0.Clone(h_mc_o0.GetName() + "_c")
    h_mc_o1 = h_mc_o1.Clone(h_mc_o1.GetName() + "_c")

    # Normalize (shape comparison)
    if normalize:
        for h in (h_dt_o0, h_dt_o1, h_mc_o0, h_mc_o1):
            integral = h.Integral(0, h.GetNbinsX() + 1)
            if integral > 0:
                h.Scale(1.0 / integral)

    # --------------------------------------
    # Color families
    # --------------------------------------

    # Orientation 0 → Blue family
    col_dt_o0 = ROOT.kBlue + 2
    col_mc_o0 = ROOT.kBlue - 4

    # Orientation 1 → Red family
    col_dt_o1 = ROOT.kRed + 2
    col_mc_o1 = ROOT.kRed - 4

    # --- Data (markers + error bars, solid)
    for h, col in [(h_dt_o0, col_dt_o0), (h_dt_o1, col_dt_o1)]:
        h.SetMarkerStyle(20)
        h.SetMarkerSize(0.9)
        h.SetMarkerColor(col)
        h.SetLineColor(col)
        h.SetLineWidth(2)

    # --- MC (dashed + semi-transparent lines)
    for h, col in [(h_mc_o0, col_mc_o0), (h_mc_o1, col_mc_o1)]:
        h.SetLineColorAlpha(col, 0.65)   # transparency (0→1)
        h.SetLineWidth(3)
        h.SetLineStyle(2)                # dashed
        h.SetFillStyle(0)
        h.SetMarkerSize(0)

    # Y range
    ymax = max(h.GetMaximum() for h in (h_dt_o0, h_dt_o1, h_mc_o0, h_mc_o1))
    if ymax > 0:
        h_dt_o0.SetMaximum(ymax * (20 if logy else 1.35))

    c = ROOT.TCanvas("c_qdc4", "c_qdc4", 900, 700)
    if logy:
        c.SetLogy()

    # Draw MC first
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
    c.SaveAs(fout4)
    print(f"[saved] {fout4}")
    c.Close()

def save_qdc_by_channel_pdf(h_dt_by_ch, h_mc_by_ch, fout, *, normalize=True, logy=False):
    ROOT.gStyle.SetOptStat(0)

    # union of channels present in either
    chans = sorted(set(h_dt_by_ch.keys()) | (set(h_mc_by_ch.keys()) if h_mc_by_ch else set()))
    if len(chans) == 0:
        print(f"[skip] no channel histograms to save for {fout}")
        return

    c = ROOT.TCanvas("c_qdc_by_ch", "c_qdc_by_ch", 900, 700)
    if logy:
        c.SetLogy()

    # open multipage PDF
    c.Print(fout + "[")

    for ch in chans:
        c.Clear()
        if logy:
            c.SetLogy()
        else:
            c.SetLogy(0)

        hdt = h_dt_by_ch.get(ch, None)
        hmc = h_mc_by_ch.get(ch, None) if h_mc_by_ch else None

        # clone so we can normalize/style safely
        if hdt:
            hdt = hdt.Clone(hdt.GetName() + f"_c_page{ch}")
        if hmc:
            hmc = hmc.Clone(hmc.GetName() + f"_c_page{ch}")

        # optional normalization (shape comparison)
        if normalize:
            if hdt:
                i = hdt.Integral(0, hdt.GetNbinsX() + 1)
                if i > 0: hdt.Scale(1.0 / i)
            if hmc:
                i = hmc.Integral(0, hmc.GetNbinsX() + 1)
                if i > 0: hmc.Scale(1.0 / i)

        # style: Data = marker+errors, MC = line
        # (keep it simple; you can color-code by orientation family later if you want)
        if hdt:
            hdt.SetMarkerStyle(20)
            hdt.SetMarkerSize(0.9)
            hdt.SetLineWidth(2)

        if hmc:
            hmc.SetLineWidth(3)
            hmc.SetLineStyle(2)  # dashed helps overlap

        # y-range
        ymax = 0.0
        if hdt: ymax = max(ymax, hdt.GetMaximum())
        if hmc: ymax = max(ymax, hmc.GetMaximum())
        if ymax > 0 and hdt:
            hdt.SetMaximum(ymax * (20 if logy else 1.35))
        elif ymax > 0 and hmc:
            hmc.SetMaximum(ymax * (20 if logy else 1.35))

        # draw
        if hmc and not hdt:
            hmc.Draw("HIST")
        elif hdt and not hmc:
            hdt.Draw("E1")
        else:
            # draw MC first, then Data on top
            hmc.Draw("HIST")
            hdt.Draw("E1 SAME")

        # legend + channel label
        leg = ROOT.TLegend(0.60, 0.74, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        if hdt: leg.AddEntry(hdt, f"Data ch={ch}", "lep")
        if hmc: leg.AddEntry(hmc, f"MC   ch={ch}", "l")
        leg.Draw()

        c.Update()
        c.Print(fout)  # add one page

    # close multipage PDF
    c.Print(fout + "]")
    c.Close()

    print(f"[saved] {fout}")

def plot_qdc_comparisons_by_group(
    mc_chains, data_chains, mc_geo_file, data_geo_file, *,
    out_dir="qdc_plots",
    year=None,
    tree_branch="Digi_ScifiHits",
    max_events_MC=None,
    max_events_Data=None,
    logz=False,
    normalize=False,
    logy=False,   
):
    """
    Saves:
      - QDC vs station comparison (Data vs MC) per (E,p)
      - QDC vs channel comparison for each (station, ori, mat) present in either side
    Filenames:
      - {out_dir}/{year}/qdc_vs_station_E{E}_{p}.pdf
      - {out_dir}/{year}/qdc_vs_chan_E{E}_{p}_st{st}_o{ori}_m{mat}.pdf
    """
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

        # Build hists (note: your current fill_scifi_hists does not use geo files;
        # keep args for future use if you later decode via geometry)
        h_station_dt, hmap_dt, h_dt_o0, h_dt_o1, h_dt_by_ch = fill_scifi_hists(dt_chain, E, p, tree_branch=tree_branch, max_events=max_events_Data)
        h_station_mc, hmap_mc, h_mc_o0, h_mc_o1, h_mc_by_ch = fill_scifi_hists(mc_chain, E, p, tree_branch=tree_branch, max_events=max_events_MC)

        
        fout_ch = f"{out_dir}/qdc_by_channel_E{E}_{p}_st2_mat0_ori0.pdf"
        save_qdc_by_channel_pdf(h_dt_by_ch, h_mc_by_ch, fout_ch,
                                normalize=normalize, logy=logy)
        
        # ---- Save station comparison ----
        # Clone with unique names to avoid ROOT name collisions across loops
        h_station_dt = h_station_dt.Clone(f"h_qdc_vs_station_dt_E{E}_{p}")
        h_station_mc = h_station_mc.Clone(f"h_qdc_vs_station_mc_E{E}_{p}")

        c = _draw_2d_compare(h_station_dt, h_station_mc, "Data", "MC")
        if logz:
            c.cd(1).SetLogz()
            c.cd(2).SetLogz()
            c.Update()

        fout = f"{out_dir}/qdc_vs_station_E{E}_{p}.pdf"
        c.SaveAs(fout)
        print(f"[saved] {fout}")
        c.Close()

        # ---- Save channel comparisons for each (st,ori,mat) ----
        keys_2d = sorted(set(hmap_dt.keys()) | set(hmap_mc.keys()))
        for (st, ori, mat) in keys_2d:
            # Ensure something exists on both sides; if missing, draw empty clone for consistent layout
            if (st, ori, mat) in hmap_dt:
                hdt = hmap_dt[(st, ori, mat)].Clone(f"h_dt_E{E}_{p}_st{st}_o{int(ori)}_m{mat}")
            else:
                # make an empty hist with same binning as MC
                htmp = hmap_mc[(st, ori, mat)]
                hdt = ROOT.TH2F(f"h_dt_E{E}_{p}_st{st}_o{int(ori)}_m{mat}",
                                htmp.GetTitle(), htmp.GetNbinsX(),
                                htmp.GetXaxis().GetXmin(), htmp.GetXaxis().GetXmax(),
                                htmp.GetNbinsY(),
                                htmp.GetYaxis().GetXmin(), htmp.GetYaxis().GetXmax())
                hdt.Sumw2()

            if (st, ori, mat) in hmap_mc:
                hmc = hmap_mc[(st, ori, mat)].Clone(f"h_mc_E{E}_{p}_st{st}_o{int(ori)}_m{mat}")
            else:
                htmp = hmap_dt[(st, ori, mat)]
                hmc = ROOT.TH2F(f"h_mc_E{E}_{p}_st{st}_o{int(ori)}_m{mat}",
                                htmp.GetTitle(), htmp.GetNbinsX(),
                                htmp.GetXaxis().GetXmin(), htmp.GetXaxis().GetXmax(),
                                htmp.GetNbinsY(),
                                htmp.GetYaxis().GetXmin(), htmp.GetYaxis().GetXmax())
                hmc.Sumw2()

            c2 = _draw_2d_compare(hdt, hmc, "Data", "MC")
            if logz:
                c2.cd(1).SetLogz()
                c2.cd(2).SetLogz()
                c2.Update()

            fout2 = f"{out_dir}/qdc_vs_chan_E{E}_{p}_station{st}_orientation{int(ori)}_mat{mat}.pdf"
            c2.SaveAs(fout2)
            print(f"[saved] {fout2}")
            c2.Close()
        
        # ---- Save 4x 1D QDC comparison on one canvas (st=2, mat=0, ori=0/1) ----
        h_dt_o0 = h_dt_o0.Clone(f"h_dt_qdc_st2_o0_m0_E{E}_{p}")
        h_dt_o1 = h_dt_o1.Clone(f"h_dt_qdc_st2_o1_m0_E{E}_{p}")
        h_mc_o0 = h_mc_o0.Clone(f"h_mc_qdc_st2_o0_m0_E{E}_{p}")
        h_mc_o1 = h_mc_o1.Clone(f"h_mc_qdc_st2_o1_m0_E{E}_{p}")

        fout4 = f"{out_dir}/qdc_1d4_E{E}_{p}_st2_mat0.pdf"
        c4 = _draw_4_qdc_1d(h_dt_o0, h_dt_o1, h_mc_o0, h_mc_o1, fout4,
                        normalize=normalize, logy=logy)


        

            
        break
        # 
            
        #save all the plots similar with these format : f"[saved] {out_dir}/qdc_E{E}_{p}.pdf"

        # ----- plot qdc 1d hits ---------
        # h_mc = qdc_hist_from_chain(mc_chain, f"h_mc_E{E}_{p}", tree_branch=tree_branch,
        #                            nbins=nbins, xmin=xmin, xmax=xmax, max_events=max_events)
        # h_dt = qdc_hist_from_chain(dt_chain, f"h_dt_E{E}_{p}", tree_branch=tree_branch,
        #                            nbins=nbins, xmin=xmin, xmax=xmax, max_events=max_events)

        # if normalize:
        #     if h_mc.Integral() > 0:
        #         h_mc.Scale(1.0 / h_mc.Integral())
        #     if h_dt.Integral() > 0:
        #         h_dt.Scale(1.0 / h_dt.Integral())

        # # --- Style MC (line) ---
        # h_mc.SetLineColor(ROOT.kRed+1)
        # h_mc.SetLineWidth(3)
        # h_mc.SetFillStyle(0)

        # # --- Convert Data hist → TGraphErrors ---
        # g_dt = ROOT.TGraphErrors(h_dt.GetNbinsX())

        # for i in range(1, h_dt.GetNbinsX()+1):
        #     x  = h_dt.GetBinCenter(i)
        #     y  = h_dt.GetBinContent(i)
        #     ey = h_dt.GetBinError(i)   # Poisson √N scaled if normalized
        #     ex = h_dt.GetBinWidth(i) / 2

        #     g_dt.SetPoint(i-1, x, y)
        #     g_dt.SetPointError(i-1, ex, ey)

        # g_dt.SetMarkerStyle(20)
        # g_dt.SetMarkerSize(1.0)
        # g_dt.SetMarkerColor(ROOT.kBlue+2)
        # g_dt.SetLineColor(ROOT.kBlue+2)

        # title = f"QDC comparison  (E={E}, beam={p})"
        # h_dt.SetTitle(title)
        # h_dt.GetXaxis().SetTitle("QDC")
        # h_dt.GetYaxis().SetTitle("Normalized entries" if normalize else "Entries")

        # c = ROOT.TCanvas(f"c_E{E}_{p}", "", 900, 700)
        # if logy:
        #     c.SetLogy()

        # # Draw MC first to define axes
        # h_mc.SetTitle(f"QDC comparison  (Y={year}, E={E}, beam={p})")
        # h_mc.GetXaxis().SetTitle("QDC")
        # h_mc.GetYaxis().SetTitle("Normalized entries")

        # h_mc.Draw("HIST")
        # g_dt.Draw("P SAME")   # points with error bars

        # leg = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
        # leg.SetBorderSize(0)
        # leg.AddEntry(h_mc, f"MC ({max_events} evt)", "l")
        # leg.AddEntry(g_dt, f"Data ({max_events} evt)", "pe")
        # leg.Draw()

        # c.SaveAs(f"{out_dir}/qdc_E{E}_{p}.pdf")

        # print(f"[saved] {out_dir}/qdc_E{E}_{p}.pdf")


def main():
    metadata = load_all_metadata("../metadata/updated")

    mc24   = metadata["MC_2024"]
    data24 = metadata["DATA_2024"]
    mc23   = metadata["MC_2023"]
    data23 = metadata["DATA_2023"]


    # print(mc23)
    # mc23_chains   = chains_by_group(mc23,   file_col="digi_path", tree_name="cbmsim", max_files=10)
    # data23_chains = chains_by_group(data23, file_col="digi_path", tree_name="cbmsim", max_files=10)
    
    

    
    # plot_qdc_comparisons_by_group(
    #     mc23_chains,
    #     data23_chains,
    #     out_dir="qdc_plot",
    #     year = "2023",
    #     tree_branch="Digi_ScifiHits",
    #     nbins=200, xmin=-50, xmax=150,
    #     max_events=200,     
    #     normalize=True,
    #     logy=True
    # )
    
    mc24_chains, mc24_geo_file   = chains_by_group(mc24,   file_col="digi_path", tree_name="cbmsim", max_files=10)
    data24_chains, data24_geo_file= chains_by_group(data24, file_col="digi_path", tree_name="cbmsim", max_files=1)
    
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
    
    plot_qdc_comparisons_by_group(
        mc24_chains,
        data24_chains,
        mc24_geo_file,
        data24_geo_file,
        out_dir="qdc_plot",
        year = "2024",
        tree_branch="Digi_ScifiHits",
        max_events_MC=2000,    
        max_events_Data=20000,    
        logz=True,
        normalize=True,
        logy=True,        
    )
    
    
if __name__ == "__main__":
    main()
