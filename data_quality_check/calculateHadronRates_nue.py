import ROOT
import os
import numpy as np
from array import array
from tqdm import tqdm
from pathlib import Path

#from hadronPositionReweighting import xy_maps

# Switches
plot_cut_vars = False
do_stats = False
make_background_tree = False
# /Switches

BASE_FILTERED_DIR = Path("/eos/user/c/cvilela/SND_nue_analysis_May24/")

signal_color = "#22833" # Paul Tol "bright green"
background_color = "#EE6677" # Paul Tol "bright red"

signal_fill_style = 3345
background_fill_style = 1001

MC_bin_factor = 1

f_out = None

samples = ["Kaons", "neutrons"]#, "Kaons_FTFP_BERT", "neutrons_FTFP_BERT"]
samplekey = {}
samplekey["Kaons"] = "K"
samplekey["neutrons"] = "neu"

bins = [[0, 10],
        [10, 20],
        [20, 30],
        [30, 40],
        [40, 50],
        [50, 60],
        [60, 70],
        [70, 80],
        [80, 90],
        [90, 100]]
#        [100, 150],
#        [150, 200]]

generated = {}
passed_cuts = {}

interaction_rates = {}
for i_sample in samples :
    interaction_rates[i_sample] = []


target_lumi = 68.551

nominal_target_mass = 830.0
actual_target_mass = 792

use_daniele_flux = True
include_KS = True
#tracktype = "maxentrack"
#tracktype = "randomtrack"

if not use_daniele_flux :
    #Thomas's flux
    calculated_lumi = 150.
    muon_flux_factor = 253./573 # In SciFi. From Simona's talk at 12th collab meet.
    denominator_bin = 2 # veto cut is applied in Thomas's histograms
    f = ROOT.TFile("EofNeutralsInSND_vetoApplied.root")
    for p in f.Get("muDIS_SNDwithVeto").GetListOfPrimitives() :
        for p_ in p.GetListOfPrimitives() :
            if p_.GetName() == "Esnd_130_fb150" :
                h_130 = p_
            if p_.GetName() == "Esnd_310_fb150" :
                h_310 = p_
            if p_.GetName() == "Esnd_2112_fb150" :
                h_2112 = p_
            if p_.GetName() == "Esnd_-2112_fb150" :
                h_m2112 = p_
    h_2112.Add(h_m2112)
    if include_KS :
        h_130.Add(h_310)
    
    for bin_low, bin_high in bins :
        low_bin = h_130.GetXaxis().FindBin(bin_low)
        up_bin = h_130.GetXaxis().FindBin(bin_high)
        interaction_rates["Kaons"].append(h_130.Integral(low_bin, up_bin)*target_lumi/calculated_lumi*muon_flux_factor)
#        interaction_rates["Kaons_FTFP_BERT"].append(h_130.Integral(low_bin, up_bin)*target_lumi/calculated_lumi*muon_flux_factor)
    
        low_bin = h_2112.GetXaxis().FindBin(bin_low)
        up_bin = h_2112.GetXaxis().FindBin(bin_high)
        interaction_rates["neutrons"].append(h_2112.Integral(low_bin, up_bin)*target_lumi/calculated_lumi*muon_flux_factor)
#        interaction_rates["neutrons_FTFP_BERT"].append(h_2112.Integral(low_bin, up_bin)*target_lumi/calculated_lumi*muon_flux_factor)


else :
    # Use Daniele's flux
    calculated_lumi = 1e-5*4

    #   f = ROOT.TFile("/eos/experiment/sndlhc/users/dancc/MuonDIS/NeutralsExpect/7952615.MC-DISneutrals.muonDis.root")
    #   denominator_bin = 1 # Veto cut not applied in Daniele's histograms?
#    f = ROOT.TFile("/eos/experiment/sndlhc/users/dancc/MuonDIS/NeutralsExpect/060423/7966998.MC-DISneutrals_.root")
    f = ROOT.TFile("/eos/experiment/sndlhc/users/dancc/MuonDIS/NeutralsExpect/120423/7977358.MC-DISneutrals_.root")
    denominator_bin = 2
    for bin_low, bin_high in bins :
        
        if bin_low == 0. :
            bin_low = 5.
        
        # Kaons
        low_bin = f.Energy_neutrals_noVeto_maxentrack_K_L0.GetXaxis().FindBin(bin_low)
        up_bin = f.Energy_neutrals_noVeto_maxentrack_K_L0.GetXaxis().FindBin(bin_high)
        #print(bin_low, bin_high, low_bin, up_bin)
        interaction_rates["Kaons"].append(f.Energy_neutrals_noVeto_maxentrack_K_L0.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)
#        interaction_rates["Kaons_FTFP_BERT"].append(f.Energy_neutrals_noVeto_maxentrack_K_L0.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)

        if include_KS :
            low_bin = f.Energy_neutrals_noVeto_maxentrack_K_S0.GetXaxis().FindBin(bin_low)
            up_bin = f.Energy_neutrals_noVeto_maxentrack_K_S0.GetXaxis().FindBin(bin_high)
            interaction_rates["Kaons"][-1] += (f.Energy_neutrals_noVeto_maxentrack_K_S0.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)
#            interaction_rates["Kaons_FTFP_BERT"][-1] += (f.Energy_neutrals_noVeto_maxentrack_K_S0.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)
            
        # neutrons
        low_bin = f.Energy_neutrals_noVeto_maxentrack_neutron.GetXaxis().FindBin(bin_low)
        up_bin = f.Energy_neutrals_noVeto_maxentrack_neutron.GetXaxis().FindBin(bin_high)
        interaction_rates["neutrons"].append(f.Energy_neutrals_noVeto_maxentrack_neutron.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)
#        interaction_rates["neutrons_FTFP_BERT"].append(f.Energy_neutrals_noVeto_maxentrack_neutron.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)
 
        low_bin = f.Energy_neutrals_noVeto_maxentrack_antineutron.GetXaxis().FindBin(bin_low)
        up_bin = f.Energy_neutrals_noVeto_maxentrack_antineutron.GetXaxis().FindBin(bin_high)
        interaction_rates["neutrons"][-1] += (f.Energy_neutrals_noVeto_maxentrack_antineutron.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)
#        interaction_rates["neutrons_FTFP_BERT"][-1] += (f.Energy_neutrals_noVeto_maxentrack_antineutron.Integral(low_bin, up_bin)*target_lumi/calculated_lumi)

print(interaction_rates)

def getPassedCuts(f, useXYmap=False):
    if not useXYmap:
        return f.cutFlow.GetBinContent(f.cutFlow.GetNbinsX())
    else:
        f.cbmsim.GetEntry(0)
        pid = f.cbmsim.MCTrack[0].GetPdgCode()
        
        n_pass = 0.
        for event in tqdm(f.cbmsim):
            x = event.MCTrack[0].GetStartX()
            y = event.MCTrack[0].GetStartY()

            n_pass += xy_maps[pid].GetBinContent(xy_maps[pid].FindBin(x, y))
            
        return n_pass

for i_sample in samples :
    generated[i_sample] = []
    passed_cuts[i_sample] = []
    for bin_low, bin_high in bins :

        if bin_low == 0 :
            bin_low = 5
            
        p = BASE_FILTERED_DIR / i_sample / "{0}_{1}_{2}_tgtarea/filtered_stage1.root".format(samplekey[i_sample], bin_low, bin_high)
        f = ROOT.TFile(p.as_posix())
        if denominator_bin == 2:
            n_sel_stage_1 = f.cbmsim.GetEntries()
            n_minus_veto_stage_1 = f.n_minus_1_NoVetoHits_0.GetEntries()
            generated[i_sample].append(f.cutFlow.GetBinContent(1)/n_minus_veto_stage_1*n_sel_stage_1)
            passed_cuts[i_sample].append(getPassedCuts(f))
        else:
            pass
#            generated[i_sample].append(f.cutFlow.GetBinContent(denominator_bin))
#            passed_cuts[i_sample].append(f.cutFlow.GetBinContent(f.cutFlow_extended.GetNbinsX()))
        f.Close()

        p_high_stat = BASE_FILTERED_DIR / i_sample / "{0}_{1}_{2}_tgtarea_highstat/filtered_stage1.root".format(samplekey[i_sample], bin_low, bin_high)
        if p_high_stat.is_file() :
            f = ROOT.TFile(p_high_stat.as_posix())
            if denominator_bin == 2:
                n_sel_stage_1 = f.cbmsim.GetEntries()
                n_minus_veto_stage_1 = f.n_minus_1_NoVetoHits_0.GetEntries()
                generated[i_sample][-1] += f.cutFlow.GetBinContent(1)/n_minus_veto_stage_1*n_sel_stage_1
                passed_cuts[i_sample][-1] += getPassedCuts(f)
            else:
                pass
#                generated[i_sample][-1] += f.cutFlow.GetBinContent(denominator_bin)
#                passed_cuts[i_sample][-1] += f.cutFlow.GetBinContent(f.cutFlow_extended.GetNbinsX())
            f.Close()

header = ["Energy", "", "Interaction rate", "Generated (no veto hits)", "Passed cuts", "Efficiency", "Yield"]
format_header = "{:>6} {:>6} {:>18}  {:>25}  {:>12}  {:>12}  {:>12}"
format_row = "{:>6} {:>6} {:>18.2E}  {:>25.2E}  {:>12.2E}  {:>12.2E}  {:>12.2E}"

for i_sample in ["neutrons", "Kaons" ] : #"neutrons_FTFP_BERT", "Kaons", "Kaons_FTFP_BERT"] :
    print(i_sample)
    print(format_header.format(*header))
    for i_bin in range(len(bins)) :
        print(format_row.format(*[bins[i_bin][0], bins[i_bin][1], interaction_rates[i_sample][i_bin], generated[i_sample][i_bin], passed_cuts[i_sample][i_bin], passed_cuts[i_sample][i_bin]/generated[i_sample][i_bin], passed_cuts[i_sample][i_bin]/generated[i_sample][i_bin]*interaction_rates[i_sample][i_bin]]))

    print(format_row.format(*[bins[0][0], bins[-1][1], sum([interaction_rates[i_sample][i_bin] for i_bin in range(len(bins))]), 
                              sum([generated[i_sample][i_bin] for i_bin in range(len(bins))]), 
                              sum([passed_cuts[i_sample][i_bin]for i_bin in range(len(bins))]), 
                              sum([interaction_rates[i_sample][i_bin]*passed_cuts[i_sample][i_bin]/generated[i_sample][i_bin] for i_bin in range(len(bins))])/sum(interaction_rates[i_sample]), 
                              sum([passed_cuts[i_sample][i_bin]/generated[i_sample][i_bin]*interaction_rates[i_sample][i_bin] for i_bin in range(len(bins))])]))
    
    print()


# Signal
signal_generated_lumi = 399*100*nominal_target_mass/actual_target_mass # Correct for target mass here
f_sig_stage1 = ROOT.TFile((BASE_FILTERED_DIR / "nuMC" / "filtered_stage1.root").as_posix())


#Avg SciFi Ver channel in [200.000000,1200.000000] Hor in [300.000000,1336.000000]
#No hits in veto
#Remove events with hits in the last (hor) and two last (ver) DS planes
#Two or more consecutive SciFi planes
#More than 35 SciFi hits
#Total US QDC > 700.000000
stage_1_cuts = [["AvgSFChan", 2],
                ["USVarsVetoBottom", 2],
                ["USVarsVetoTop", 2],
                ["NoVetoHits", 1],
                ["TwoConsecutiveSciFiPlanes", 1],
                ["SciFiContinuity", 1],
                ["USPlanesHit", 1],
                ["NSciFiHits", 1],
                ["US_QDC", 1],
                ["NoLastDSPlanesHits", 1]]

"""
  KEY: TH1D	-1_AvgSFChan_0;1	AvgSFChan
  KEY: TH1D	-1_AvgSFChan_1;1	AvgSFChan
  KEY: TH1D	-1_USBarsVeto_0_2.000000_1_2.000000_0;1	USBarsVeto_0_2.000000_1_2.000000
  KEY: TH1D	-1_USBarsVeto_0_2.000000_1_2.000000_1;1	USBarsVeto_0_2.000000_1_2.000000
  KEY: TH1D	-1_USBarsVeto_0_8.000000_1_8.000000_0;1	USBarsVeto_0_8.000000_1_8.000000
  KEY: TH1D	-1_USBarsVeto_0_8.000000_1_8.000000_1;1	USBarsVeto_0_8.000000_1_8.000000
  KEY: TH1D	-1_NoVetoHits_0;1	NoVetoHits
  KEY: TH1D	-1_At least two consecutive SciFi planes_0;1	At least two consecutive SciFi planes
  KEY: TH1D	-1_SciFiContinuity_0;1	SciFiContinuity
  KEY: TH1D	-1_USPlanesHit_0;1	USPlanesHit
  KEY: TH1D	-1_SciFiMinHits_0;1	SciFiMinHits
  KEY: TH1D	-1_USQDC_0;1	USQDC
  KEY: TH1D	-1_DSVetoCut_0;1	DSVetoCut
"""
#                ["US_QDC", 1],
#                ["DSActivityCut", 1]]

stage_2_cuts = []
#stage_2_cuts = [["n_DS_tracks",1], 
#                ["event_dir", 1],
#                ["ds_track_extrap_fiducial", 1],
#                ["doca", 1],
#                ["scifi_nhits", 1],

#                ["DS_hits", 1]]

sig_species = ["nueCC", "numuCC", "nutauCC0mu", "nutauCC1mu", "NC"]

passes_cuts = {}
for i_species in sig_species :
    passes_cuts[i_species] = []
    for i_cut in range(-1, len(stage_1_cuts)) :
        passes_cuts[i_species].append(f_sig_stage1.Get(i_species+"_"+str(i_cut)+"_Enu").GetEntries()*target_lumi/signal_generated_lumi)
    if len(stage_2_cuts) :
        for i_cut in range(-1, len(stage_2_cuts)) :
            passes_cuts[i_species].append(f_sig_stage2.Get(i_species+"_"+str(i_cut)+"_Enu").GetEntries()*target_lumi/signal_generated_lumi)

header_sig = ["Cut"]
format_header_sig = "{:>38}"
format_sig_yield = "{:>38}"
format_sig_eff = "{:>38}"
for i_species in sig_species :
    header_sig.append(i_species)
    format_header_sig += " {:>14}"
    format_sig_yield += " {:>14.3f}"
    format_sig_eff += " {:>14.2E}"

print(format_header_sig.format(*header_sig))
print(format_sig_yield.format("", *[passes_cuts[i_species][0] for i_species in sig_species]))
for i_cut in range(len(stage_1_cuts)) :
    print(format_sig_yield.format(stage_1_cuts[i_cut][0], *[passes_cuts[i_species][i_cut+1] for i_species in sig_species]))
for i_cut in range(len(stage_2_cuts)) :
    this_i_cut = len(stage_1_cuts)+i_cut+2
    print(format_sig_yield.format(stage_2_cuts[i_cut][0], *[passes_cuts[i_species][this_i_cut] for i_species in sig_species]))
print()

print(format_header_sig.format(*header_sig))
print(format_sig_eff.format("", *([1]*len(sig_species))))
for i_cut in range(len(stage_1_cuts)) :
    print(format_sig_eff.format(stage_1_cuts[i_cut][0],*[passes_cuts[i_species][i_cut+1]/passes_cuts[i_species][i_cut] for i_species in sig_species]))
for i_cut in range(len(stage_2_cuts)) :
    this_i_cut = len(stage_1_cuts)+i_cut+2
    print(format_sig_eff.format(stage_2_cuts[i_cut][0],*[passes_cuts[i_species][this_i_cut]/passes_cuts[i_species][this_i_cut-1] for i_species in sig_species]))
print()

print(format_header_sig.format(*header_sig))
print(format_sig_eff.format("", *([1]*len(sig_species))))
for i_cut in range(len(stage_1_cuts)) :
    print(format_sig_eff.format(stage_1_cuts[i_cut][0],*[passes_cuts[i_species][i_cut+1]/passes_cuts[i_species][0] for i_species in sig_species]))
for i_cut in range(len(stage_2_cuts)) :
    this_i_cut = len(stage_1_cuts)+i_cut+2
    print(format_sig_eff.format(stage_2_cuts[i_cut][0],*[passes_cuts[i_species][this_i_cut]/passes_cuts[i_species][0] for i_species in sig_species]))
print()

f_data_stage1 = ROOT.TFile((BASE_FILTERED_DIR / "data_2022_2023" / "filtered_stage1.root").as_posix())

data_cutflow = f_data_stage1.cutFlow
print("{0:>85} {1:>16} {2:>16} {3:>16}".format("",
                                                "Event yield",
                                                "Cut-by-cut eff.",
                                                "Sequential eff."))
for i in range(1, data_cutflow.GetNbinsX()+1) :
    print("{0:>85} {1:>16.2E} {2:>16.2E} {3:>16.2E}".format(data_cutflow.GetXaxis().GetBinLabel(i),
                                                             data_cutflow.GetBinContent(i),
                                                             data_cutflow.GetBinContent(i)/data_cutflow.GetBinContent(i-1) if i > 1 else 1,
                                                             data_cutflow.GetBinContent(i)/data_cutflow.GetBinContent(1)))

if plot_cut_vars :
    import SNDLHCPlotter

    SNDLHCPlotter.init_style()

    ROOT.gROOT.SetBatch()

    
    os.makedirs("Plots/sequential", exist_ok = True)
    os.makedirs("Plots/nminusone", exist_ok = True)
    os.makedirs("Plots/finalsample", exist_ok = True)
    if f_out is None :
        f_out = ROOT.TFile("hadron_rates.root", "RECREATE")
    canvi = []

    ROOT.gStyle.SetOptStat(0)
    for i_cut in range(len(stage_1_cuts)) :
        for i_var in range(stage_1_cuts[i_cut][1]) :
            canvi.append(ROOT.TCanvas())
            print(str(i_cut-1)+"_"+stage_1_cuts[i_cut][0]+"_"+str(i_var))
            h_data = f_data_stage1.Get(str(i_cut-1)+"_"+stage_1_cuts[i_cut][0]+"_"+str(i_var))

            h_data.SetMarkerStyle(8)
            h_data.SetLineColor(ROOT.kBlack)
            h_data.SetMarkerColor(ROOT.kBlack)
            h_nu = f_sig_stage1.Get(str(i_cut-1)+"_"+stage_1_cuts[i_cut][0]+"_"+str(i_var))
            h_nu.Scale(target_lumi/signal_generated_lumi)
            h_nu.SetLineColor(ROOT.kAzure)
            
            h_data.Draw("E")
            h_nu.Draw("SAMEHIST")
            canvi[-1].RedrawAxis()
            canvi[-1].SaveAs("Plots/sequential/stage_1_"+str(i_cut)+"_"+str(i_var)+".png")
            canvi[-1].SaveAs("Plots/sequential/stage_1_"+str(i_cut)+"_"+str(i_var)+".pdf")
    
    for i_cut in range(len(stage_2_cuts)) :
        for i_var in range(stage_2_cuts[i_cut][1]) :
            canvi.append(ROOT.TCanvas())
            h_data = f_data_stage2.Get(str(i_cut-1)+"_"+stage_2_cuts[i_cut][0]+"_"+str(i_var))
            h_data.SetMarkerStyle(8)
            h_data.SetLineColor(ROOT.kBlack)
            h_data.SetMarkerColor(ROOT.kBlack)
            h_nu = f_sig_stage2.Get(str(i_cut-1)+"_"+stage_2_cuts[i_cut][0]+"_"+str(i_var))
            h_nu.Scale(target_lumi/signal_generated_lumi)
            h_nu.SetLineColor(ROOT.kAzure)
            
            h_data.Draw("E")
            h_nu.Draw("SAMEHIST")
            canvi[-1].RedrawAxis()
            canvi[-1].SaveAs("Plots/sequential/stage_2_"+str(i_cut)+"_"+str(i_var)+".png")
            canvi[-1].SaveAs("Plots/sequential/stage_2_"+str(i_cut)+"_"+str(i_var)+".pdf")

            canvi.append(ROOT.TCanvas())
            h_data = f_data_stage2.Get("n_minus_1_"+stage_2_cuts[i_cut][0]+"_"+str(i_var))
            h_data.SetMarkerStyle(8)
            h_data.SetLineColor(ROOT.kBlack)
            h_data.SetMarkerColor(ROOT.kBlack)
            h_nu = f_sig_stage2.Get("n_minus_1_"+stage_2_cuts[i_cut][0]+"_"+str(i_var))
            h_nu.Scale(target_lumi/signal_generated_lumi)
            h_nu.SetLineColor(ROOT.kAzure)
            h_data.Draw("E")
            h_nu.Draw("SAMEHIST")
            canvi[-1].RedrawAxis()
            canvi[-1].SaveAs("Plots/nminusone/stage_2_n_minus_1_"+str(i_cut)+"_"+str(i_var)+".png")
            canvi[-1].SaveAs("Plots/nminusone/stage_2_n_minus_1_"+str(i_cut)+"_"+str(i_var)+".pdf")

    final_sample_plot_vars = []
    import SndlhcGeo
    snd_geo = SndlhcGeo.GeoInterface("/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V5_2023.root")
    scifiDet = ROOT.gROOT.GetListOfGlobals().FindObject('Scifi')
    muFilterDet = ROOT.gROOT.GetListOfGlobals().FindObject('MuFilter')

    h_finalvar_data = []
    h_finalvar_signal = []
    h_finalvar_background = []

    def barycenter_XY(event, isMC = False) :
        if not isMC :
            scifiDet.InitEvent(event.EventHeader)
#            muFilterDet.InitEvent(event.EventHeader)
        a = ROOT.TVector3()
        b = ROOT.TVector3()
        
        n_x = 0
        mean_x = 0.
        n_y = 0
        mean_y = 0.

        for hit in event.Digi_ScifiHits :
            if not hit.isValid() :
                continue
            scifiDet.GetSiPMPosition(hit.GetDetectorID(), a, b)
            if hit.isVertical() :
                mean_x += (a.X() + b.X())/2.
                n_x += 1
            else :
                mean_y += (a.Y() + b.Y())/2.
                n_y += 1
        return [mean_x/n_x, mean_y/n_y]
#    final_sample_plot_vars.append(["sf_barycenter_xy", barycenter_XY, 2, [[8, -50, -10], [8, 15, 55]]])

    def eta(event, isMC = False) :
        if not isMC :
            scifiDet.InitEvent(event.EventHeader)
#            muFilterDet.InitEvent(event.EventHeader)
        a = ROOT.TVector3()
        b = ROOT.TVector3()
        
        n_x = 0
        mean_x = 0.
        n_y = 0
        mean_y = 0.

        for hit in event.Digi_ScifiHits :
            if not hit.isValid() :
                continue
            scifiDet.GetSiPMPosition(hit.GetDetectorID(), a, b)
            if hit.isVertical() :
                mean_x += (a.X() + b.X())/2.
                n_x += 1
            else :
                mean_y += (a.Y() + b.Y())/2.
                n_y += 1

        r = ((mean_x/n_x)**2 + (mean_y/n_y)**2)**0.5

        L = 48326.2 # use center of emulsion
        cot_theta = L/r
        cosec_theta = (L**2 + r**2)**0.5/r
        
        tan_half_theta = cosec_theta - cot_theta
        return -np.log(tan_half_theta)

#    final_sample_plot_vars.append(["eta", eta, 1, [20, 7, 9]])

    def radius(event, isMC = False) :
        if not isMC :
            scifiDet.InitEvent(event.EventHeader)
#            muFilterDet.InitEvent(event.EventHeader)
        a = ROOT.TVector3()
        b = ROOT.TVector3()
        
        n_x = 0
        mean_x = 0.
        n_y = 0
        mean_y = 0.

        for hit in event.Digi_ScifiHits :
            if not hit.isValid() :
                continue
            scifiDet.GetSiPMPosition(hit.GetDetectorID(), a, b)
            if hit.isVertical() :
                mean_x += (a.X() + b.X())/2.
                n_x += 1
            else :
                mean_y += (a.Y() + b.Y())/2.
                n_y += 1

        r = ((mean_x/n_x)**2 + (mean_y/n_y)**2)**0.5

        return r

#    final_sample_plot_vars.append(["r", radius, 1, [10, 25, 65]])

    def sf_vs_mufi(event, isMC = False) :
        if not isMC :
            scifiDet.InitEvent(event.EventHeader)
            muFilterDet.InitEvent(event.EventHeader)
            
        n_scifi = 0
        for hit in event.Digi_ScifiHits :
            if not hit.isValid() :
                continue
            n_scifi += 1
    
        US_QDC = 0.
        for hit in event.Digi_MuFilterHits :
            if hit.GetSystem() != 2 :
                continue
            if not hit.isValid() :
                continue
            for key, value in hit.GetAllSignals() :
                US_QDC += value
        if isMC :
            US_QDC -= 100
            
        return [n_scifi, US_QDC]
#    final_sample_plot_vars.append(["sf_mufi", sf_vs_mufi, 2, [[5, 0, 1000], [5, 600, 15600]]])

    def sf(event, isMC = False) :
        if not isMC :
            scifiDet.InitEvent(event.EventHeader)
            muFilterDet.InitEvent(event.EventHeader)
            
        n_scifi = 0
        for hit in event.Digi_ScifiHits :
            if not hit.isValid() :
                continue
            n_scifi += 1
    
        return n_scifi
    final_sample_plot_vars.append(["sf", sf, 1, [10, 0, 1000]])

    def mufi(event, isMC = False) :
        if not isMC :
            scifiDet.InitEvent(event.EventHeader)
            muFilterDet.InitEvent(event.EventHeader)
            
        US_QDC = 0.
        for hit in event.Digi_MuFilterHits :
            if hit.GetSystem() != 2 :
                continue
            if not hit.isValid() :
                continue
            for key, value in hit.GetAllSignals() :
                US_QDC += value
        if isMC :
            US_QDC -= 100
            
        return US_QDC
    final_sample_plot_vars.append(["mufi", mufi, 1, [10, 600, 15600]])

    t_neutrons = ROOT.TChain("cbmsim")
    t_Kaons = ROOT.TChain("cbmsim")

#    t_neutrons_track = ROOT.TChain("cbmsim")
#    t_Kaons_track = ROOT.TChain("cbmsim")

    for bin_low, bin_high in bins :
#        t_neutrons.Add("Filterv2_{0}_{1}_{2}/{0}_{1}_{2}_stage1.root".format("neutrons", bin_low, bin_high))
#        t_neutrons.Add("Filterv2_{0}_{1}_{2}_double/{0}_{1}_{2}_double_stage1.root".format("neutrons", bin_low, bin_high))
#        t_neutrons.Add("Filterv3_{0}_{1}_{2}_double/{0}_{1}_{2}_double_stage1.root".format("neutrons", bin_low, bin_high))

        t_neutrons.Add("Filterv3_tgtarea{0}_{1}_{2}_tgtarea/{0}_{1}_{2}_tgtarea_stage1.root".format("neutrons", bin_low, bin_high))


#        t_neutrons_track.Add("Filterv2_{0}_{1}_{2}/{0}_{1}_{2}_stage1_track.root".format("neutrons", bin_low, bin_high))
#        t_neutrons_track.Add("Filterv2_{0}_{1}_{2}_double/{0}_{1}_{2}_double_stage1_track.root".format("neutrons", bin_low, bin_high))
#        t_neutrons_track.Add("Filterv3_{0}_{1}_{2}_double/{0}_{1}_{2}_double_stage1_track.root".format("neutrons", bin_low, bin_high))

        t_Kaons.Add("Filterv2_{0}_{1}_{2}/{0}_{1}_{2}_stage1.root".format("Kaons", bin_low, bin_high))
        t_Kaons.Add("Filterv2_{0}_{1}_{2}_double/{0}_{1}_{2}_double_stage1.root".format("Kaons", bin_low, bin_high))
#        t_Kaons.Add("Filterv3_{0}_{1}_{2}_double/{0}_{1}_{2}_double_stage1.root".format("Kaons", bin_low, bin_high))

#        t_Kaons_track.Add("Filterv2_{0}_{1}_{2}/{0}_{1}_{2}_stage1_track.root".format("Kaons", bin_low, bin_high))
#        t_Kaons_track.Add("Filterv2_{0}_{1}_{2}_double/{0}_{1}_{2}_double_stage1_track.root".format("Kaons", bin_low, bin_high))
#        t_Kaons_track.Add("Filterv3_{0}_{1}_{2}_double/{0}_{1}_{2}_double_stage1_track.root".format("Kaons", bin_low, bin_high))

    t_signal = ROOT.TChain("cbmsim")
 #   t_signal_track = ROOT.TChain("cbmsim")
    t_signal.Add("Filterv3_nuMC/nuMC_stage1.root")
#    t_signal_track.Add("Filterv3_nuMC/nuMC_stage1_track.root")

    t_data = ROOT.TChain("rawConv")
    t_data.Add("Filter_nue_data_2022/filtered_MC_stage1.root")

#    print(t_neutrons.GetEntries(), t_neutrons_track.GetEntries())
#    t_neutrons.AddFriend(t_neutrons_track)
#    print(t_Kaons.GetEntries(), t_Kaons_track.GetEntries())
#    t_Kaons.AddFriend(t_Kaons_track)
#    print(t_signal.GetEntries(), t_signal_track.GetEntries())
#    t_signal.AddFriend(t_signal_track)
    
    for var_name, var_fun, nD, bins in final_sample_plot_vars :
        if nD == 2 :
            h_finalvar_data.append(ROOT.TGraph())
            h_finalvar_data[-1].SetName(var_name+"_data")
            h_finalvar_signal.append(ROOT.TH2D(var_name+"_signal", var_name+"_signal", 
                                               bins[0][0], bins[0][1], bins[0][2],
                                               bins[1][0], bins[1][1], bins[1][2]))
            h_finalvar_background.append(ROOT.TH2D(var_name+"_background", var_name+"_background", 
                                                   bins[0][0], bins[0][1], bins[0][2],
                                                   bins[1][0], bins[1][1], bins[1][2]))
        elif nD == 1 :
            
            h_finalvar_data.append(ROOT.TH1D(var_name+"_data", var_name+"_data", 
                                               bins[0], bins[1], bins[2]))
            h_finalvar_signal.append(ROOT.TH1D(var_name+"_signal", var_name+"_signal", 
                                               bins[0]*MC_bin_factor, bins[1], bins[2]))
            h_finalvar_background.append(ROOT.TH1D(var_name+"_background", var_name+"_background", 
                                                   bins[0]*MC_bin_factor, bins[1], bins[2]))
        else :
            print("Number of histogram dimensions not supported")
            exit()
                  
        for event in t_neutrons :
            ret = var_fun(event, True)

            i_sample = "neutrons"
            E = event.MCTrack.At(0).GetEnergy()
            if E < 100 :
                i_bin = int(E/10)
            elif E < 150 :
                i_bin = 10
            elif E < 200 :
                i_bin = 11
            else :
                print("Uknown energy bin.")
                exit(-1)
            
            if nD == 2 :
                h_finalvar_background[-1].Fill(ret[0], ret[1], 1./generated[i_sample][i_bin]*interaction_rates[i_sample][i_bin])
            elif nD == 1 :
                h_finalvar_background[-1].Fill(ret, 1./generated[i_sample][i_bin]*interaction_rates[i_sample][i_bin]/MC_bin_factor)

        for event in t_Kaons :

            i_sample = "Kaons"
            E = event.MCTrack.At(0).GetEnergy()
            if E < 100 :
                i_bin = int(E/10)
            elif E < 150 :
                i_bin = 10
            elif E < 200 :
                i_bin = 11
            else :
                print("Uknown energy bin.")
                exit(-1)

            ret = var_fun(event, True)
            if nD == 2 :
                h_finalvar_background[-1].Fill(ret[0], ret[1], 1./generated[i_sample][i_bin]*interaction_rates[i_sample][i_bin])
#                print("background {} {} {} {}".format(var_name, ret[0], ret[1], 1./generated[i_sample][i_bin]*interaction_rates[i_sample][i_bin]))
            elif nD == 1 :
                h_finalvar_background[-1].Fill(ret, 1./generated[i_sample][i_bin]*interaction_rates[i_sample][i_bin]*MC_bin_factor)

        for event in t_signal :
            ret = var_fun(event, True)
            if nD == 2 :
                h_finalvar_signal[-1].Fill(ret[0], ret[1], target_lumi/signal_generated_lumi)
#                print("signal {} {} {} 1.0".format(var_name, ret[0], ret[1]))
            elif nD == 1 :
                h_finalvar_signal[-1].Fill(ret, target_lumi/signal_generated_lumi*MC_bin_factor)

        for event in t_data :
            ret = var_fun(event, True)
            if nD == 2 :
                h_finalvar_data[-1].AddPoint(ret[0], ret[1])
            elif nD == 1 :
                h_finalvar_data[-1].Fill(ret)
            
        if nD == 2 :
            canvi.append(ROOT.TCanvas())
            h_finalvar_background[-1].Draw("COLZ")
            h_finalvar_data[-1].Draw("SAME*")
            canvi[-1].SaveAs("Plots/finalsample/test_background_"+var_name+".png")
            canvi.append(ROOT.TCanvas())
            h_finalvar_signal[-1].Draw("COLZ")
            h_finalvar_data[-1].Draw("SAME*")
            canvi[-1].SaveAs("Plots/finalsample/test_signal_"+var_name+".png")

        elif nD == 1 :
            canvi.append(ROOT.TCanvas())

#            h_finalvar_signal[-1].Scale(1/h_finalvar_signal[-1].Integral())
#            h_finalvar_background[-1].Scale(1./h_finalvar_background[-1].Integral())

            max_val = np.max([h_finalvar_signal[-1].GetMaximum(),  h_finalvar_background[-1].GetMaximum()])
            import scipy.stats
            
            xaxlabelfont = 4 
            xaxlabelsize = 22
            yaxlabelfont = 4; yaxlabelsize = 22
            axtitlefont = 6; axtitlesize = 26
            legendfont = 5
            leftmargin = 0.15
            rightmargin = 0.05
            topmargin = 0.05
            bottommargin = 0.15
            canvi[-1].SetBottomMargin(bottommargin)
            canvi[-1].SetLeftMargin(leftmargin)
            canvi[-1].SetRightMargin(rightmargin)
            canvi[-1].SetTopMargin(topmargin)

            h_finalvar_data[-1].SetMarkerColor(ROOT.kBlack)
            h_finalvar_data[-1].SetLineColor(ROOT.kBlack)
            h_finalvar_data[-1].SetMarkerStyle(20)
            h_finalvar_data[-1].SetBinErrorOption(ROOT.TH1.kPoisson)
            h_finalvar_data[-1].Draw("E")

            xaxtitle = "Number of SciFi hits"
            yaxtitle = "Events / 100 SciFi hits"

            xax = h_finalvar_data[-1].GetXaxis()
            #xax.SetNdivisions(5,4,0,ROOT.kTRUE)
            xax.SetLabelFont(10*xaxlabelfont+3)
            xax.SetLabelSize(xaxlabelsize)

            if xaxtitle is not None:
                xax.SetTitle(xaxtitle)
            xax.SetTitleFont(10*axtitlefont+3)
            xax.SetTitleSize(axtitlesize)
            xax.SetTitleOffset(1.2)
        
            yax = h_finalvar_data[-1].GetYaxis()
            yax.SetMaxDigits(3)
            yax.SetNdivisions(8,4,0,ROOT.kTRUE)
            yax.SetLabelFont(10*yaxlabelfont+3)
            yax.SetLabelSize(yaxlabelsize)
        
            if yaxtitle is not None: 
                yax.SetTitle(yaxtitle)
            yax.SetTitleFont(10*axtitlefont+3)
            yax.SetTitleSize(axtitlesize)
            yax.SetTitleOffset(1.2)
    
            h_finalvar_signal[-1].SetMaximum(1.1*max_val)
            h_finalvar_signal[-1].SetMinimum(0.00001)
            h_finalvar_signal[-1].SetLineColor(ROOT.TColor.GetColor(signal_color))
            h_finalvar_signal[-1].Draw("SAMEHIST")
            h_finalvar_background[-1].SetLineColor(ROOT.TColor.GetColor(background_color))
            h_finalvar_background[-1].Draw("SAMEHIST")
            bck_5sig = h_finalvar_background[-1].Clone("background_five")
            print(bck_5sig.Integral())
            bck_5sig.Scale(scipy.stats.poisson.isf(scipy.stats.norm.sf(5), 0.078)/bck_5sig.Integral(0, -1)*MC_bin_factor)
            bck_5sig.SetLineStyle(2)
            bck_5sig.SetLineWidth(3)
            bck_5sig.Draw("SAMEHIST")
            h_finalvar_data[-1].Draw("SAMEE")

            SNDLHCPlotter.writeSND(canvi[-1], text_in=False, extratext = "")

            pentryheight = 0.06
            nentries = 1 + 5
            if nentries>3: pentryheight = pentryheight*0.8
            plegendbox = ([leftmargin+0.45,1-topmargin-pentryheight*nentries, 1-rightmargin-0.03,1-topmargin-0.03])
            
            legend = ROOT.TLegend(plegendbox[0],plegendbox[1],plegendbox[2],plegendbox[3])
            legend.SetNColumns(1)
            legend.SetName('legend')
            legend.SetFillColor(ROOT.kWhite)
            legend.SetTextFont(10*legendfont+3)
            legend.SetBorderSize(0)

            legend.AddEntry(h_finalvar_data[-1], "Data", "PEL")
            legend.AddEntry(h_finalvar_signal[-1], "Neutrino simulation", "L")
            legend.AddEntry(h_finalvar_background[-1], "Neutral hadron simulation", "L")
            legend.AddEntry(bck_5sig, "#splitline{Neutral hadron simulation}{at 5 #sigma limit}", "L")
            legend.Draw()
            canvi[-1].RedrawAxis()
            ROOT.gPad.Update()
            canvi[-1].SaveAs("Plots/finalsample/test_"+var_name+".png")
            canvi[-1].SaveAs("Plots/finalsample/test_"+var_name+".pdf")


#            canvi[-1].SetLogy()
#            canvi[-1].SaveAs("Plots/finalsample/test_"+var_name+"_logy.png")
#            canvi[-1].SaveAs("Plots/finalsample/test_"+var_name+"_logy.pdf")

            canvi.append(ROOT.TCanvas())
            canvi[-1].SetBottomMargin(bottommargin)
            canvi[-1].SetLeftMargin(leftmargin)
            canvi[-1].SetRightMargin(rightmargin)
            canvi[-1].SetTopMargin(topmargin)

            h_finalvar_data[-1].Draw("E")
            

            xaxtitle = "Number of SciFi hits"
            yaxtitle = "Events / 100 SciFi hits"

            xax = h_finalvar_data[-1].GetXaxis()
            #xax.SetNdivisions(5,4,0,ROOT.kTRUE)
            xax.SetLabelFont(10*xaxlabelfont+3)
            xax.SetLabelSize(xaxlabelsize)

            if xaxtitle is not None:
                xax.SetTitle(xaxtitle)
            xax.SetTitleFont(10*axtitlefont+3)
            xax.SetTitleSize(axtitlesize)
            xax.SetTitleOffset(1.2)
        
            yax = h_finalvar_data[-1].GetYaxis()
            yax.SetMaxDigits(3)
            yax.SetNdivisions(8,4,0,ROOT.kTRUE)
            yax.SetLabelFont(10*yaxlabelfont+3)
            yax.SetLabelSize(yaxlabelsize)
        
            if yaxtitle is not None: 
                yax.SetTitle(yaxtitle)
            yax.SetTitleFont(10*axtitlefont+3)
            yax.SetTitleSize(axtitlesize)
            yax.SetTitleOffset(1.2)

            stack = ROOT.THStack()
            h_finalvar_signal[-1].SetFillColor(ROOT.TColor.GetColor(signal_color))
            h_finalvar_signal[-1].SetFillStyle(signal_fill_style)
            h_finalvar_signal[-1].SetLineWidth(1)

            h_finalvar_background[-1].SetFillColor(ROOT.TColor.GetColor(background_color))
            h_finalvar_background[-1].SetFillStyle(background_fill_style)
            h_finalvar_background[-1].SetLineWidth(1)

            stack.Add(h_finalvar_signal[-1])
            stack.Add(h_finalvar_background[-1])

            stack.SetMinimum(0.00001)
            stack.Draw("SAMEHIST")
        
            bck_5sig.Draw("SAMEHIST")
            h_finalvar_data[-1].Draw("SAMEE")

            SNDLHCPlotter.writeSND(canvi[-1], text_in=False, extratext = "")

            legend_stack = ROOT.TLegend(plegendbox[0],plegendbox[1],plegendbox[2],plegendbox[3])
            legend_stack.SetNColumns(1)
            legend_stack.SetName('legend_stack')
            legend_stack.SetFillColor(ROOT.kWhite)
            legend_stack.SetTextFont(10*legendfont+3)
            legend_stack.SetBorderSize(0)

            legend_stack.AddEntry(h_finalvar_data[-1], "Data", "PEL")
            legend_stack.AddEntry(h_finalvar_signal[-1], "Neutrino simulation", "f")
            legend_stack.AddEntry(h_finalvar_background[-1], "Neutral hadron simulation", "f")
            legend_stack.AddEntry(bck_5sig, "#splitline{Neutral hadron simulation}{at 5 #sigma limit}", "L")
            legend_stack.Draw()

            canvi[-1].RedrawAxis()
            ROOT.gPad.Update()

            canvi[-1].SaveAs("Plots/finalsample/test_"+var_name+"_stack.png")
            canvi[-1].SaveAs("Plots/finalsample/test_"+var_name+"_stack.pdf")

            canvi.append(ROOT.TCanvas())
            canvi[-1].SetBottomMargin(bottommargin)
            canvi[-1].SetLeftMargin(leftmargin)
            canvi[-1].SetRightMargin(rightmargin)
            canvi[-1].SetTopMargin(topmargin)

            h_finalvar_data[-1].Draw("E")
            

            xaxtitle = "Number of SciFi hits"
            yaxtitle = "Events / 100 SciFi hits"

            xax = h_finalvar_data[-1].GetXaxis()
            #xax.SetNdivisions(5,4,0,ROOT.kTRUE)
            xax.SetLabelFont(10*xaxlabelfont+3)
            xax.SetLabelSize(xaxlabelsize)

            if xaxtitle is not None:
                xax.SetTitle(xaxtitle)
            xax.SetTitleFont(10*axtitlefont+3)
            xax.SetTitleSize(axtitlesize)
            xax.SetTitleOffset(1.2)
        
            yax = h_finalvar_data[-1].GetYaxis()
            yax.SetMaxDigits(3)
            yax.SetNdivisions(8,4,0,ROOT.kTRUE)
            yax.SetLabelFont(10*yaxlabelfont+3)
            yax.SetLabelSize(yaxlabelsize)
        
            if yaxtitle is not None: 
                yax.SetTitle(yaxtitle)
            yax.SetTitleFont(10*axtitlefont+3)
            yax.SetTitleSize(axtitlesize)
            yax.SetTitleOffset(1.2)

            stack = ROOT.THStack()
            h_finalvar_signal[-1].SetFillColor(ROOT.TColor.GetColor(signal_color))
            h_finalvar_signal[-1].SetFillStyle(signal_fill_style)
            h_finalvar_signal[-1].SetLineWidth(1)

            h_finalvar_background[-1].SetFillColor(ROOT.TColor.GetColor(background_color))
            h_finalvar_background[-1].SetFillStyle(background_fill_style)
            h_finalvar_background[-1].SetLineWidth(1)

            stack.Add(h_finalvar_signal[-1])
#            stack.Add(h_finalvar_background[-1])

            stack.SetMinimum(0.00001)
            stack.Draw("SAMEHIST")
        
            bck_5sig.Draw("SAMEHIST")
            h_finalvar_data[-1].Draw("SAMEE")

            SNDLHCPlotter.writeSND(canvi[-1], text_in=False, extratext = "")

            legend_stack = ROOT.TLegend(plegendbox[0],plegendbox[1],plegendbox[2],plegendbox[3])
            legend_stack.SetNColumns(1)
            legend_stack.SetName('legend_stack')
            legend_stack.SetFillColor(ROOT.kWhite)
            legend_stack.SetTextFont(10*legendfont+3)
            legend_stack.SetBorderSize(0)

            legend_stack.AddEntry(h_finalvar_data[-1], "Data", "PEL")
            legend_stack.AddEntry(h_finalvar_signal[-1], "Neutrino simulation", "f")
#            legend_stack.AddEntry(h_finalvar_background[-1], "Neutral hadron simulation", "f")
            legend_stack.AddEntry(bck_5sig, "#splitline{Neutral hadron simulation}{at 5 #sigma limit}", "L")
            legend_stack.Draw()

            canvi[-1].RedrawAxis()
            ROOT.gPad.Update()

            canvi[-1].SaveAs("Plots/finalsample/test_"+var_name+"_stack_no_background.png")
            canvi[-1].SaveAs("Plots/finalsample/test_"+var_name+"_stack_no_background.pdf")
            
#            canvi[-1].SetLogy()
#            canvi[-1].SaveAs("Plots/finalsample/test_"+var_name+"_stack_logy.png")
#            canvi[-1].SaveAs("Plots/finalsample/test_"+var_name+"_stack_logy.pdf")
    
if do_stats :    
    # Toy MC
    import scipy.stats
    import heapq
    # 1 - Get distribution of Poisson means giving 0, 1 and 2 events, using a flat prior from 0 to 50
    mu_max = 75
    n_per_trial = 1000000
    n_trials = 500
    N_max = 25
    
    from array import array 
    if f_out is None :
        f_out = ROOT.TFile("hadron_rates.root", "RECREATE")
    
    a = {}
    t = {}
    
    for output in ["neutrons", "Kaons", "total"] :
        for suffix in ["", "_exclude_0_10", "_FTFP_BERT", "_FTFP_BERT_exclude_0_10"] :
            a[output+suffix] = array('f', [0])
            t[output+suffix] = ROOT.TTree(output+suffix, output+suffix)
            t[output+suffix].Branch('lambda', a[output+suffix], 'lambda/F')
    
    
    for i in range(N_max+1) :
        this_name = "poisson_k_"+str(i)
        a[this_name] = array('f', [0])
        t[this_name] = ROOT.TTree(this_name, this_name)
        t[this_name].Branch('lambda', a[this_name], this_name+'/F')
     
    pois_mus = []
    for i in range(N_max+1) :
        pois_mus.append(np.array([]))
    
    for i_trial in range(n_trials) :
        print("Trial {0}".format(i_trial))
        these_mus = scipy.stats.uniform.rvs(size=n_per_trial)*mu_max
        these_throws = scipy.stats.poisson.rvs(mu=these_mus)
        for i in range(N_max+1) :
            pois_mus[i] = np.concatenate((pois_mus[i], these_mus[these_throws==i]))
            pois_mus[i].sort(kind = "mergesort")
    
    for i_mu, k_mu in enumerate(pois_mus) :
        for mu in k_mu :
            a["poisson_k_"+str(i_mu)][0] = mu
            t["poisson_k_"+str(i_mu)].Fill()
    
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(N_max+1) :
        plt.hist(pois_mus[i], label = str(i), bins = 1000, range = (0, 100), histtype = "step")
    plt.legend()
    plt.yscale("log")
    plt.ylim(bottom = 0.5)
    plt.savefig("poisson_means.png")
    
    reverse_quantiles = [0.5, 1-3.173E-01/2, 3.173E-01/2, 2.700E-03, 5.733E-07]
    
    for i in range(N_max+1) :
        print(i, len(pois_mus[i]), [pois_mus[i][-int(rq*len(pois_mus[i]))] for rq in reverse_quantiles], pois_mus[i][-1])
    
    print()
    
    header = ["Energy", "", "Mean", "0.500", "0.159", "0.842", "0.997", "5 sigma"]
    format_header = "{:>6} {:>6} {:>6} {:>6}  {:>25}  {:>12}  {:>12}  {:>12}"
    format_row = "{:>6} {:>6} {:>6.2E} {:>6.2E} {:>25.2E}  {:>12.2E}  {:>12.2E}  {:>12.2E}"
    
    minimum_array_size = int(min([len(pois_mus[i]) for i in range(N_max+1)]))
    
    tot = {}
    tot_exclude_0_10 = {}
    
    for i_sample in ["neutrons", "Kaons"] : #["neutrons", "neutrons_FTFP_BERT", "Kaons", "Kaons_FTFP_BERT"] :
        print(i_sample)
        print(format_header.format(*header))
        for i_bin in range(len(bins)) :
            print(format_row.format(*([bins[i_bin][0], bins[i_bin][1], np.mean(pois_mus[int(passed_cuts[i_sample][i_bin])])/generated[i_sample][i_bin]*interaction_rates[i_sample][i_bin]] + [pois_mus[int(passed_cuts[i_sample][i_bin])][-int(reverse_quantiles[q]*len(pois_mus[int(passed_cuts[i_sample][i_bin])]))]/generated[i_sample][i_bin]*interaction_rates[i_sample][i_bin] for q in [0, 1, 2, 3, 4]])))
        
    
        summed = [np.random.choice(pois_mus[int(passed_cuts[i_sample][i_bin])], size=minimum_array_size, replace=False)/generated[i_sample][i_bin]*interaction_rates[i_sample][i_bin] for i_bin in range(len(bins))]
        summed = np.vstack(summed)
    
        tot[i_sample] = summed.sum(axis = 0)
        for l in tot[i_sample] :
            a[i_sample][0] = l
            t[i_sample].Fill()
                
            
        tot[i_sample].sort()
        tot_exclude_0_10[i_sample] = summed[1:].sum(axis = 0)

        for l in tot_exclude_0_10[i_sample] :
            a[i_sample+"_exclude_0_10"][0] = l
            t[i_sample+"_exclude_0_10"].Fill()
        tot_exclude_0_10[i_sample].sort()
        print(format_row.format(*([0, 200, np.mean(tot[i_sample])] + [tot[i_sample][-int(reverse_quantiles[q]*minimum_array_size)] for q in [0, 1, 2, 3, 4]])))
        print(format_row.format(*([10, 200, np.mean(tot_exclude_0_10[i_sample])] + [tot_exclude_0_10[i_sample][-int(reverse_quantiles[q]*minimum_array_size)] for q in [0, 1, 2, 3, 4]])))
    
    print("neutrons + Kaons")
    summed = np.vstack([np.random.choice(tot["neutrons"], size = minimum_array_size, replace = False), 
                        np.random.choice(tot["Kaons"], size = minimum_array_size, replace = False)])
    tot_default = summed.sum(axis = 0)
    
    for l in tot_default :
        a["total"][0] = l
        t["total"].Fill()
    
    tot_default.sort()
    
    summed_exclude_0_10 = np.vstack([np.random.choice(tot_exclude_0_10["neutrons"], size = minimum_array_size, replace = False), 
                                     np.random.choice(tot_exclude_0_10["Kaons"], size = minimum_array_size, replace = False)])
    tot_default_exclude_0_10 = summed_exclude_0_10.sum(axis = 0)
    for l in tot_default_exclude_0_10 :
        a["total_exclude_0_10"][0] = l
        t["total_exclude_0_10"].Fill()
    
    tot_default_exclude_0_10.sort()
    print("Lambda")
    print(format_row.format(*([0, 200, np.mean(tot_default)] + [tot_default[-int(reverse_quantiles[q]*minimum_array_size)] for q in [0, 1, 2, 3, 4]])))
    print(format_row.format(*([10, 200, np.mean(tot_default_exclude_0_10)] + [tot_default_exclude_0_10[-int(reverse_quantiles[q]*minimum_array_size)] for q in [0, 1, 2, 3, 4]])))
    
    tot_default_k = scipy.stats.poisson.rvs(mu=tot_default)
    tot_default_k.sort()
    tot_default_k_exclude_0_10 = scipy.stats.poisson.rvs(mu=tot_default_exclude_0_10)
    tot_default_k_exclude_0_10.sort()
    print("k")
    print(format_row.format(*([0, 200, np.mean(tot_default_k)] + [tot_default_k[-int(reverse_quantiles[q]*minimum_array_size)] for q in [0, 1, 2, 3, 4]])))
    print(format_row.format(*([10, 200, np.mean(tot_default_k_exclude_0_10)] + [tot_default_k_exclude_0_10[-int(reverse_quantiles[q]*minimum_array_size)] for q in [0, 1, 2, 3, 4]])))
    
    print("neutrons + Kaons FTFP BERT")
    summed = np.vstack([np.random.choice(tot["neutrons_FTFP_BERT"], size = minimum_array_size, replace = False), 
                        np.random.choice(tot["Kaons_FTFP_BERT"], size = minimum_array_size, replace = False)])
    tot_ftfp_bert = summed.sum(axis = 0)

    for l in tot_ftfp_bert :
        a["total_FTFP_BERT"][0] = l
        t["total_FTFP_BERT"].Fill()


    tot_ftfp_bert.sort()
    
    summed_exclude_0_10 = np.vstack([np.random.choice(tot_exclude_0_10["neutrons_FTFP_BERT"], size = minimum_array_size, replace = False), 
                                     np.random.choice(tot_exclude_0_10["Kaons_FTFP_BERT"], size = minimum_array_size, replace = False)])
    
    tot_ftfp_bert_exclude_0_10 = summed_exclude_0_10.sum(axis = 0)

    for l in tot_ftfp_bert_exclude_0_10 :
        a["total_FTFP_BERT_exclude_0_10"][0] = l
        t["total_FTFP_BERT_exclude_0_10"].Fill()


    tot_ftfp_bert_exclude_0_10.sort()
    print("Lambda")
    print(format_row.format(*([0, 200, np.mean(tot_ftfp_bert)] + [tot_ftfp_bert[-int(reverse_quantiles[q]*minimum_array_size)] for q in [0, 1, 2, 3, 4]])))
    print(format_row.format(*([10, 200, np.mean(tot_ftfp_bert_exclude_0_10)] + [tot_ftfp_bert_exclude_0_10[-int(reverse_quantiles[q]*minimum_array_size)] for q in [0, 1, 2, 3, 4]])))
    
    tot_ftfp_bert_k = scipy.stats.poisson.rvs(mu=tot_ftfp_bert)
    tot_ftfp_bert_k.sort()
    tot_ftfp_bert_k_exclude_0_10 = scipy.stats.poisson.rvs(mu=tot_ftfp_bert_exclude_0_10)
    tot_ftfp_bert_k_exclude_0_10.sort()
    print("k")
    print(format_row.format(*([0, 200, np.mean(tot_ftfp_bert_k)] + [tot_ftfp_bert_k[-int(reverse_quantiles[q]*minimum_array_size)] for q in [0, 1, 2, 3, 4]])))
    print(format_row.format(*([10, 200, np.mean(tot_ftfp_bert_k_exclude_0_10)] + [tot_ftfp_bert_k_exclude_0_10[-int(reverse_quantiles[q]*minimum_array_size)] for q in [0, 1, 2, 3, 4]])))
    
    print("significance of 8 observed events")
    q_8 = np.searchsorted(tot_default_k, 8)
    q_8_exclude_0_10 = np.searchsorted(tot_default_k_exclude_0_10, 8)
    print(q_8, q_8_exclude_0_10, len(tot_default_k))
    print("           ", scipy.stats.norm.isf((-q_8+len(tot_default_k))/len(tot_default_k)/2.))
    print("excl. 0_10 ", scipy.stats.norm.isf((-q_8_exclude_0_10+len(tot_default_k_exclude_0_10))/len(tot_default_k_exclude_0_10)/2.))

if make_background_tree :
    #numpify bins
    bins = np.array(bins)
    
    prescale = 1.0

    in_background_tree = ROOT.TChain("cbmsim")

    for i_sample in samples :
        for bin_low, bin_high in bins :
            if bin_low == 0 :
                bin_low = 5

            p = BASE_FILTERED_DIR / i_sample / "{0}_{1}_{2}_tgtarea/filtered_stage1.root".format(samplekey[i_sample], bin_low, bin_high)
            in_background_tree.Add(p.as_posix())

            p_high_stat = BASE_FILTERED_DIR / i_sample / "{0}_{1}_{2}_tgtarea_highstat/filtered_stage1.root".format(samplekey[i_sample], bin_low, bin_high)
            if p_high_stat.is_file():
                in_background_tree.Add(p_high_stat.as_posix())
    
    out_background_file = ROOT.TFile("neutron_kaon_nue_stage1_noprescale.root", "RECREATE")
    background_tree = in_background_tree.CloneTree(0)

    weight = array('f', [0.])
    weight_branch = background_tree.Branch("per_invfb_weight", weight, 'per_invfb_weight/F')

    for i in tqdm(range(in_background_tree.GetEntries())) :

        if np.random.uniform() > prescale :
            continue

        in_background_tree.GetEntry(i)

        primaryPDG = in_background_tree.MCTrack[0].GetPdgCode()
        primaryEnergy = in_background_tree.MCTrack[0].GetEnergy()
        
        try :
            this_bin = np.where(np.logical_and(primaryEnergy >= bins[:,0], primaryEnergy < bins[:,1]))[0][0]
        except IndexError :
            print("ENERGY BIN OUT OF BOUNDS!")
            print(primaryPDG, primaryEnergy)
            if primaryEnergy > 80 :
                this_bin = len(bins)-1
            else :
                this_bin = -1

        if primaryPDG == 2112 : 
            this_sample = "neutrons"
        elif primaryPDG == 130 :
            this_sample = "Kaons"
        else :
            raise RuntimeError("Unknown particle type: {}".format(primaryPDG))
        if this_bin >= 0 :
            weight[0] = 1./generated[this_sample][this_bin]*interaction_rates[this_sample][this_bin]/target_lumi/prescale # Divide by target lumi to give weight per inverse femtobarn
        else :
            weight[0] = 0.

        background_tree.Fill()
    out_background_file.Write()
    out_background_file.Close()


if f_out is not None :
    f_out.Write()
    f_out.Close()