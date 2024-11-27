import os
import ROOT
from tqdm import tqdm



ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

def draw(converted_df):
    # Get the maximum and minimum plane number
    max_plane = converted_df.Max("plane").GetValue()
    min_plane = converted_df.Min("plane").GetValue()
    print(f"Max plane number in converted file ({folder}): {max_plane}")
    print(f"Min plane number in converted file ({folder}): {min_plane}")

    # Draw distribution of 'plane'
    canvas = ROOT.TCanvas("c1", "Plane Distribution", 800, 600)
    hist = converted_df.Histo1D(("plane_hist", "Plane Distribution;Plane;Counts",100, min_plane, max_plane+1), "plane")
    hist.Draw()
    canvas.SaveAs(f"{out_dir}/plane_distribution_{folder}.png")

    # Draw distribution of 'detId'
    canvas_detid = ROOT.TCanvas("c2", "detId Distribution", 800, 600)
    hist_detid = converted_df.Histo1D(("detid_hist", "detId Distribution;detId;Counts", 100, 0, 1e7), "Hits.detId")
    hist_detid.Draw()
    canvas_detid.SaveAs(f"{out_dir}/detid_distribution_{folder}.png")


def main():
    converted_dir = '/eos/user/z/zhibin/sndData/converted/real_muon/muon_2023_reprocess/'
    original_dir = '/eos/experiment/sndlhc/convertedData/physics/2023_reprocess/'
    out_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot_eval'

    total_original_events = 0
    total_converted_events = 0
    total_station_one_events = 0
    total_station_two_events = 0
    total_station_one_or_two_events = 0

    folders = os.listdir(converted_dir)

    count = 0
    for folder in tqdm(folders, desc="Processing Folders", unit="folder"):
        converted_file = os.path.join(converted_dir, folder, "*.root")
        original_file = os.path.join(original_dir, folder, "*.root")

        # Read original_file as RDataFrame and get total events
        original_df = ROOT.RDataFrame("cbmsim", original_file)
        original_events = original_df.Count().GetValue()
        total_original_events += original_events
        #print(f"Total events in original file ({folder}): {original_events}")

        # Read converted_file as RDataFrame
        converted_df = ROOT.RDataFrame("cbmsim", converted_file)
        converted_df = converted_df.Define("station", "Hits.detId/1000000")

        converted_events = converted_df.Count().GetValue()
        total_converted_events += converted_events

        # Count the total number of events that no hits where station == 1
        station_one_filter = converted_df.Filter("(ROOT::VecOps::All(station != 1))")
        station_one_events = station_one_filter.Count().GetValue()
        total_station_one_events += station_one_events

        # Count the total number of events that no hits where station == 2
        station_two_filter = converted_df.Filter("!(ROOT::VecOps::All(station != 2))")
        station_two_events = station_two_filter.Count().GetValue()
        total_station_two_events += station_two_events

        # Count the total number of events that no hits where station == 1 and station == 2
        station_one_or_two_filter = converted_df.Filter("ROOT::VecOps::All(station != 1) and ROOT::VecOps::All(station != 2)")
        station_one_or_two_events = station_one_or_two_filter.Count().GetValue()
        total_station_one_or_two_events += station_one_or_two_events

        count+=1
        #if count >10:
        #    break

    if total_original_events > 0:
        veto_eff = total_converted_events / total_original_events
    else:
        veto_eff = 0

    if total_converted_events > 0:
        scifi_ineff_1 = (total_station_one_events) / total_converted_events
        scifi_ineff_2 = (total_station_two_events) / total_converted_events
        scifi_ineff_1and2 = (total_station_one_or_two_events) / total_converted_events
    else:
        scifi_ineff_1 = 0
        scifi_ineff_2 = 0
        scifi_ineff_1and2 = 0

    # Print out the sums and efficiencies after the loop
    print("\nSummary of all folders:")
    print(f"Total original events across all folders: {total_original_events:.2e}")
    print(f"Total converted events across all folders: {total_converted_events:.2e}")
    print(f"Total events without hits in station 1 across all folders: {total_station_one_events:.2e}")
    print(f"Total events without hits in station 2 across all folders: {total_station_two_events:.2e}")
    print(f"Total events without hits in station 1 and 2 across all folders: {total_station_one_or_two_events:.2e}")

    # Print efficiencies and inefficiencies
    print("\nEfficiencies and Inefficiencies:")
    print(f"Veto Efficiency: {veto_eff:.2e}")
    print(f"SciFi Inefficiency for Station 1: {scifi_ineff_1:.2e}")
    print(f"SciFi Inefficiency for Station 2: {scifi_ineff_2:.2e}")
    print(f"SciFi Inefficiency for Station 1 and 2: {scifi_ineff_1and2:.2e}")


if __name__ == "__main__":
    main()