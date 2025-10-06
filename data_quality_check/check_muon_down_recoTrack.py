import ROOT

def snapshot_fraction(
    input_file,
    input_tree="cbmsim",
    output_file="sample.root",
    output_tree="cbmsim",
    fraction=1000  # 1/N of the data
):
    """
    Snapshot 1/fraction of entries from a ROOT tree into a new file.

    Args:
        input_file (str): Path to input ROOT file.
        input_tree (str): Name of input TTree (default: 'cbmsim').
        output_file (str): Output ROOT file name.
        output_tree (str): Name for output TTree.
        fraction (int): Take 1/fraction of entries (default: 1/1000).
    """
    # Create the RDataFrame
    df = ROOT.RDataFrame(input_tree, input_file)

    # Count total entries
    n_total = df.Count().GetValue()
    n_sample = max(1, n_total // fraction)

    # Take only the first 1/N entries
    df_sample = df.Range(n_sample)

    # Snapshot to new file
    df_sample.Snapshot(output_tree, output_file)

    print(f"Snapshot saved to '{output_file}' with tree '{output_tree}'")
    print(f"Selected {n_sample} out of {n_total} entries (1/{fraction})")


def check_muon_down_recoTracks(digi_file):
    #read digi file into tree
    # check branch "reco_muonTracks" 
    
    import ROOT

def check_muon_down_recoTracks(digi_file):

    f = ROOT.TFile.Open(digi_file)
    if not f or f.IsZombie():
        print(f"Error: Cannot open file '{digi_file}'")
        return

    tree = f.Get("cbmsim")
    if not tree:
        print("Error: Tree 'cbmsim' not found.")
        f.Close()
        return

    track_type_map = {
        1:  "ST SciFi",
        11: "HT SciFi",
        3:  "ST DS",
        13: "HT DS",
    }
    n_events = tree.GetEntries()
    print(f"Total events in tree: {tree.GetEntries()}")
    print(f"Checking first {n_events} events...\n")

    for i in range(n_events):
        tree.GetEntry(i)
        recoTracks = tree.Reco_MuonTracks
        n_tracks = recoTracks.GetEntries()
        print(f"Event {i}: {n_tracks} reco muon track(s)")


        for j in range(n_tracks):
            track = recoTracks.At(j)
            # print(dir(track))
            ttype = track.getTrackType()
            ttype_str = track_type_map.get(ttype, f"Unknown ({ttype})") if ttype is not None else "Unknown"

            start = track.getStart()
            stop = track.getStop()

            sx, sy, sz = (start.X(), start.Y(), start.Z()) if start else (None, None, None)
            ex, ey, ez = (stop.X(),  stop.Y(),  stop.Z())  if stop  else (None, None, None)
            # Print basic info — adapt as needed
            print(f"  Track {j}:")
            print(f"    type       : {ttype_str}")
            print(f"    chi2/NDF: {track.getChi2Ndf()}")
            print(f"    start      : ({sx:.2f}, {sy:.2f}, {sz:.2f})" if sx is not None else "    start      : None")
            print(f"    stop       : ({ex:.2f}, {ey:.2f}, {ez:.2f})" if ex is not None else "    stop       : None")
        print("-" * 40)
        

    f.Close()


if __name__ == "__main__":
    digi_file = "/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_1.8_Bfield_4xstat/sndLHC.Ntuple-TGeant4-160urad_magfield_2022TCL6_muons_rock_2e8pr_Trks.root"
    # snapshot_fraction(
    #     input_file=digi_file,
    #     output_file="./test_data/sample_muon_down_recoTracks.root",
    #     fraction=10000
    # )
    
    check_muon_down_recoTracks(digi_file)
    