import os
import ROOT
from tqdm import tqdm
import argparse

ROOT.gROOT.SetBatch(True)


def build_chain(input_dir):
    """Build a TChain over all sndsw_raw-*.root files in the directory."""
    chain = ROOT.TChain("cbmsim")
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.startswith("sndsw_raw-") and f.endswith(".root")
    )
    if not files:
        raise FileNotFoundError(f"No sndsw_raw-*.root files in {input_dir}")

    for f in files[:100]:
        chain.Add(os.path.join(input_dir, f))

    return chain


def main(args):
    raw_data_dir = args.input_dir or "/eos/experiment/sndlhc/convertedData/physics/2024/run_241/run_008285/"

    chain = build_chain(raw_data_dir)
    n_entries = chain.GetEntries()
    if n_entries == 0:
        print("No entries in chain, nothing to do.")
        return

    print(f"Chain has {n_entries} entries from directory:\n  {raw_data_dir}")

    # Force branch addresses to be set before CloneTree
    chain.GetEntry(0)

    out_file = ROOT.TFile(args.output, "RECREATE")
    # Clone the structure only (0 entries)
    out_tree = chain.CloneTree(0)

    n_selected = 0
    max_events = args.max_events

    event_file_map = []   # list of tuples: (global_event_id, file_name)

    for i in tqdm(range(n_entries), total=n_entries):
        chain.GetEntry(i)

        n_scifi = chain.Digi_ScifiHits.GetEntries()

        veto_flag = False
        for aHit in chain.Digi_MuFilterHits:
            if aHit.isValid() and aHit.GetSystem() == 1:
                veto_flag = True
                break

        if veto_flag or n_scifi <= args.n_scifi:
            continue

        # Save mapping: global event index -> file name
        file_name = chain.GetFile().GetName()
        event_file_map.append((i, os.path.basename(file_name)))

        out_tree.Fill()
        n_selected += 1
        print(f'selected: {n_selected}')

        if args.max_events > 0 and n_selected >= args.max_events:
            break

    out_file.cd()
    out_tree.Write()
    out_file.Close()

    print(f"Done. Selected {n_selected} events.")
    print(f"Output written to {args.output}")
    
    map_path = args.output + ".event_map.txt"
    with open(map_path, "w") as f:
        for evt_id, fname in event_file_map:
            f.write(f"{evt_id}\t{fname}\n")

    print(f"Saved event→file mapping to: {map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter cbmsim events by veto and SciFi multiplicity.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/eos/experiment/sndlhc/convertedData/physics/2024/run_241/run_008285",
        help="Directory containing sndsw_raw-*.root files "
             "(default: hard-coded 2024 run_008285 path in script).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./test_data/filtered_data_2024_run8285_n=50.root",
        help="Output ROOT file name (default: filtered_cbmsim.root).",
    )
    parser.add_argument(
        "--n-scifi",
        type=int,
        default=200,
        help="Minimum number of SciFi digi hits required (exclusive, i.e. keep if n_scifi > N).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=50,
        help="Maximum number of selected events to write (default: 20). "
             "Use <=0 for no limit.",
    )

    args = parser.parse_args()
    main(args)
