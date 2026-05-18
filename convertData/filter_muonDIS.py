import argparse
import ROOT


def get_n_hits(branch_obj):
    """Return number of hits for ROOT TClonesArray / std::vector-like branches."""
    if branch_obj is None:
        return 0
    if hasattr(branch_obj, "GetEntries"):
        return branch_obj.GetEntries()
    if hasattr(branch_obj, "GetEntriesFast"):
        return branch_obj.GetEntriesFast()
    if hasattr(branch_obj, "size"):
        return branch_obj.size()
    return len(branch_obj)


def main(args):
    print("start processing filter muonDIS")

    in_file = ROOT.TFile.Open(args.digi_path, "READ")
    if not in_file or in_file.IsZombie():
        raise RuntimeError(f"Cannot open input file: {args.digi_path}")

    in_tree = in_file.Get("cbmsim")
    if not in_tree:
        raise RuntimeError("Cannot find tree 'cbmsim' in input file")

    out_file = ROOT.TFile.Open(args.out_path, "RECREATE")

    # Clone full tree structure, but no entries yet
    out_tree = in_tree.CloneTree(0)

    n_total = in_tree.GetEntries()
    n_saved = 0

    for i, event in enumerate(in_tree):
        # mu_hits = getattr(event, "Digi_MuFilterHits", None)
        scifi_hits = getattr(event, "Digi_ScifiHits", None)

        # n_mu = get_n_hits(mu_hits)
        n_scifi = get_n_hits(scifi_hits)

        if n_scifi > 5:
            out_tree.Fill()
            n_saved += 1

        if i % 10000 == 0:
            print(f"processed {i}/{n_total}")

    out_file.cd()
    out_tree.Write()

    # Save original number of entries as metadata
    h_entries = ROOT.TH1I("original_entries", "Original number of entries", 1, 0, 1)
    h_entries.SetBinContent(1, n_total)
    h_entries.Write()

    out_file.Close()
    in_file.Close()

    eff = n_saved / n_total if n_total > 0 else 0.0

    print("finished processing filter muonDIS")
    print(f"original entries: {n_total}")
    print(f"saved entries: {n_saved}")
    print(f"filter efficiency: {eff:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)

    args = parser.parse_args()
    main(args)