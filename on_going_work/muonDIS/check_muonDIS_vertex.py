import os
import pandas as pd
import ROOT
from tqdm import tqdm


def read_metadata(directory="/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated"):
    """Load all processed metadata CSVs into a dictionary."""
    metadata_dict = {}
    for file in os.listdir(directory):
        # read only MuonDIS
        if file.endswith(".csv") and file.startswith("MC_muonDIS"):
            key = file.replace(".csv", "")
            metadata_dict[key] = pd.read_csv(os.path.join(directory, file))
    return metadata_dict


def build_chain_from_metadata(df, tree_name="cbmsim", path_column="digi_path"):
    """Build a TChain from file paths stored in a metadata dataframe."""
    chain = ROOT.TChain(tree_name)

    n_added = 0
    for path in df[path_column].dropna():
        path = str(path).strip()
        if not path:
            continue
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            continue
        chain.Add(path)
        n_added += 1

    print(f"Added {n_added} files to TChain('{tree_name}')")
    print(f"Total entries in chain: {chain.GetEntries()}")
    return chain


pdg_db = ROOT.TDatabasePDG.Instance()

def pdg_to_name(pdg):
    """Convert PDG code to particle name safely."""
    part = pdg_db.GetParticle(int(pdg))
    if part:
        return part.GetName()
    return f"unknown({pdg})"

def classify_track(track):
    """Return primary/secondary label and mother id."""
    mother_id = int(track.GetMotherId())
    if mother_id < 0:
        return "primary", mother_id
    return "secondary", mother_id

def process(chain, max_events=10):
    n_entries = min(int(chain.GetEntries()), max_events)

    for i_event in tqdm(range(n_entries)):
        nb = chain.GetEntry(i_event)
        if nb <= 0:
            print(f"Event {i_event}: failed to read")
            continue

        mc_tracks = chain.MCTrack
        if mc_tracks is None:
            print(f"Event {i_event}: MCTrack is None")
            continue

        n_tracks = int(mc_tracks.GetEntriesFast())
        if n_tracks < 2:
            print(f"Event {i_event}: only {n_tracks} tracks")
            continue

        print(f"\n=== Event {i_event} | n_tracks = {n_tracks} ===")

        for i in range(n_tracks):
            tr = mc_tracks.At(i)
            if tr is None:
                print(f"Track {i}: null pointer")
                continue

            try:
                pdg = int(tr.GetPdgCode())
                if pdg == 0:
                    continue
                name = pdg_to_name(pdg)

                x = tr.GetStartX()
                y = tr.GetStartY()
                z = tr.GetStartZ()

                label, mother_id = classify_track(tr)

                print(
                    f"Track {i}: "
                    f"PDG={pdg} ({name}), "
                    f"type={label}, mother={mother_id}, "
                    f"start=({x:.3f}, {y:.3f}, {z:.3f})"
                )

            except Exception as e:
                print(f"Track {i}: failed to read: {e}")

            del tr

        del mc_tracks


def main():
    METADATA_dict = read_metadata()
    # print(METADATA_dict)
    muonDIS_metadata = METADATA_dict["MC_muonDIS_sii_metadata"]

    chain = build_chain_from_metadata(muonDIS_metadata, tree_name="cbmsim", path_column="digi_path")
    process(chain, max_events=5)

if __name__ == "__main__":
    main()