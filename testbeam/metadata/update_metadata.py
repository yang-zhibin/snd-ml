import os
import csv
# import ROOT
import pandas as pd
import glob
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import difflib
from math import ceil
import ROOT
import re



pdg_to_particle = {
    "11": "e-",
    "-11": "e+",
    "211": "pi+",
    "-211": "pi-",
    "13": "mu-",
    "-13": "mu+",
    "22": "photon"
}



def get_MC_particle_type_and_energy(df, particle_subfolder):
    if particle_subfolder == "testbeam2023":
        df["beam_energy"] = df["partition"].str.split("_").str[0]
        df["beam_type"] = df["partition"].str.split("_").str[1].map(pdg_to_particle).fillna("unknown")
    elif particle_subfolder == "testbeam2024":
        df["beam_energy"] = df["subfolder"].str.split("_").str[0]
        df["beam_type"] = df["subfolder"].str.split("_").str[1].map(pdg_to_particle).fillna("unknown")
    else:
        print("Subfolder not recognized.")
    
    return df


def get_beam_type(run, particle_subfolder):
    run_number = int(run.split("_")[1])
    if particle_subfolder == "testbeam_June2023_H8":
        if ((100517 <= run_number <= 100561) and run_number != 100519) or (run_number == 100579) or (100622 <= run_number <= 100636) or (run_number == 100677) or (100659 <= run_number <= 100674):
            return "pi+"
        elif (100637 <= run_number <= 100639) or (100641 <= run_number <= 100651) or (100653 <= run_number <= 100656):
            return "pi-"
        elif (run_number == 100519) or (run_number == 100568) or (run_number == 100571) or (run_number == 100605) or (run_number == 100640) or (run_number == 100652) or (run_number == 100657) or (run_number == 100679) or (run_number == 100675) or (run_number == 100676) or (run_number == 100678):
            return "mu-"
    elif particle_subfolder == "testbeam_24":
        if (run_number == 100890) or (run_number == 100896) or (100902 <= run_number <= 100933):
            return "e-"
        elif (run_number == 100891) or (run_number == 100894) or (100934 <= run_number <= 100945):
            return "pi-"
        elif (100946 <= run_number <= 100985):
            return "pi+"
        elif run_number == 100892:
            return "mu-"
    return "no type"


def get_beam_energy(run, particle_subfolder):
    run_number = int(run.split("_")[1])
    if particle_subfolder == "testbeam_June2023_H8":
        if ((100517 <= run_number <= 100537) and run_number != 100519) or (100543 <= run_number <= 100555) or (run_number == 100579) or (run_number == 100627) or (100624 <= run_number <= 100625) or (100634 <= run_number <= 100636) or (100659 <= run_number <= 100660) or (100667 <= run_number <= 100672):
            return "180GeV"
        elif (100538 <= run_number <= 100542) or (run_number == 100623) or (run_number == 100626) or (run_number == 100633) or (run_number == 100661) or (run_number == 100662) or (run_number == 100665) or (run_number == 100666) or (run_number == 100673) or (run_number == 100674):
            return "140GeV"
        elif (run_number == 100558):
            return "120GeV"
        elif (100628 <= run_number <= 100632) or (run_number == 100561) or (run_number == 100622) or (run_number == 100663) or (run_number == 100664) or (run_number == 100677):
            return "100GeV"
        elif (run_number == 100638) or (run_number == 100639) or (100641 <= run_number <= 100645) or (run_number == 100650)  or (run_number == 100651)  or (run_number == 100653):
            return "300GeV"
        elif (100654 <= run_number <= 100656) or (100646 <= run_number <= 100649) or (run_number == 100637):
            return "240GeV"
        elif (run_number == 100675) or (run_number == 100676) or (run_number == 100678):
            return "160GeV"
    elif particle_subfolder == "testbeam_24":
        if (run_number == 100890) or (run_number == 100896) or (100911 <= run_number <= 100917):
            return "100GeV"
        elif (run_number == 100902) or (100918 <= run_number <= 100924):
            return "200GeV"
        elif (run_number == 100904) or (run_number == 100929) or (run_number == 100931):
            return "250GeV"
        elif (100925 <= run_number <= 100927) or (100905 <= run_number <= 100910):
            return "300GeV"
        elif (run_number == 100932) or (run_number == 100933):
            return "50GeV"
        elif (100934 <= run_number <= 100945) or (run_number == 100928) or (run_number == 100892):
            return "150GeV"
        elif (100946 <= run_number <= 100985) or (run_number == 100891) or (run_number == 100894):
            return "180GeV"
    return "no energy"



def get_new_path(digi_path, pre_fix, file_type, eos_path, year):
    
    # Trouver la sous-partie à partir de 'sndlhc/'
    split_key = "sndlhc/"
    if split_key not in digi_path:
        raise ValueError(f"'{split_key}' not found in path: {digi_path}")

    relative_path = digi_path.split(split_key, 1)[1] 
    
    folder, filename = os.path.split(relative_path)
    
    #replace the file type with file_type
    name, _ = os.path.splitext(filename)
    
    # Créer le nouveau nom de fichier avec le préfixe et la nouvelle extension
    new_filename = f"{pre_fix}_{name}{file_type}"
    
    new_path = os.path.join(eos_path, "TestBeam", year, folder, new_filename)
    
    return new_path




def get_real_particle_type_and_energy(df, particle_subfolder):
    if (particle_subfolder == "testbeam_June2023_H8" or particle_subfolder == "testbeam_24"):
        df["beam_energy"] = df["partition"].apply(lambda x: get_beam_energy(x, particle_subfolder))
        df["beam_type"] = df["partition"].apply(lambda x: get_beam_type(x, particle_subfolder))
    else:
        print("Subfolder not recognized.")
    
    return df

def split_root_file_ranges(input_path, out_dir, partition,
                           tree_name="cbmsim", chunk_size=2000,
                           n_entries_override=None):

    f_in = ROOT.TFile.Open(input_path)
    tree = f_in.Get(tree_name)
    n_entries = int(n_entries_override) if n_entries_override else int(tree.GetEntries())

    base = os.path.splitext(os.path.basename(input_path))[0]
    n_chunks = int(ceil(n_entries / chunk_size)) if n_entries > 0 else 0

    outputs = []
    file_entries = []

    final_dir = os.path.join(out_dir, str(partition))
    os.makedirs(final_dir, exist_ok=True)

    for i in range(n_chunks):
        start = i * chunk_size
        n_take = min(chunk_size, n_entries - start)

        out_name = f"{base}_{i+1}.root"
        out_path = os.path.join(final_dir, out_name)

        f_out = ROOT.TFile(out_path, "RECREATE")
        new_tree = tree.CopyTree("", "", n_take, start)
        new_tree.Write()
        f_out.Close()

        outputs.append(out_path)
        file_entries.append(n_take)

    f_in.Close()
    return outputs, file_entries



def split_raw_and_digi_df(
    df,
    eos_raw_dir,
    eos_digi_dir,
    tree_name="cbmsim",
    chunk_size=2000,
):
    # preserve originals
    df = df.copy()
    df["original_raw_path"] = df["raw_path"]
    df["original_digi_path"] = df["digi_path"]

    new_rows = []

    for idx, row in df.iterrows():
        raw_in = row["original_raw_path"]
        digi_in = row["original_digi_path"]
        partition = row["partition"]

        # Open both to determine aligned number of entries (min of both)
        f_raw = ROOT.TFile.Open(raw_in)
        f_digi = ROOT.TFile.Open(digi_in)
        if not f_raw or f_raw.IsZombie():
            raise RuntimeError(f"Could not open raw: {raw_in}")
        if not f_digi or f_digi.IsZombie():
            raise RuntimeError(f"Could not open digi: {digi_in}")

        t_raw = f_raw.Get(tree_name)
        t_digi = f_digi.Get(tree_name)
        if not t_raw:
            raise RuntimeError(f"TTree '{tree_name}' not found in raw: {raw_in}")
        if not t_digi:
            raise RuntimeError(f"TTree '{tree_name}' not found in digi: {digi_in}")

        n_raw = int(t_raw.GetEntries())
        n_digi = int(t_digi.GetEntries())
        n_use = min(n_raw, n_digi)  # keep event index alignment
        f_raw.Close()
        f_digi.Close()

        if n_use == 0:
            # nothing to split; leave paths as-is (or handle differently if you prefer)
            continue

        raw_outs, raw_entries  = split_root_file_ranges(raw_in,  eos_raw_dir, partition, tree_name, chunk_size, n_entries_override=n_use)
        digi_outs, digi_entries  = split_root_file_ranges(digi_in, eos_digi_dir,partition, tree_name, chunk_size, n_entries_override=n_use)

        if len(raw_outs) != len(digi_outs):
            # Shouldn't happen given same n_use & chunk_size, but guard anyway
            m = min(len(raw_outs), len(digi_outs))
            raw_outs, digi_outs = raw_outs[:m], digi_outs[:m]

        # Replace current row with chunk 1 paths
        df.at[idx, "raw_path"] = raw_outs[0]
        df.at[idx, "digi_path"] = digi_outs[0]
        df.at[idx, "n_event"] = digi_entries[0]


        # Add extra rows for chunk 2..N
        for k in range(1, len(raw_outs)):
            nr = row.copy()
            nr["raw_path"] = raw_outs[k]
            nr["digi_path"] = digi_outs[k]
            nr["n_event"] = digi_entries[k]
            new_rows.append(nr)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df
    
    
def extract_info(file_name):
    parts = file_name.replace("_metadata.csv", "").split("_")
    data_type = "_".join(parts[:2])  # First two parts as data_type
    subfolder = "_".join(parts[2:]) if len(parts) > 2 else ""  # Remaining as subfolder
    return data_type, subfolder


def to_raw(digi_path: str) -> str:
    raw_folder = "/eos/experiment/sndlhc/raw_data/testbeam_24"
    pat = re.compile(r""".*/run_(10\d+)/sndsw_raw-([^.\/]+)\.root$""")
    m = pat.match(digi_path)
    if not m:
        raise ValueError(f"Unexpected digi_path format: {digi_path}")
    run_id, file_id = m.group(1), m.group(2)
    return f"{raw_folder}/run_{run_id}/data_{file_id}.root"

def main(args):

    
    csv_file = args.csv_input
    csv_name = os.path.basename(csv_file)
    
    if "2024" in csv_name or  "real_data_testbeam_24" in csv_name :
        year = "2024"
    elif "2023" in csv_name:
        year = "2023"
    
    
    file_exists = os.path.isfile(csv_file)
    if (not file_exists):
        print(f'{csv_file} does not exist, please first generate it.')
        return 0
    
    data_type, particle_subfolder = extract_info(csv_name)
    print(data_type, particle_subfolder)
    
    df = pd.read_csv(csv_file)
    
    # drop rows with n_event == 0 (if column exists)
    if "n_event" in df.columns:
        df = df[df["n_event"] != 0].reset_index(drop=True)
    else:
        print("Warning: 'n_event' column not found; no rows dropped.")
    
    if data_type == "MC_data":
        df = get_MC_particle_type_and_energy(df, particle_subfolder)
    elif data_type == "real_data":
        df = get_real_particle_type_and_energy(df, particle_subfolder)
    else:
        print("Unexpected data type, stopping.")
        return 1
    
    if not (('beam_energy' in df.columns) and ('beam_type' in df.columns)):
        print("Update failed")
        return
    
    
    # process for 2024
    if year == "2024":
        # drop subfolder =='180GeV_211_Fe'
        if "subfolder" in df.columns:
            df = df[df["subfolder"] != "180GeV_211_Fe"].reset_index(drop=True)
        else:
            print("Warning: 'subfolder' column not found; no rows dropped.")
        
        #in the df, digi_path column is like ...digi__folder/run_10****/sndsw_raw-xxxx.root, i want to get the raw_path {raw_folder}/run_10****/data_xxxx.root
        
        if data_type == "real_data":
            df["raw_path"] = df["digi_path"].apply(to_raw)
        
        df["feature_path"] = df["digi_path"].apply(get_new_path, pre_fix="feature", file_type=".root", eos_path=args.eos_path, year = year)
        df["hit_path"] = df["digi_path"].apply(get_new_path, pre_fix="hit", file_type=".root", eos_path=args.eos_path, year=year)
        df["pt_hit_path"] = df["digi_path"].apply(get_new_path, pre_fix="pt_hit", file_type=".pt.gz", eos_path=args.eos_path, year=year)
        
        
        versions = ["v2", "v3", "v4", "v5","v6", "v7","v6_2", "v7_2", "v8"]
        for v in versions:
            df[f"prediction_testbeam_{year}_GravNet_{v}_output_path"] = (
                df["digi_path"].apply(
                    get_new_path,
                    pre_fix=f"prediction_testbeam_{year}_GravNet_{v}_output",
                    file_type=".root",
                    eos_path=args.eos_path,
                    year=year
                )
            )
            
            
        corrupted_csv_name = [
            'log_outputs.csv',
            'first_peak_150GeV_300GeV.csv'
        ]

        # directory of the input CSV (same dir as corrupted CSVs)
        corrupted_csv_dir = os.path.dirname(args.csv_input)

        # collect all corrupted digi paths
        corrupted_digi_paths = set()

        for name in corrupted_csv_name:
            csv_path = os.path.join(corrupted_csv_dir, name)
            if not os.path.exists(csv_path):
                print(f"[WARN] Corrupted CSV not found: {csv_path}")
                continue

            try:
                tmp = pd.read_csv(csv_path)
                if 'digi_path' in tmp.columns:
                    corrupted_digi_paths.update(
                        tmp['digi_path'].dropna().astype(str)
                    )
                else:
                    print(f"[WARN] 'digi_path' column missing in {csv_path}")
            except Exception as e:
                print(f"[WARN] Failed to read {csv_path}: {e}")

        # ensure consistent dtype before comparison
        df['digi_path'] = df['digi_path'].astype(str)
        # drop rows with corrupted digi paths
        df = df[~df['digi_path'].isin(corrupted_digi_paths)]
        
        df = df[~df['beam_type'].isin(['mu-', 'no type'])]
        
        df.to_csv(args.csv_output, index=False)
    
    #process for 2023
    elif year == "2023":
        #devide digi to multiple file
        df["original_raw_path"] = df["raw_path"]
        df["original_n_event"] = df["n_event"]
        
        CHUNK_SIZE   = 2000          
        TREE_NAME    = "cbmsim"      
        df2 = split_raw_and_digi_df(
            df,
            eos_raw_dir=f"{args.eos_path}/TestBeam/{year}/",
            eos_digi_dir=f"{args.eos_path}/TestBeam/{year}/",
            tree_name=TREE_NAME,
            chunk_size=CHUNK_SIZE,
        )
        
        df2.to_csv(args.csv_output, index=False)
        def make_prefixed_path(path, prefix, file_type=".root"):
            directory = os.path.dirname(path)
            base = os.path.splitext(os.path.basename(path))[0]
            return os.path.join(directory, f"{prefix}_{base}{file_type}")
        # raw_path -> new_digi_path, new_feature_path, new_hit_path
        # digi_path ->feature_path, hit_path, 
        # From raw_path
        df2["new_digi_path"]    = df2["raw_path"].apply(lambda p: make_prefixed_path(p, "new_digi"))
        df2["new_feature_path"] = df2["raw_path"].apply(lambda p: make_prefixed_path(p, "new_feature"))
        df2["new_hit_path"]     = df2["raw_path"].apply(lambda p: make_prefixed_path(p, "new_hit"))

        # From digi_path
        df2["feature_path"] = df2["digi_path"].apply(lambda p: make_prefixed_path(p, "feature"))
        df2["hit_path"]     = df2["digi_path"].apply(lambda p: make_prefixed_path(p, "hit"))
        df2.to_csv(args.csv_output, index=False)
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--forceRerun",dest="force_rerun",action="store_true",help="Force rerun")
    parser.add_argument("-i", "--csv_input", dest="csv_input", help="csv input", required=True)
    parser.add_argument("-o", "--csv_output", dest="csv_output", help="csv output", required=True)
    parser.add_argument("-e", "--eos_path", dest="eos_path", help="eos path",required=True)
    args = parser.parse_args()
    main(args)