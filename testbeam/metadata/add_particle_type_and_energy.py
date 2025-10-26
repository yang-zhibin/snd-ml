import os
import csv
import ROOT
import pandas as pd
import glob
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import difflib


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


def get_feature_path(digi_path):
    """
    Construit le chemin de sortie du fichier feature à partir du digi_path.
    Exemple :
      /eos/experiment/sndlhc/.../sndsw_raw-0000.root
      → /eos/user/s/sfrankha/.../feature_sndsw_raw-0000.root
    """
    # Trouver la sous-partie à partir de 'sndlhc/'
    split_key = "sndlhc/"
    if split_key not in digi_path:
        raise ValueError(f"'{split_key}' not found in path: {digi_path}")

    relative_path = digi_path.split(split_key, 1)[1] 
    
    folder, filename = os.path.split(relative_path)
    
    feature_filename = f"feature_{filename}"
    
    feature_path = os.path.join("/eos/user/s/sfrankha", "sndlhc", folder, feature_filename)
    
    return feature_path


def get_hit_path(digi_path):
    """
    Construit le chemin de sortie du fichier feature à partir du digi_path.
    Exemple :
      /eos/experiment/sndlhc/.../sndsw_raw-0000.root
      → /eos/user/s/sfrankha/.../feature_sndsw_raw-0000.root
    """
    # Trouver la sous-partie à partir de 'sndlhc/'
    split_key = "sndlhc/"
    if split_key not in digi_path:
        raise ValueError(f"'{split_key}' not found in path: {digi_path}")

    relative_path = digi_path.split(split_key, 1)[1] 
    
    folder, filename = os.path.split(relative_path)

    hit_filename = f"hit_{filename}"

    hit_path = os.path.join("/eos/user/s/sfrankha", "sndlhc", folder, hit_filename)

    return hit_path


def get_real_particle_type_and_energy(df, particle_subfolder):
    if (particle_subfolder == "testbeam_June2023_H8" or particle_subfolder == "testbeam_24"):
        df["beam_energy"] = df["partition"].apply(lambda x: get_beam_energy(x, particle_subfolder))
        df["beam_type"] = df["partition"].apply(lambda x: get_beam_type(x, particle_subfolder))
    else:
        print("Subfolder not recognized.")
    
    return df

    
    
def extract_info(file_name):
    parts = file_name.replace("_metadata.csv", "").split("_")
    data_type = "_".join(parts[:2])  # First two parts as data_type
    subfolder = "_".join(parts[2:]) if len(parts) > 2 else ""  # Remaining as subfolder
    return data_type, subfolder


def main(args):
    csv_file = args.csv_input
    csv_name = os.path.basename(csv_file)
    
    file_exists = os.path.isfile(csv_file)
    if (not file_exists):
        print(f'{csv_file} does not exist, please first generate it.')
        return 0
    
    data_type, particle_subfolder = extract_info(csv_name)
    print(data_type, particle_subfolder)
    
    df = pd.read_csv(csv_file)
    
    df["feature_path"] = df["digi_path"].apply(get_feature_path)
    df["hit_path"] = df["digi_path"].apply(get_hit_path)
    
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
    
    df.to_csv(args.csv_output, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--forceRerun",dest="force_rerun",action="store_true",help="Force rerun")
    parser.add_argument("-i", "--csv_input", dest="csv_input", help="csv input", required=True)
    parser.add_argument("-o", "--csv_output", dest="csv_output", help="csv output", required=True)
    args = parser.parse_args()
    main(args)