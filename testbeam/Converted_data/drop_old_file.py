import os
import pandas as pd
import ROOT
from tqdm import tqdm

CSV = "/afs/cern.ch/user/z/zhibin/work/snd-ml/testbeam/metadata/updated/real_data_testbeam_24_metadata.csv"
THRESH = 1e4
TREE = "sndData"   # change if your tree name is different

df = pd.read_csv(CSV)

def entries(path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie(): 
        return None
    t = f.Get(TREE)
    n = int(t.GetEntries()) if t else None
    f.Close()
    return n

for _, r in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
    for col in ["feature_path", "hit_path"]:
        p = r[col]
        if not os.path.exists(p):
            continue
        n = entries(p)
        if n is not None and n > THRESH:
            os.remove(p)
            print("deleted", p, "entries", n)
