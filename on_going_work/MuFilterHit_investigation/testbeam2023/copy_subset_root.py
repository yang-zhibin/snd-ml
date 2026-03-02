import ROOT
from tqdm import tqdm

infile  = "/eos/experiment/sndlhc/MonteCarlo/testbeam2023/100GeV_211/sndLHC.PG_211-TGeant4.root"
outfile = "./sndLHC.PG_211-TGeant4_1kentries.root"
treename = "cbmsim"   # change this

fin = ROOT.TFile.Open(infile)
tin = fin.Get(treename)

fout = ROOT.TFile(outfile, "RECREATE")

# Clone tree structure only (no data yet)
tout = tin.CloneTree(0)

n = min(1000, tin.GetEntries())  # safety if tree has <1000 entries

for i in tqdm(range(n), desc="Copying events", unit="evt"):
    tin.GetEntry(i)
    tout.Fill()

tout.Write()
fout.Close()
fin.Close()

print(f"Copied {n} events to {outfile}")