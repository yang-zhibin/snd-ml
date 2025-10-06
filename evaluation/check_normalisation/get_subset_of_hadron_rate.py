import ROOT

tree_name = "cbmsim"
in_path   = "/eos/user/c/cvilela/neutron_kaon_nue_stage1_noprescale.root"
out_path  = "neutron_kaon_nue_stage1_noprescale.root"

df = ROOT.RDataFrame(tree_name, in_path)

n_entries = df.Count().GetValue()
n_subset  = int(n_entries * 0.01)

# Select the first 1% of entries
df_subset = df.Range(0, n_subset)

# Snapshot to new file
df_subset.Snapshot(tree_name, out_path)