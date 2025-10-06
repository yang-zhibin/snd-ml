import ROOT

ROOT.gROOT.ProcessLine(".L /afs/cern.ch/user/z/zhibin/work/snd-ml/convertData/EventClass.h+")
filename = "/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/vetoTagged_feature_MC_neutrino_volTarget_100fb-1_0.root"  
f = ROOT.TFile.Open(filename)
tree_name = 'sndData'
tree = f.Get(tree_name)

#Prepare TClonesArray to read vetoHits branch
vetoHits = ROOT.TClonesArray("VetoHit")
tree.SetBranchAddress("vetoHits", vetoHits)

# Event loop
for evt in range(tree.GetEntries()):
    tree.GetEntry(evt)
    n_hits = vetoHits.GetEntriesFast()
    print(f"\n=== Event {evt} | nVetoHits = {n_hits} ===")

    for i in range(n_hits):
        vh = vetoHits.At(i)  # get VetoHit
        print(f"  VetoHit #{i}:")
        print(f"    time: {vh.hit_time:.2f} ns")
        print(f"    QDC:  {vh.qdc:.1f}")
        print(f"    plane: {vh.veto_plane}")
        print(f"    energy_loss: {vh.energy_loss:.4f} GeV")

        n_scifi = vh.scifiPoints.size()
        print(f"    nScifiPoints = {n_scifi}")
        for j in range(n_scifi):
            sp = vh.scifiPoints[j]
            print(f"      [{j}] pdg: {sp.pdg}, dE: {sp.energy_loss:.6f}, pos: ({sp.x:.2f}, {sp.y:.2f}, {sp.z:.2f})")