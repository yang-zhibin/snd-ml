import ROOT, os
import SndlhcGeo
import SndlhcMuonReco

digi_path = '/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_20240126_digCPP.root'
geo_path  = '/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/geofile_full.Genie-TGeant4.root'

par_file = os.environ['SNDSW_ROOT'] + "/python/TrackingParams.xml"

geo = SndlhcGeo.GeoInterface(geo_path)
ROOT.gROOT.GetListOfGlobals().Add(geo.modules['Scifi'])
ROOT.gROOT.GetListOfGlobals().Add(geo.modules['MuFilter'])

f = ROOT.TFile.Open(digi_path)
tree_name = "cbmsim" if f.Get("cbmsim") else "rawConv"
tree = f.Get(tree_name)

ioman = ROOT.FairRootManager.Instance()
ioman.SetTreeName(tree_name)
run = ROOT.FairRunAna()
run.SetSource(ROOT.FairFileSource(f))
run.SetSink(ROOT.FairRootFileSink(ROOT.TMemFile("dummy.root", "RECREATE")))
run.SetEventHeaderPersistence(False)

muon_reco = SndlhcMuonReco.MuonReco()
muon_reco.SetParFile(par_file)
muon_reco.SetTrackingCase("passing_mu_Sf")  # adjust based on your XML!
muon_reco.SetHoughSpaceFormat("linearSlopeIntercept")
muon_reco.SetStandalone()
run.AddTask(muon_reco)

run.Init()
muon_reco.SetScaleFactor(1)

tree = f.Get(tree_name)
for evt in range(10):
    tree.GetEntry(evt)
    muon_reco.Exec("")
    tracks = muon_reco.kalman_tracks
    for i in range(tracks.GetEntries()):
        tr = tracks.At(i)
        if not tr or not tr.getFitStatus().isFitConverged():
            continue
        state = tr.getFittedState()
        print(f"[event {evt}] Track {i} pos:", state.getPos(), "mom:", state.getMom())

