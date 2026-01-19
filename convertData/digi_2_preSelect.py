import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import array
from tqdm import tqdm


def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    snd_geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(snd_geo.modules['Scifi'])
    lsOfGlobals.Add(snd_geo.modules['MuFilter'])
    return snd_geo

def open_root_file(file_path, mode='read'):
    """Open and return a ROOT file and its primary tree."""
    file = ROOT.TFile(file_path, mode)
    tree = file.Get('cbmsim')
    return file, tree

def create_output_file(path, mode):
    """Create and return a new ROOT file and a new tree for output."""
    out_file = ROOT.TFile(path, mode)
    new_tree = ROOT.TTree('sndData', 'converted cbmsim tree')
    return out_file, new_tree

def process_vetoHitTime(all_hits, branch_vars):
    # Filter veto hits (detType == 1)
    veto_hits = [h for h in all_hits if h["detType"] == 1]

    # Compute earliest and latest per station
    per_station = {}
    for s in (1, 2, 3):
        times = [h["hit_time"] for h in veto_hits if h["station"] == s]
        per_station[s] = {
            "earliest": min(times) if times else -1,  # use -1 if no hit
            "latest":   max(times) if times else -1,
        }

    # Compute overall earliest/latest
    all_times = [h["hit_time"] for h in veto_hits]
    overall_earliest = min(all_times) if all_times else -1
    overall_latest   = max(all_times) if all_times else -1

    # Fill the branch variables
    branch_vars["vetoHitTime_earlist"][0] = overall_earliest
    branch_vars["vetoHitTime_latest"][0]  = overall_latest

    for s in (1, 2, 3):
        branch_vars[f"vetoHitTime_earlist_veto{s}"][0] = per_station[s]["earliest"]
        branch_vars[f"vetoHitTime_latest_veto{s}"][0]  = per_station[s]["latest"]
    
    

def process_hits(event, snd_geo, branch_vars):
    """Process all hits in the event and update hits array and averages."""
    # Scifi = snd_geo.modules['Scifi']
    # MuFilter = snd_geo.modules['MuFilter']
    # A, B = ROOT.TVector3(), ROOT.TVector3()


    scifi_counts = [0] * 5  # scifi1 to scifi5
    veto_counts = [0] * 3 # veto1 to veto3
    ds_counts = [0] * 4  # ds1 to ds4
    us_counts = [0] * 5  # us1 to us5

    nVetoPlanes = snd_geo.snd_geo.MuFilter.NVetoPlanes

    # Process SciFi hits
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        station = detID // 1000000
        if 1 <= station <= 5:
            scifi_counts[station - 1] += 1


    # Process MuFilter hits
    all_hits = []
    for aHit in event.Digi_MuFilterHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        detType = aHit.GetSystem()
        station = (detID // 1000) % 10
        
        if nVetoPlanes==2 and detType==1:
            station = station+1
        
        hit_time = aHit.GetTime()
        
        all_hits.append({
            "detType": detType,
            "station": station+1,
            "isVertical": aHit.isVertical(),
            "hit_time": hit_time
        })

        if detType == 1:
            veto_counts[station] += 1
            
        elif detType == 3 and 0 <= station <= 3:
            ds_counts[station ] += 1
        elif detType == 2 and 0 <= station <= 4:
            us_counts[station ] += 1

    process_vetoHitTime(all_hits, branch_vars)
    
    return veto_counts, scifi_counts, us_counts, ds_counts


import SndlhcMuonReco
import SndlhcTracking



def main(args):
    print("start processing digi to preSelection")
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)
    
    out_file, new_tree = create_output_file(args.out_path, args.mode)

    branches = [
        ("runId", 'i'), ("eventId", 'i'), ("isMC", 'i'), ("eventIndex", 'i'),("pdgCode", 'i'), ("energy", 'd'),
        ("At_least_1_non_veto_hit", 'i'),
        
        ("count_veto1", 'i'), ("count_veto2", 'i'), ("count_veto3", 'i'),  ("count_veto", 'i'), 
        ("count_scifi1", 'i'), ("count_scifi2", 'i'), ("count_scifi3", 'i'),("count_scifi4", 'i'), ("count_scifi5", 'i'), ("count_scifi", 'i'),
        ("count_us1", 'i'), ("count_us2", 'i'), ("count_us3", 'i'),("count_us4", 'i'), ("count_us5", 'i'), ("count_us", 'i'),
        ("count_ds1", 'i'), ("count_ds2", 'i'), ("count_ds3", 'i'), ("count_ds4", 'i'),("count_ds", 'i'),
        
        ("cut_H_if_DS_hits_must_all_US_hits",'i'),
        ("cut_G_has_consecutive_scifi_hits",'i'),
        ("preSelect", 'i'),
        ("preSelect_vetoFree", 'i'),
        ("preSelect_vetoTagged", 'i'),
        
        ("vetoHitTime_earlist", 'd'), ("vetoHitTime_latest", 'd'),
        ("vetoHitTime_earlist_veto1", 'd'), ("vetoHitTime_latest_veto1", 'd'),
        ("vetoHitTime_earlist_veto2", 'd'), ("vetoHitTime_latest_veto2", 'd'),
        ("vetoHitTime_earlist_veto3", 'd'), ("vetoHitTime_latest_veto3", 'd'),
    ]

    # Dictionary to hold branch variables
    branch_vars = {}

    # Create branches dynamically
    for name, dtype in branches:
        branch_vars[name] = array.array(dtype, [-999])  # Initialize the array
        new_tree.Branch(name, branch_vars[name], f"{name}/{dtype.upper()}")
    
    # # --- FairRoot infrastructure ---
    
    # run = ROOT.FairRunAna()
    # #avoiding some error messages
    # xrdb = ROOT.FairRuntimeDb.instance()
    # xrdb.getContainer("FairBaseParSet").setStatic()
    # xrdb.getContainer("FairGeoParSet").setStatic()

    # source = ROOT.FairFileSource(args.digi_path)
    # run.SetSource(source)

    # sink  = ROOT.FairRootFileSink(f"{os.path.dirname(args.out_path)}/dummy.root")
    # run.SetSink(sink)
    # OT = sink.GetOutTree()

    # # --- Hough tracking tasks ---
    # HT_Sf    = SndlhcMuonReco.MuonReco()
    # HT_DS    = SndlhcMuonReco.MuonReco()

    # parameter_file = os.environ['SNDSW_ROOT']+"/python/TrackingParams.xml"
    # for ht in [HT_Sf, HT_DS]:
    #     ht.SetParFile(parameter_file)
    #     ht.SetHoughSpaceFormat("linearSlopeIntercept")
    #     ht.ForceGenfitTrackFormat()
    #     run.AddTask(ht)

    # HT_Sf.SetTrackingCase("passing_mu_Sf")
    # HT_DS.SetTrackingCase("passing_mu_DS")

    # # --- Simple straight-line tracking ---
    # trackTask = SndlhcTracking.Tracking()
    # trackTask.SetName('simpleTracking')
    # run.AddTask(trackTask)

    # # --- Initialise tasks ---
    # run.Init()

    # # Access task outputs
    # ioman = ROOT.FairRootManager.Instance()
    # OT    = sink.GetOutTree()       
        
        
    # Process each event
    scifi_count_threshold = 200
    
    print('------debug')
    
    if "muon" in args.type:
        scifi_count_threshold = 5
    print(f"Data type: {args.type}, scifi_count_threshold:{scifi_count_threshold}")
    for i_event, event in tqdm(enumerate(raw_tree), total=raw_tree.GetEntries()):
        #if i_event % 10000 == 0:
        #    print(f"processed {i_event} events")
        
        # OT.Reco_MuonTracks = ROOT.TObjArray(10)
        # # --- Load the event for FairTasks ---
        # source.GetInTree().GetEvent(i_event)

        # # ----------------------
        # # 1) HOUGH RECONSTRUCTION
        # # ----------------------
        # # Clear previous tracks
        # for ht in [HT_Sf, HT_DS]:
        #     ht.kalman_tracks.Delete()

        # # Example: run only SciFi-based Hough tracking
        # HT_Sf.Exec(0)
        # HT_DS.Exec(0)
        

        # # Collect Hough tracks
        # hough_tracks = []
        # for ht in [HT_Sf, HT_DS]:
        #     for trk in ht.kalman_tracks:
        #         hough_tracks.append(trk)


        # # ----------------------
        # # 2) SIMPLE TRACKING
        # # ----------------------
        # trackTask.fittedTracks.Delete()

        # # Available modes:
        # #   "Scifi"
        # #   "DS"
        # #   "ScifiDS"
        # trackTask.ExecuteTask("ScifiDS")

        # simple_tracks = []
        # for trk in trackTask.fittedTracks:
        #     print(trk)
        #     simple_tracks.append(trk)


        # # ----------------------
        # # 3) Now use your tracks
        # # ----------------------
        # # hough_tracks : list of genfit::Track from Hough reco
        # # simple_tracks : list of genfit::Track from simple reco

        # print(f"Event {i_event}: Hough={len(hough_tracks)}, Simple={len(simple_tracks)}")

        # # Example: extract fitted state
        # for trk in hough_tracks:
        #     print(trk.__repr__())
        #     print(dir(trk))
        #     state = trk.getFittedState()
        #     mom   = state.getMom()
        #     pos   = state.getPos()
        #     mom.Print()
        #     pos.Print()
        
        branch_vars["eventIndex"][0] = i_event
        branch_vars["runId"][0] = event.EventHeader.GetRunId()

        if ('MC' in  args.type):
            branch_vars["isMC"][0] = 1
            #print(dir(event.EventHeader))
            
            try:
                branch_vars["eventId"][0] = event.EventHeader.GetEventNumber()
            except Exception:
                branch_vars["eventId"][0] = event.EventHeader.GetMCEntryNumber()
            # Particle codes and initial position
            
            event_pdg0 = raw_tree.MCTrack[0].GetPdgCode()
            event_pdg1 = raw_tree.MCTrack[1].GetPdgCode()
            
            branch_vars["energy"][0] = raw_tree.MCTrack[0].GetEnergy()
            

            neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
            if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
                branch_vars["pdgCode"][0] = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
            else:
                branch_vars["pdgCode"][0] = event_pdg0


        elif('real' in  args.type):
            branch_vars["isMC"][0] = 0
            branch_vars["pdgCode"][0] = 0
            branch_vars["eventId"][0] = event.EventHeader.GetEventNumber()
        
        veto_counts, scifi_counts, us_counts, ds_counts = process_hits(event, snd_geo, branch_vars)
        
        # Fill Veto counts
        for i in range(3):
            branch_vars[f"count_veto{i+1}"][0] = veto_counts[i]
        branch_vars["count_veto"][0] = sum(veto_counts)

        # Fill SciFi counts
        for i in range(5):
            branch_vars[f"count_scifi{i+1}"][0] = scifi_counts[i]
        branch_vars["count_scifi"][0] = sum(scifi_counts)

        # Fill Upstream (US) counts
        for i in range(5):
            branch_vars[f"count_us{i+1}"][0] = us_counts[i]
        branch_vars["count_us"][0] = sum(us_counts)

        # Fill Downstream (DS) counts
        for i in range(4):
            branch_vars[f"count_ds{i+1}"][0] = ds_counts[i]
        branch_vars["count_ds"][0] = sum(ds_counts)


        
        if ((sum(scifi_counts)+sum(us_counts)+sum(ds_counts)) > 0):
            branch_vars["At_least_1_non_veto_hit"][0] = 1
        else:
            branch_vars["At_least_1_non_veto_hit"][0] = 0
                
        has_consecutive_hits = any(scifi_counts[i] and scifi_counts[i + 1] for i in range(len(scifi_counts) - 1))
        if has_consecutive_hits:
            branch_vars["cut_G_has_consecutive_scifi_hits"][0] = 1
        else:
            branch_vars["cut_G_has_consecutive_scifi_hits"][0] = 0
            
        
        
        
        if (sum(ds_counts)) > 0 and not (
            all(us_counts[i] > 0 for i in range(len(us_counts)))
        ):
            branch_vars["cut_H_if_DS_hits_must_all_US_hits"][0] = 0
        else:
            branch_vars["cut_H_if_DS_hits_must_all_US_hits"][0] = 1
        
        if (branch_vars["count_scifi"][0]>scifi_count_threshold and branch_vars["count_veto"][0] == 0):
            branch_vars["preSelect_vetoFree"][0] = 1
        else:
            branch_vars["preSelect_vetoFree"][0] = 0
        
        if (branch_vars["count_scifi"][0]>scifi_count_threshold and branch_vars["count_veto"][0] > 0):
            branch_vars["preSelect_vetoTagged"][0] = 1
        else:
            branch_vars["preSelect_vetoTagged"][0] = 0
        
        if (branch_vars["count_scifi"][0]>scifi_count_threshold):
            branch_vars["preSelect"][0] = 1
        else:
            branch_vars["preSelect"][0] = 0
            
        
        #for debug
        branch_vars["preSelect"][0] = 1
        new_tree.Fill()
        # if i_event>30:
        #     break

    # Finalize the output file
    new_tree.Write()
    out_file.Close()
    print("finish processing digi to feature")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)
    parser.add_argument("-mo", "--mode", dest="mode", help="open root file mode", default='RECREATE')
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)

    args = parser.parse_args()

    main(args)