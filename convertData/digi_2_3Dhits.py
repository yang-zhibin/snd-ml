import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
from array import array
import time
import math
import numpy as np
import gzip
import pickle

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

def check_ecut_mc_hits(v_ecut_mc_points, h_ecut_mc_points, dmax=1):
    for v_point in v_ecut_mc_points:
        for h_point in h_ecut_mc_points:
            dx = v_point[0] - h_point[0]
            dy = v_point[1] - h_point[1]
            dz = v_point[2] - h_point[2]

            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            if distance < dmax:  
                #print(distance)
                return True
    return False


def process_hits_from_station_dict(station_hits, args):
    stations = []
    positions = []
    labels = []

    for station, hits in station_hits.items():
        vert_hits = hits["vertical"]
        hor_hits  = hits["horizontal"]

        if not vert_hits or not hor_hits:
            continue

        for v in vert_hits:
            for h in hor_hits:
                if ('MC' in  args.type):
                    label = len(v["trackID_set"] & h["trackID_set"]) > 0

                    if label == False and len(v["ecut_mc_points"])>0 and len(h["ecut_mc_points"])>0:
                        label = check_ecut_mc_hits(v["ecut_mc_points"], h["ecut_mc_points"])
                else:
                    label = False
                x = (v["x1"] + v["x2"]) / 2
                y = (h["y1"] + h["y2"]) / 2
                z = (v["z1"] + h["z1"]) / 2

                stations.append(station)
                positions.append([x, y, z])
                labels.append(label)
    stations = np.array(stations, dtype=int)
    positions = np.array(positions, dtype=float)
    labels = np.array(labels, dtype=int)
    return stations, positions, labels

def process_hits(event, snd_geo, args):
    Scifi = snd_geo.modules['Scifi']
    MuFilter = snd_geo.modules['MuFilter']
    A, B = ROOT.TVector3(), ROOT.TVector3()
    hit2MC = event.Digi_ScifiHits2MCPoints[0]

    station_hits = {}
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        
        detID = aHit.GetDetectorID()
        station = aHit.GetStation()
        orientation = aHit.isVertical()

        Scifi.GetSiPMPosition(detID, A, B)
        linksToMCPoints = hit2MC.wList(detID)

        trackIDs = []
        ecut_mc_points = []

        for mc_point_i, _ in linksToMCPoints:
            scifi_point = event.ScifiPoint[mc_point_i]
            trackID = scifi_point.GetTrackID()

            if trackID >= -1:
                trackIDs.append(trackID)

            if trackID == -2:
                x = scifi_point.GetX()
                y = scifi_point.GetY()
                z = scifi_point.GetZ()
                ecut_mc_points.append([x, y, z])

        hit_data= {
            "detID": detID,
            "x1": A.x(), "y1": A.y(), "z1": A.z(),
            "x2": B.x(), "y2": B.y(), "z2": B.z(),
            "trackID_set": set(trackIDs),
            "ecut_mc_points":ecut_mc_points
        }
        if station not in station_hits:
            station_hits[station] = {"vertical": [], "horizontal": []}

        key = "vertical" if orientation else "horizontal"
        station_hits[station][key].append(hit_data)

    stations, positions, labels = process_hits_from_station_dict(station_hits, args)

    return stations, positions, labels


def main(args):
    print("start processing digi to 3D hit")
    start_time = time.time()
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)
    out_file = args.out_path
    
    # Process each event
    event_data = []
    for i_event, event in enumerate(raw_tree):
        #print(dir(event))
        # if len(event.Digi_ScifiHits)>100:
        #     continue
        
        runId = event.EventHeader.GetRunId()
        if ('MC' in  args.type):
            if ('kaon' in args.type or 'neutron' in args.type):
                try:
                    eventId = event.EventHeader.GetEventNumber()
                except Exception:
                    eventId = event.EventHeader.GetMCEntryNumber()
            else:
                eventId = event.EventHeader.GetMCEntryNumber()
            # Particle codes and initial position
            event_pdg0 = event.MCTrack[0].GetPdgCode()
            event_pdg1 = event.MCTrack[1].GetPdgCode()

            neutrino_pdgCode = [12, -12, 14, -14, 16, -16]
            if (event_pdg0 == event_pdg1) and (event_pdg0 in neutrino_pdgCode):
                pdgCode = event_pdg0 - 100 if event_pdg0 < 0 else event_pdg0 + 100
            else:
                pdgCode = event_pdg0

        elif('real' in  args.type):
            pdgCode= 0
            eventId = event.EventHeader.GetEventNumber()
        

        stations, positions, labels = process_hits(event, snd_geo, args)
        if len(positions)==0:
            stations, positions, labels = np.array([0]), np.array([[0,0,0]]), np.array([0])
        #print(stations.shape, positions.shape, labels.shape)
        one_event_data = {
                    "pdgCode": pdgCode,
                    "runId": runId,
                    "eventId": eventId,
                    "hits_staton": stations, # [n_hits,1]
                    "hits_pos": positions,    # [n_hits,3]
                    "hits_label": labels,  # [n_hits,1]
                }
        print(f"event id: {i_event},digi hits: {len(event.Digi_ScifiHits)}, number of hits: {len(positions):.3e}" )
        #if len(positions)>100:
        #    continue
        event_data.append(one_event_data)
        #if(i_event>5):
        #    break

    compress_level=9
    with gzip.open(out_file, 'wb', compresslevel=compress_level) as f:
        pickle.dump(event_data, f)
        #torch.save(event_data, chunk_out_path)
    print(f"Saved data to {out_file}, compress level {compress_level}")

    elapsed = time.time() - start_time
    print(f"[TIMER] 3D hits processing took {elapsed:.3f} seconds")


    # Finalize the output file
    # with gzip.open(out_file, 'wb') as f:
    #     torch.save(event_data, f)
    #     #torch.save(event_data, chunk_out_path)
    # print(f"Saved data to {out_file}")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)
    parser.add_argument("-o", "--outPath", dest="out_path", help="output path", required=True)
    parser.add_argument("-t", "--type", dest='type', help='data type, MC or real', required=True)

    args = parser.parse_args()

    main(args)

# python digi_2_3Dhits.py -d /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP.root -g /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/geofile_full.Genie-TGeant4.root -o ./test_data/3dHits_test.pkl.gz -t MC