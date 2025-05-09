import ROOT
import SndlhcGeo


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

def main():
    geo_path = '/afs/cern.ch/user/z/zhibin/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/geofile_full.Genie-TGeant4.root'
    digi_path = '/afs/cern.ch/user/z/zhibin/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_20240126_digCPP.root'
    snd_geo = setup_geometry(geo_path)
    raw_data, raw_tree = open_root_file(digi_path)

    # station, orientation, trackID, x1, y1, z1, x2, y2, z2
    
    for i_event, event in enumerate(raw_tree):
        hit2MC = event.Digi_ScifiHits2MCPoints[0]
        print("n of Digi_ScifiHit:", len(event.Digi_ScifiHits))
        for aHit in event.Digi_ScifiHits:
            detID = aHit.GetDetectorID()
            linksToMCPoints = hit2MC.wList(detID)
            print(f"detID: {detID}")
            for mc_point_i, _ in linksToMCPoints:
                trackID = event.ScifiPoint[mc_point_i].GetTrackID()
                print(f"    trackID:{trackID}")


            #print(f"linksToMCPoints of detid_{detID}:",linksToMCPoints)
            #break
        break


if __name__ == "__main__": 
    main()