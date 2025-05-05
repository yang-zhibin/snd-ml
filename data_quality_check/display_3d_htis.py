
import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
import pandas as pd


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


def process_hits(event, snd_geo):
    Scifi = snd_geo.modules['Scifi']
    A, B = ROOT.TVector3(), ROOT.TVector3()


    # for branch in event.GetListOfBranches():
    #     print(branch.GetName())
    #print(dir(event))
    scifi_hits = pd.DataFrame(columns=['x', 'y', 'z', 'pdgCode', 'detID'])
    print("reading scifi point")
    for aHit in event.ScifiPoint:
        #print(dir(aHit))
        scifi_hits.loc[len(scifi_hits)] = [aHit.GetX(), aHit.GetY(), aHit.GetZ(),  aHit.PdgCode(), aHit.GetDetectorID()]

    scifi_hits = scifi_hits.sort_values(by='z')
    print(scifi_hits)
    columns = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'orientation', 'station', 'detID']
    digi_hits = pd.DataFrame(columns=columns)

    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        detID = aHit.GetDetectorID()
        station =int(detID/1E6)
        Scifi.GetSiPMPosition(detID, A, B)

        orientation = aHit.isVertical()
        x1, y1, z1 = A.x(), A.y(), A.z()
        x2, y2, z2 = B.x(), B.y(), B.z()
        digi_hits.loc[len(digi_hits)] = [ x1, y1, z1, x2, y2, z2, orientation, station, detID]
    digi_hits = digi_hits.sort_values(by='z1')
    print(digi_hits)

    scifi_hits.to_csv("scifi_hits.csv")
    digi_hits.to_csv("digi_hits.csv")
    return scifi_hits,digi_hits

def plot_hits(digi_hits, scifi_hits):
    pass


def main(args):
    print("displaing 3d hits")
    snd_geo = setup_geometry(args.geo_path )
    raw_data, raw_tree = open_root_file(args.digi_path)


    for i_event, event in enumerate(raw_tree):
        if not (event.Digi_ScifiHits.GetEntriesFast() or event.Digi_MuFilterHits.GetEntriesFast()):
            print("not scifi hit or MuFilter hit")
        else:
            digi_hits, scifi_hits = process_hits(event, snd_geo)

        plot_hits(digi_hits, scifi_hits)
        break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--digiPath", dest="digi_path", help="digitized data file path", required=True)
    parser.add_argument("-g", "--geoPath", dest="geo_path", help="geo path", required=True)

    args = parser.parse_args()

    main(args)