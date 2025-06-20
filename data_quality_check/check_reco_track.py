import pandas as pd
import os
import ROOT


def read_reco_track_files():
    folder = "/eos/experiment/sndlhc/users/sii/2024/8329/"  

    files = []
    for fname in os.listdir(folder):
        if not fname.startswith("sndsw_raw-") or not fname.endswith("_8329_muonReco.root"):
            continue
        try:
            parts = fname[len("sndsw_raw-"):].split("_")
            partition = int(parts[0])
            sub_partition = int(parts[1])
            full_path = os.path.join(folder, fname)
            files.append((partition, sub_partition, full_path))
        except (IndexError, ValueError):
            continue  # skip if format doesn't match

    files.sort()
    return files
    



def main():
    reco_track_files = read_reco_track_files()
    

    reco_chain = ROOT.TChain("rawConv")
    
    count = 0
    for _, _, file_path in reco_track_files:
        reco_chain.Add(file_path)
        print(file_path)
        if count > 0:
            break
        count+=1

    for branch in reco_chain.GetListOfBranches():
        print(branch.GetName())
    for i in range(reco_chain.GetEntries()):
        reco_chain.GetEntry(i)
        reco_tracks = reco_chain.Reco_MuonTracks
        print(f"{i} event")
        for track in reco_tracks:
            #print("Track object:", track)
            
            # Print all attributes/methods
            #print(dir(track))  # Uncomment if you want to explore all available members
            #pos = track.extrapolateToPlaneAtZ(0)
            #print(dir(pos))
            # Now call and print the requested methods
            #print("extrapolateToPlaneAtZ(0):", track.extrapolateToPlaneAtZ(0))  # you can change Z value
            #print("getAngleXZ():", track.getAngleXZ())
            #print("getAngleYZ():", track.getAngleYZ())
            #print("getChi2():", track.getChi2())
            print("     getChi2Ndf():", track.getChi2Ndf())
            print("     getNdf():", track.getNdf())
            #print("getSlopeXZ():", track.getSlopeXZ())
            #print("getSlopeYZ():", track.getSlopeYZ())
            print("     getStart():", track.getStart().X(), track.getStart().Y(), track.getStart().Z())
            print("     getStop():", track.getStop().X(), track.getStop().Y(), track.getStop().Z())
            print("     getTrackFlag():", track.getTrackFlag())
            #print("getTrackMom():", track.getTrackMom())
            #print("getTrackPoints():", list(track.getTrackPoints()))
            print("     getTrackType():", track.getTrackType())

            print("     ---")

            #break
        if i>100:
            break



if __name__ == "__main__":
    main()