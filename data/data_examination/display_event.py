import ROOT
import matplotlib.pyplot as plt
import numpy as np

import SndlhcGeo 

# Your other existing functions and imports remain here...

def main():
    # Parameters
    drawCluster = False
    drawScifi = True
    drawReco = False
    drawMChit = False
    drawMuHit = True

    out_name = 'plots/real_hits.png'
    
    # File paths (use your paths here)
    digi_file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP.root"
    geo_file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/geofile_full.Genie-TGeant4.root"
    recoMuon_file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP_muonReco.root"

    # Read ROOT files
    digi = ROOT.TFile(digi_file_path, 'read')
    d_tree = digi.Get('cbmsim')
    reco = ROOT.TFile(recoMuon_file_path, 'read')
    r_tree = reco.Get('cbmsim')
    geo = SndlhcGeo.GeoInterface(geo_file_path)

    # Geometry setup
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(geo.modules['Scifi'])
    lsOfGlobals.Add(geo.modules['MuFilter'])
    Scifi = geo.modules['Scifi']
    Mufi = geo.modules['MuFilter']
    nav = ROOT.gGeoManager.GetCurrentNavigator()

    # Initialize vectors for positions
    A, B = ROOT.TVector3(), ROOT.TVector3()

    # Plot setup
    fig = plt.figure(figsize=(20, 16))
    ax_3d = fig.add_subplot(221, projection='3d')  # 3D Plot
    ax_xy = fig.add_subplot(222)  # XY Projection
    ax_xz = fig.add_subplot(223)  # XZ Projection
    ax_yz = fig.add_subplot(224)  # YZ Projection

    color_dict = {
        0: 'red',
        1: 'green',
        2: 'blue',
        3: 'orange',
        4: 'purple',
        5: 'darkred', 
        6: 'darkblue',
        -1: 'grey'
    }

    # Iterate over events
    for i_event in range(r_tree.GetEntries()):
        d_tree.GetEntry(i_event)
        r_tree.GetEntry(i_event)

        # Extract hits and plot them (example with Scifi hits)
        for aHit in d_tree.Digi_ScifiHits:
            if not drawScifi:
                break
            detID = aHit.GetDetectorID()
            geo.modules['Scifi'].GetSiPMPosition(detID, A, B)
            point1 = [A.x(), A.y(), A.z()]
            point2 = [B.x(), B.y(), B.z()]
            
            wall = which_wall(A.z())
            color = color_dict.get(wall, 'grey')
            
            # Extract x, y, z coordinates
            x_coords = [point1[0], point2[0]]
            y_coords = [point1[1], point2[1]]
            z_coords = [point1[2], point2[2]]

            # Plot in 3D
            ax_3d.plot(x_coords, y_coords, z_coords, marker='o', color=color, alpha=0.1)

            # Plot in 2D projections
            ax_xy.plot(x_coords, y_coords, color=color, alpha=0.1)  # XY Projection
            ax_xz.plot(x_coords, z_coords, color=color, alpha=0.1)  # XZ Projection
            ax_yz.plot(y_coords, z_coords, color=color, alpha=0.1)  # YZ Projection

    # Labels and titles for 3D plot
    ax_3d.set_xlabel('X Label')
    ax_3d.set_ylabel('Y Label')
    ax_3d.set_zlabel('Z Label')
    ax_3d.set_title('3D Line Plot with Multiple Lines')

    # Labels and titles for 2D projections
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.set_title('XY Projection')

    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_xz.set_title('XZ Projection')

    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    ax_yz.set_title('YZ Projection')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)

if __name__ == "__main__":
    main()
