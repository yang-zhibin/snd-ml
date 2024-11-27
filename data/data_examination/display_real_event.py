import ROOT
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import SndlhcGeo 

def which_wall(pos):
    wall_coords = [0, 300.+1, 313.+1, 326.+1, 339.+1, 352.+3]
    for i in range(5):
        if (pos>wall_coords[i]) and (pos<wall_coords[i+1]):
            return i
    return -1

def calculate_new_coordinates(point1, point2, new_z):
    # Extract x, y, and z coordinates of the points
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    # Calculate the change in z
    delta_z = z2 - z1
    
    # Calculate the scaling factor for the change in z
    scale_factor = (new_z - z1) / delta_z
    
    # Calculate the new x and y coordinates based on the scaling factor
    new_x = x1 + scale_factor * (x2 - x1)
    new_y = y1 + scale_factor * (y2 - y1)
    
    return [new_x, new_y, new_z]

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def main():
    # param
    drawCluster = False
    drawScifi = True
    drawReco = False
    drawMChit = False
    drawMuHit = True

    pdf_name = 'plots/scifi_first_event.pdf'
    
    # read files
    digi_file_path = "data/scifi1.root"
    geo_file_path = "/eos/experiment/sndlhc/convertedData/physics/2023_reprocess/geofile_sndlhc_TI18_V4_2023.root"

    digi = ROOT.TFile(digi_file_path, 'read')
    d_tree = digi.Get('cbmsim')

    geo = SndlhcGeo.GeoInterface(geo_file_path)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(geo.modules['Scifi'])
    lsOfGlobals.Add(geo.modules['MuFilter'])
    Scifi = geo.modules['Scifi']
    Mufi = geo.modules['MuFilter']
    nav = ROOT.gGeoManager.GetCurrentNavigator()

    A, B = ROOT.TVector3(), ROOT.TVector3()

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

    x_lim = (-80, 0)
    y_lim = (0, 80)
    z_lim = (250, 600)

    # Create a PdfPages object to save plots
    with PdfPages(pdf_name) as pdf:

        for i_event in range(d_tree.GetEntries()):
            d_tree.GetEntry(i_event)
            print(f"Processing event: {i_event}")

            # Create a figure for each event
            fig_3d = plt.figure(figsize=(20, 16))
            ax_3d = fig_3d.add_subplot(111, projection='3d')

            # Create 2D projections (use figure sizes corresponding to limit ranges)
            fig_xy, ax_xy = plt.subplots(figsize=((x_lim[1] - x_lim[0]) / 10, (y_lim[1] - y_lim[0]) / 10))
            fig_xz_yz, (ax_xz, ax_yz) = plt.subplots(2, 1, figsize=((z_lim[1] - z_lim[0]) / 10, (x_lim[1] - x_lim[0] + y_lim[1] - y_lim[0]) / 10))

            for aHit in d_tree.Digi_ScifiHits:
                if not drawScifi:
                    break

                detID = aHit.GetDetectorID()
                geo.modules['Scifi'].GetSiPMPosition(detID, A, B)

                point1 = [A.x(), A.y(), A.z()]
                point2 = [B.x(), B.y(), B.z()]

                wall = which_wall(A.z())

                x_coords = [point1[0], point2[0]]
                y_coords = [point1[1], point2[1]]
                z_coords = [point1[2], point2[2]]

                # Plot in 3D
                ax_3d.plot(x_coords, y_coords, z_coords, marker='o', color=color_dict.get(wall, 'grey'), alpha=0.3)

                # Plot in 2D projections
                ax_xy.plot(x_coords, y_coords, marker='o', color=color_dict.get(wall, 'grey'), alpha=0.3)
                ax_xz.plot(z_coords, x_coords, marker='o', markersize=8, linewidth=2, color=color_dict.get(wall, 'grey'), alpha=0.6)
                ax_yz.plot(z_coords, y_coords, marker='o', markersize=8, linewidth=2, color=color_dict.get(wall, 'grey'), alpha=0.6)

            for aHit in d_tree.Digi_MuFilterHits:
                if not drawMuHit:
                    break

                detID = aHit.GetDetectorID()
                geo.modules['MuFilter'].GetPosition(detID, A, B)

                point1 = [A.x(), A.y(), A.z()]
                point2 = [B.x(), B.y(), B.z()]

                n_sys = detID // 10000 + 3

                x_coords = [point1[0], point2[0]]
                y_coords = [point1[1], point2[1]]
                z_coords = [point1[2], point2[2]]

                # Plot in 3D
                ax_3d.plot(x_coords, y_coords, z_coords, marker='o', color=color_dict.get(n_sys, 'grey'), alpha=0.3)

                # Plot in 2D projections
                ax_xy.plot(x_coords, y_coords, marker='o', color=color_dict.get(n_sys, 'grey'), alpha=0.3)
                ax_xz.plot(z_coords, x_coords, marker='o', markersize=8, linewidth=2, color=color_dict.get(n_sys, 'grey'), alpha=0.6)
                ax_yz.plot(z_coords, y_coords, marker='o', markersize=8, linewidth=2, color=color_dict.get(n_sys, 'grey'), alpha=0.6)

            # Set axis limits for 3D plot (fixed)
            ax_3d.set_xlim(x_lim)
            ax_3d.set_ylim(y_lim)
            ax_3d.set_zlim(z_lim)
            ax_3d.set_xlabel('X Label')
            ax_3d.set_ylabel('Y Label')
            ax_3d.set_zlabel('Z Label')
            ax_3d.set_title(f'3D Visualization of SciFi and MuFilter Hits - Event {i_event}')
            ax_3d.grid(True)

            # Set the view so that the Z-axis is at the bottom, then rotate up by 45 degrees and counter-clockwise by 45 degrees
            #ax_3d.view_init(elev=60, azim=60)

            # Set the same axis limits for XY projection (fixed)
            ax_xy.set_xlim(x_lim)
            ax_xy.set_ylim(y_lim)
            ax_xy.set_xlabel('X Label')
            ax_xy.set_ylabel('Y Label')
            ax_xy.set_title(f'XY Projection - Event {i_event}')
            ax_xy.grid(True)

            # Set axis labels and limits for XZ and YZ projections (real scale with given limits)
            ax_xz.set_xlim(z_lim)
            ax_xz.set_ylim(x_lim)
            ax_xz.set_xlabel('Z Label')
            ax_xz.set_ylabel('X Label')
            ax_xz.set_title(f'XZ Projection - Event {i_event}')
            ax_xz.grid(True)

            ax_yz.set_xlim(z_lim)
            ax_yz.set_ylim(y_lim)
            ax_yz.set_xlabel('Z Label')
            ax_yz.set_ylabel('Y Label')
            ax_yz.set_title(f'YZ Projection - Event {i_event}')
            ax_yz.grid(True)

            # Adjust spacing between subplots for combined XZ-YZ plot
            fig_xz_yz.tight_layout()

            # Save the plots for the current event to the PDF
            pdf.savefig(fig_3d)
            pdf.savefig(fig_xy)
            pdf.savefig(fig_xz_yz)

            # Close figures to free memory
            plt.close(fig_3d)
            plt.close(fig_xy)
            plt.close(fig_xz_yz)

            # Stop after the first event
            #break

if __name__ == "__main__":
    main()
