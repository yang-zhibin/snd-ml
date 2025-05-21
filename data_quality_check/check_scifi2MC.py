import ROOT
import SndlhcGeo
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import matplotlib.cm as cm
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import math


def plot_intersections_and_mc_triptych(intersections_df, mc_df, pdf, event_pdg0):

    """
    Plot 3D reconstructed intersections (all), MC truth hits, and only true label intersections
    in three separate subplots.

    Parameters:
        intersections_df (pd.DataFrame): ['x', 'y', 'z', 'label', 'station']
        mc_df (pd.DataFrame): ['x', 'y', 'z']
    """
    os.makedirs("plot", exist_ok=True)

    fig = plt.figure(figsize=(18, 6))
    

    # Subplot 1: All reconstructed intersections
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    for label, color in [(True, 'green'), (False, 'red')]:
        subset = intersections_df[intersections_df['label'] == label]
        ax1.scatter(subset['x'], subset['y'], subset['z'],
                    color=color, label=f"Label {label}", s=1, alpha=0.01)

    ax1.set_title(f"All Reconstructed Intersections, pdg: {event_pdg0}")
    ax1.set_xlabel("x [cm]")
    ax1.set_ylabel("y [cm]")
    ax1.set_zlabel("z [cm]")
    ax1.set_xlim(-50, 0) 
    ax1.set_ylim(10, 60)
    ax1.set_zlim(275, 375)
    ax1.legend(fontsize='small')

    # Subplot 2: MC Truth Hits
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    mc_normal = mc_df[mc_df['trackID'] != -2]
    mc_special = mc_df[mc_df['trackID'] == -2]

    # Plot standard MC hits
    ax2.scatter(mc_normal['x'], mc_normal['y'], mc_normal['z'],
                color='blue', label="MC Truth Hits", s=1, alpha=0.05)

    # Plot special hits with trackID == -2 in yellow
    ax2.scatter(mc_special['x'], mc_special['y'], mc_special['z'],
                color='yellow', label="trackID = -2", s=1, alpha=0.05)

    ax2.set_title("MC Truth SciFi Hits")
    ax2.set_xlabel("x [cm]")
    ax2.set_ylabel("y [cm]")
    ax2.set_zlabel("z [cm]")
    ax2.set_xlim(-50, 0)
    ax2.set_ylim(10, 60)
    ax2.set_zlim(275, 375)
    ax2.legend(fontsize='small')

    # Subplot 3: Only true label intersections
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    true_hits = intersections_df[intersections_df['label'] == True]
    ax3.scatter(true_hits['x'], true_hits['y'], true_hits['z'],
                color='green', label="Label True Only", s=1, alpha=0.1)

    ax3.set_title("Only True Label Intersections")
    ax3.set_xlabel("x [cm]")
    ax3.set_ylabel("y [cm]")
    ax3.set_zlabel("z [cm]")
    ax3.set_xlim(-50, 0)
    ax3.set_ylim(10, 60)
    ax3.set_zlim(275, 375)
    ax3.legend(fontsize='small')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

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


# def check_ecut_mc_hits(v_ecut_mc_points, h_ecut_mc_points, dmax=1):
#     v_points = np.atleast_2d(np.array(v_ecut_mc_points, dtype=float))
#     h_points = np.atleast_2d(np.array(h_ecut_mc_points, dtype=float))

#     diffs = v_points[:, np.newaxis, :] - h_points[np.newaxis, :, :]  # Shape (N_v, N_h, 3)
#     abs_diffs = np.abs(diffs)

#     condition = (abs_diffs[..., 0] < dmax) & (abs_diffs[..., 1] < dmax) & (abs_diffs[..., 2] < dmax)

#     if np.any(condition):
#         return True

#     return False

# def check_ecut_mc_hits(v_ecut_mc_points, h_ecut_mc_points):
#     v_points = np.array(v_ecut_mc_points)
#     h_points = np.array(h_ecut_mc_points)

#     # Efficient broadcasting to compute pairwise differences
#     diffs = v_points[:, np.newaxis, :] - h_points[np.newaxis, :, :]  # Shape: (N_v, N_h, 3)

#     # Early exclusion: check if any dx, dy > 2
#     if np.all(np.abs(diffs[..., 0]) > 2) or np.all(np.abs(diffs[..., 1]) > 2):
#         return False

#     # Compute squared Euclidean distances
#     distances_squared = np.sum(diffs ** 2, axis=-1)

#     # Check if any distance < 1 cm (i.e., squared < 1)
#     mask = distances_squared < 1
#     if np.any(mask):
#         distance = np.sqrt(distances_squared[mask][0])
#         #rint(distance)
#         return True

#     return False


# def check_ecut_mc_hits(v_ecut_mc_points, h_ecut_mc_points, dmax=1):
#     v = np.asarray(v_ecut_mc_points, dtype=np.float32)
#     h = np.asarray(h_ecut_mc_points, dtype=np.float32)

#     if v.ndim == 1:
#         v = v[None, :]
#     if h.ndim == 1:
#         h = h[None, :]

#     # Broadcasting-friendly early filtering based on bounding box
#     v_min, v_max = v.min(axis=0) - dmax, v.max(axis=0) + dmax
#     mask = np.all((h >= v_min) & (h <= v_max), axis=1)
#     h = h[mask]

#     if h.size == 0:
#         return False

#     # Compute differences and check box condition
#     for vi in v:
#         diffs = np.abs(h - vi)
#         if np.any(np.all(diffs < dmax, axis=1)):
#             return True

#     return False

def process_hits_from_station_dict(station_hits):
    import time
    start_time = time.time()
    intersections = []

    for station, hits in station_hits.items():
        vert_hits = hits["vertical"]
        hor_hits  = hits["horizontal"]

        if not vert_hits or not hor_hits:
            continue

        for v in vert_hits:
            for h in hor_hits:

                label = len(v["trackID_set"] & h["trackID_set"]) > 0

                if label == False and len(v["ecut_mc_points"])>0 and len(h["ecut_mc_points"])>0:
                    label = check_ecut_mc_hits(v["ecut_mc_points"], h["ecut_mc_points"])
                    
                    #return 0
                # print()
                # print(" ver trackID:",v["trackID_set"])
                # print(" hor trackID:",h["trackID_set"])
                # print(" label", label)
                x = (v["x1"] + v["x2"]) / 2
                y = (h["y1"] + h["y2"]) / 2
                z = (v["z1"] + h["z1"]) / 2

                intersections.append({
                    "station": station,
                    "detID_ver": v["detID"],
                    "detID_hor": h["detID"],
                    "x": x, "y": y, "z": z,
                    "label": label,
                })

    elapsed = time.time() - start_time
    print(f"[INFO] Number of intersections: {len(intersections)}")
    print(f"[TIMER] Intersection processing took {elapsed:.3f} seconds")

    #print(pd.DataFrame(intersections))

    #plot_intersections_3d(pd.DataFrame(intersections))
    return pd.DataFrame(intersections)
def main():
    geo_path = '/afs/cern.ch/user/z/zhibin/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/241/geofile_full.Genie-TGeant4.root'
    digi_path = '/afs/cern.ch/user/z/zhibin/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/241/sndLHC.Genie-TGeant4_20240126_digCPP.root'
    snd_geo = setup_geometry(geo_path)
    raw_data, raw_tree = open_root_file(digi_path)

    A, B = ROOT.TVector3(), ROOT.TVector3()
    Scifi = snd_geo.modules['Scifi']
    MuFilter = snd_geo.modules['MuFilter']

    # station, orientation, trackID, x1, y1, z1, x2, y2, z2 # trackID will be a list
    with PdfPages("plot/reco_vs_mc_3d_triptych_all_events.pdf") as pdf:
        for i_event, event in enumerate(raw_tree):
            event_pdg0 = event.MCTrack[0].GetPdgCode()
            hit2MC = event.Digi_ScifiHits2MCPoints[0]
            print("n of Digi_ScifiHit:", len(event.Digi_ScifiHits))
            station_hits = {}
            for trk in event.MCTrack:
                #print(dir(trk))
                break
            #break
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
                        # t = scifi_point.GetTime()  # optional
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

            mc_hits = []

            for aHit in event.ScifiPoint:
                x = aHit.GetX()
                y = aHit.GetY()
                z = aHit.GetZ()
                trackID = aHit.GetTrackID()
                mc_hits.append({
                                "x": x,
                                "y": y,
                                "z": z,
                                "trackID":trackID,
                                    })
            mc_df = pd.DataFrame(mc_hits)
            intersections_df = process_hits_from_station_dict(station_hits)
            if len(intersections_df)<1:
                continue
            plot_intersections_and_mc_triptych(intersections_df, mc_df, pdf, event_pdg0)

            if i_event>=20:
                break


if __name__ == "__main__": 
    main()