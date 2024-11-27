import ROOT
import os
from argparse import ArgumentParser
import SndlhcGeo
from array import array
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(geo.modules['Scifi'])
    lsOfGlobals.Add(geo.modules['MuFilter'])
    return geo

def process_events(event):
    A, B = ROOT.TVector3(), ROOT.TVector3()

    scifi_avg_ver = 0
    scifi_avg_hor = 0
    scifi_n_ver = 0
    scifi_n_hor = 0

    DS_avg_ver = 0
    DS_avg_hor = 0
    DS_n_ver = 0
    DS_n_hor = 0



    # Process SciFi hits
    for aHit in event.Digi_ScifiHits:
        if not aHit.isValid():
            continue
        # Calculate average positions
        detID = aHit.GetDetectorID()
        channel = aHit.GetSiPMChan()
        mat = aHit.GetMat()
        sipm = aHit.GetSiPM()
        x = channel + sipm * 128 + mat * 4 * 128
        if aHit.isVertical():
            scifi_avg_ver += x
            scifi_n_ver += 1
        else:
            scifi_avg_hor += x
            scifi_n_hor += 1

    # Process MuFilter hits


    for aHit in event.Digi_MuFilterHits:
        if not aHit.isValid():
            continue

        detID = aHit.GetDetectorID()
        # DS hit averaging, considering only system '3' which is downstream
        if aHit.GetSystem() == 3:
            x = detID % 1000
            if aHit.isVertical():
                DS_avg_ver += x
                DS_n_ver += 1
            else:
                DS_avg_hor += x
                DS_n_hor += 1

    # Compute final averages
    if scifi_n_hor > 0:
        scifi_avg_hor /= scifi_n_hor
    else:
        scifi_avg_hor = -1

    if scifi_n_ver > 0:
        scifi_avg_ver /= scifi_n_ver
    else:
        scifi_avg_ver = -1

    if DS_n_hor > 0:
        DS_avg_hor /= DS_n_hor
    else:
        DS_avg_hor = -1

    if DS_n_ver > 0:
        DS_avg_ver /= DS_n_ver
    else:
        DS_avg_ver = -1


    # Update label averages
    return  scifi_avg_ver, scifi_avg_hor, DS_avg_ver, DS_avg_hor

def plot_avg_ds(df, name):

    # Plot the 2D histogram
    plt.figure(figsize=(8, 6))
    plt.hist2d(df["DS_avg_ver"].dropna(), df["DS_avg_hor"].dropna(), bins=60, cmap='Blues')

    #plt.xlim(0, 60)
    #plt.ylim(60, 120)


    # Define the coordinates for the box
    left_top = (10, 105)
    right_bottom = (50, 70)

    # Calculate width and height
    width = right_bottom[0] - left_top[0]
    height = left_top[1] - right_bottom[1]

    # Get current axis and add the rectangle patch
    #ax = plt.gca()
    #rect = patches.Rectangle(left_top, width, -height, linewidth=1, edgecolor='red', facecolor='none')
    #ax.add_patch(rect)

    # Add colorbar, labels, and title
    plt.colorbar(label="Frequency")
    plt.xlabel("DS_avg_ver")
    plt.ylabel("DS_avg_hor")
    plt.title(f"{name} (2D Histogram of DS_avg_ver vs DS_avg_hor)")

    # Save the plot
    plt.savefig(f"plots/{name}_2d_histogram_DS_avg_ver_vs_DS_avg_hor.png", dpi=300)
    #plt.show()


def main():
    geo_file = "/eos/experiment/sndlhc/convertedData/physics/2023_reprocess/geofile_sndlhc_TI18_V4_2023.root"

    geo = setup_geometry(geo_file)
    Scifi = geo.snd_geo.Scifi
    MuFilter = geo.snd_geo.MuFilter

    digi_file_path = "/eos/user/z/zhibin/sndData/converted/veto_ineff/scifi_plane1_partition20.root"
    
    digi = ROOT.TFile(digi_file_path, 'read')
    d_tree = digi.Get('cbmsim')

    columns = ["scifi_avg_ver", "scifi_avg_hor", "DS_avg_ver", "DS_avg_hor"]
    df = pd.DataFrame(columns=columns)

    for i_event, event in enumerate(d_tree):
        scifi_avg_ver, scifi_avg_hor, DS_avg_ver, DS_avg_hor = process_events(event)
        # add return to df
        df = df.append({
            "scifi_avg_ver": scifi_avg_ver,
            "scifi_avg_hor": scifi_avg_hor,
            "DS_avg_ver": DS_avg_ver,
            "DS_avg_hor": DS_avg_hor
        }, ignore_index=True)

        #break
    df.to_csv("data/scifi_plane1.csv", index=False)

def cal_pos_df(digi_file_path, name):
    geo_file = "/eos/experiment/sndlhc/convertedData/physics/2023_reprocess/geofile_sndlhc_TI18_V4_2023.root"

    geo = setup_geometry(geo_file)
    Scifi = geo.snd_geo.Scifi
    MuFilter = geo.snd_geo.MuFilter

    #digi_file_path = "/eos/user/z/zhibin/sndData/converted/veto_ineff/scifi_plane1_partition20.root"
    
    digi = ROOT.TFile(digi_file_path, 'read')
    d_tree = digi.Get('cbmsim')

    columns = ["scifi_avg_ver", "scifi_avg_hor", "DS_avg_ver", "DS_avg_hor"]
    df = pd.DataFrame(columns=columns)

    for i_event, event in enumerate(d_tree):
        scifi_avg_ver, scifi_avg_hor, DS_avg_ver, DS_avg_hor = process_events(event)
        # add return to df
        df = df.append({
            "scifi_avg_ver": scifi_avg_ver,
            "scifi_avg_hor": scifi_avg_hor,
            "DS_avg_ver": DS_avg_ver,
            "DS_avg_hor": DS_avg_hor
        }, ignore_index=True)

        #break
    df.to_csv(f"data/{name}.csv", index=False)

def read_processed_data():
    folder = '/eos/user/z/zhibin/sndData/converted/veto_ineff/'
    scifi_plane1 = []
    scifi_plane2 = []
    veto_and_us15_ds14_partition1 = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("scifi") and file.endswith("root"):
                matching_files.append(os.path.join(root, file))
    return matching_files

if __name__ == "__main__":
    main()
    partition = 20

    # scifi1345 = f"/eos/user/z/zhibin/sndData/converted/veto_ineff/veto_and_us15_ds14_scfifi1345_partition{partition}.root"
    # scifi2345 = f"/eos/user/z/zhibin/sndData/converted/veto_ineff/veto_and_us15_ds14_scfifi2345_partition{partition}.root"

    # scifi_plane1 = f"/eos/user/z/zhibin/sndData/converted/veto_ineff/scifi_plane1_partition{partition}.root"
    # scifi_plane2 = f"/eos/user/z/zhibin/sndData/converted/veto_ineff/scifi_plane2_partition{partition}.root"
    
    # cal_pos_df(scifi1345, 'scifi1345')
    # cal_pos_df(scifi2345, 'scifi2345')
    # cal_pos_df(scifi_plane1, 'scifi_plane1')
    # cal_pos_df(scifi_plane2, 'scifi_plane2')
    #read_processed_data()

    #df = pd.read_csv("data/scifi1.csv")
    df_scifi1345 =pd.read_csv('data/scifi1345.csv')
    df_scifi2345 = pd.read_csv('data/scifi2345.csv')
    df_scifi_plane1 = pd.read_csv('data/scifi_plane1.csv')
    df_scifi_plane2 = pd.read_csv('data/scifi_plane2.csv')

    plot_avg_ds(df_scifi1345, 'scifi1345')
    plot_avg_ds(df_scifi2345, 'scifi2345')
    plot_avg_ds(df_scifi_plane1, 'scifi_plane1')
    plot_avg_ds(df_scifi_plane2, 'scifi_plane2')

