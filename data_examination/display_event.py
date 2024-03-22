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

def main():

    # param
    drawCluster = False
    drawScifi = True
    drawReco = False
    drawMChit = False
    drawMuHit = False

    out_name = '/afs/cern.ch/user/z/zhibin/work/snd-ml/data_examination/plots/3d_plot_vm_Scifi.png'
    

    # read files
    digi_file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP.root"
    geo_file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/geofile_full.Genie-TGeant4.root"
    recoMuon_file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP_muonReco.root"

    digi = ROOT.TFile(digi_file_path, 'read')
    d_tree = digi.Get('cbmsim')

    reco = ROOT.TFile(recoMuon_file_path, 'read')
    r_tree = reco.Get('cbmsim')


    geo = SndlhcGeo.GeoInterface(geo_file_path)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(geo.modules['Scifi'])
    lsOfGlobals.Add(geo.modules['MuFilter'])
    Scifi = geo.modules['Scifi']
    Mufi = geo.modules['MuFilter']
    nav = ROOT.gGeoManager.GetCurrentNavigator()

    A, B = ROOT.TVector3(), ROOT.TVector3()
    for i_event in range(r_tree.GetEntries()):
        d_tree.GetEntry(i_event)
        r_tree.GetEntry(i_event)
        #if (i_event==0):
        #    break
        if len(r_tree.Reco_MuonTracks)>0:
            break

    
    

    i_hit= 0
    lines = []
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')

    #help(geo.modules['Scifi'])
    color_dict = {
        0: 'red',
        1: 'green',
        2: 'blue',
        3: 'orange',
        4: 'purple',
        5: 'darkred', 
        6: 'darkblue' ,
        -1: 'grey'
    }

    event_pdg0 = d_tree.MCTrack[0].GetPdgCode()
    event_pdg1 = d_tree.MCTrack[1].GetPdgCode()

    for cluster in d_tree.Cluster_Scifi:
        if (drawCluster==False):
            break

        
        cluster.GetPosition(A, B)
        point1 = [A.x(),A.y(), A.z()]
        point2 = [B.x(),B.y(), B.z()]

        wall = which_wall(A.z())
        if wall!=0:
            continue

        x_coords = [point1[0], point2[0]]
        y_coords = [point1[1], point2[1]]
        z_coords = [point1[2], point2[2]]
        ax.plot(x_coords, y_coords, z_coords, color = 'brown', alpha=0.3)
        

    print("PDG:",event_pdg0)
    for muon_trk in r_tree.Reco_MuonTracks:
        if (drawReco==False):
            break
        #print("working on reco")
        mom = muon_trk.getFittedState().getMom()
        pos = muon_trk.getFittedState().getPos()

        point_pos = [pos.x(),pos.y(), pos.z()]
        point_mom = [mom.x(),mom.y(), mom.z()]

        point1 = calculate_new_coordinates(point_pos, point_mom, 280)
        point2 = calculate_new_coordinates(point_pos, point_mom, 600)

        print(point1, point2)
        
        x_coords = [point1[0], point2[0]]
        y_coords = [point1[1], point2[1]]
        z_coords = [point1[2], point2[2]]

        ax.plot(x_coords, y_coords, z_coords, marker='1', alpha=0.5)


    i_track=0
    for mc_track in d_tree.MCTrack:
        if (drawMChit==False):
            break
        #if(i_track>30):
        #    break
        i_track+=1
        z1 = mc_track.GetStartZ()
        x1 = mc_track.GetStartX()
        y1 = mc_track.GetStartY()
        if not ( x1 < 0 and x1 > -50 and y1 < 60 and y1 > 10): # z1<300.+1 and z1 >0 and
            continue
        print("mctrack", x1, y1, z1)

        ax.scatter(x1, y1, z1, s=10)

    
    for aHit in d_tree.Digi_ScifiHits:
        if (drawScifi==False):
            break
        #print('hit_id:', i_hit)
        detID = aHit.GetDetectorID()
        vert = aHit.isVertical()
        geo.modules['Scifi'].GetSiPMPosition(detID, A, B)

        point1 = [A.x(),A.y(), A.z()]
        point2 = [B.x(),B.y(), B.z()]
        
        wall = which_wall(A.z())
        
        #if wall!=0:
        #    continue
        #print("point1", point1, "point2",point2)

        # Extract x, y, z coordinates for each point
        x_coords = [point1[0], point2[0]]
        y_coords = [point1[1], point2[1]]
        z_coords = [point1[2], point2[2]]

        ax.plot(x_coords, y_coords, z_coords, marker='o', color=color_dict[wall], alpha=0.1)


        i_hit+=1

        #if (i_hit>30):
        #    break
    
    for aHit in d_tree.Digi_MuFilterHits:
        if (drawMuHit==False):
            break
        #print('hit_id:', i_hit)
        detID = aHit.GetDetectorID()
        vert = aHit.isVertical()
        geo.modules['MuFilter'].GetPosition(detID, A, B)

        point1 = [A.x(),A.y(), A.z()]
        point2 = [B.x(),B.y(), B.z()]
        

        n_sys = detID // 10000 + 3
        
        #if wall!=0:
        #    continue
        #print("point1", point1, "point2",point2)

        # Extract x, y, z coordinates for each point
        x_coords = [point1[0], point2[0]]
        y_coords = [point1[1], point2[1]]
        z_coords = [point1[2], point2[2]]

        ax.plot(x_coords, y_coords, z_coords, marker='o', color=color_dict[n_sys], alpha=0.1)


        i_hit+=1

        #if (i_hit>30):
        #    break


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Line Plot with Multiple Lines')

    
    plt.savefig(out_name, dpi=300)

if __name__ == "__main__":
    main()