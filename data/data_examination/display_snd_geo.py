import ROOT
import SndlhcGeo
import matplotlib.pyplot as plt
import numpy as np

ROOT.ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

def snd_transformation(x_old, y_old, z_old):
    # Define the rotation matrix R

    R = np.array([
        [0.38646054170418, -0.92228282252868, -0.00653031195550],
        [0.01398585186500, -0.00121946161748,  0.99990144957439],
        [-0.92219989462877, -0.38651378782896,  0.01242763713624]
    ])

    # Define the translation vector T
    T = np.array([1664.3132966, -2386.2683455, 2569.2604796])

    # Define the old coordinate vector
    old_coords = np.array([x_old, y_old, z_old])

    # Apply the transformation: new_coords = R * old_coords + T
    new_coords = np.dot(R, old_coords) 

    return new_coords

def setup_geometry(geo_file):
    """Initialize and return the geometry configurations."""
    geo = SndlhcGeo.GeoInterface(geo_file)
    lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
    lsOfGlobals.Add(geo.modules['Scifi'])
    lsOfGlobals.Add(geo.modules['MuFilter'])
    return geo

def check_detId():
    geo_file = "/eos/experiment/sndlhc/convertedData/physics/2023_reprocess/geofile_sndlhc_TI18_V4_2023.root"

    geo = setup_geometry(geo_file)
    Scifi = geo.snd_geo.Scifi
    MuFilter = geo.snd_geo.MuFilter
    A, B = ROOT.TVector3(), ROOT.TVector3()

    for i in range(100):
        print(f'detId:{i}')
        geo.modules['Scifi'].GetSiPMPosition(i, A, B)

        point1 = [A.x(), A.y(), A.z()]
        point2 = [B.x(), B.y(), B.z()]
        print(point1, point2)
        if i > 10:
            break


def main():
    geo_file = "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18.root"

    geo = setup_geometry(geo_file)
    Scifi = geo.snd_geo.Scifi
    MuFilter = geo.snd_geo.MuFilter
    #Emulsion = snd_geo.modules['Emulsion']

    #print(dir(geo.sGeo))
    scifi1 = geo.sGeo.GetVolume('ScifiVolume1_1000000')
    print(dir(scifi1))
   #print((scifi1.GetShape()))
    # Define each set of points

    # Set labels
    
    points = [Scifi['Xpos0']/1000, Scifi['Ypos0']/1000,Scifi['Zpos1']/1000]
    new_point = snd_transformation(points[0],points[1],points[2])
    print(points)
    print(new_point)
    
    # Draw the detector
    #Scifi.Draw("ogl")  # Use "ogl" for OpenGL 3D rendering; omit or adjust for 2D views
    #canvas.Update() 
    #canvas.SaveAs("plots/scifi_detector_visualization.png") 
    #print(Emulsion)
    #print(modules)

    #print(snd_geo)
    #print(Scifi.Draw())
    #print(type(Scifi))
    print(dir(Scifi))
    #print(dir(MuFilter))

    #print(dir(snd_geo))



if __name__ == '__main__':

    main()
    #check_detId()