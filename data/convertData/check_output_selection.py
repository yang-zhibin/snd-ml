import ROOT
import os
import numpy as np

def read_neutrino(path):
    pass

def read_real_data(path):
    pass
   
def main():
    neutrino_path = '/eos/user/z/zhibin/sndData/converted/Neutrinos_v2/'
    real_data_path = '/eos/user/z/zhibin/sndData/converted/real_muon/2023_reprocess/'
    rdf_neutrino = read_neutrino(neutrino_path)
    rdf_real_data = read_real_data(real_data_path)

    #apply cuts and save




if __name__ == "__main__":
    main()


