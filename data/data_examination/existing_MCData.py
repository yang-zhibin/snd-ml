import ROOT
import os

def predict_class(df):
    print("predicting results")
    df = df.Define("ParticleType", """
        if (pdgCode == 12 || pdgCode == -12) return std::string("ve");
        else if (pdgCode == 14 || pdgCode == -14) return std::string("vm");
        else if (pdgCode == 16 || pdgCode == -16) return std::string("vt");
        else if (pdgCode == 112 ||pdgCode == -112 || pdgCode == 114 || pdgCode == -114 || pdgCode == 116 || pdgCode == -116) return std::string("NC");
        else if (pdgCode == 130 ||pdgCode == 310) return std::string("kaon");
        else if (pdgCode == 2112) return std::string("neutron");
        else if (pdgCode == 13 || pdgCode == -13) return std::string("muon");
        else return std::string("unknown");
        """)

    return df


# Neutrinos
def Neutrino_MCData():
    print('reading Neutrinos data...')
    tchain = ROOT.TChain("cbmsim")
    total_event = 0

    data_type = 'Neutrinos'
    #path = '/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000'
    path = '/eos/user/z/zhibin/sndData/converted/Neutrinos/'
    file_list = []
    file_ending = 'converted'

    for root, dirs, files in os.walk(path):
        for file_name in files:
            if 'converted' in file_name:
                file_list.append(os.path.join(root, file_name))
    print(file_list)
    print("number of files:",len(file_list))
    
    rdf = ROOT.RDataFrame('cbmsim', file_list)
    rdf = predict_class(rdf)
    original_events = rdf.Count().GetValue()
    ve_count = rdf.Filter(f'ParticleType=="ve"').Count().GetValue()
    vm_count = rdf.Filter(f'ParticleType=="vm"').Count().GetValue()
    vt_count = rdf.Filter(f'ParticleType=="vt"').Count().GetValue()
    nc_count = rdf.Filter(f'ParticleType=="NC"').Count().GetValue()

    print("Total neutrino MC events:", original_events)
    print(f've:{ve_count}, vm:{vm_count}, vt:{vt_count},NC:{nc_count},')

# Muon
def Muon_MCData():
    print('reading muon data...')
    tchain = ROOT.TChain("cbmsim")
    total_event = 0

    #muons_up_2.0 MC set of 2x2 m^2 scoring plane; 50M IP1 pp collisions simulated
    data_type = 'muons_up_2.0 '
    file_list = []
    path = '/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_up'
    file_ending = 'dig.root'

    for file_name in os.listdir(path):
        if file_name.endswith(file_ending):
            file_list.append(os.path.join(path, file_name))
    #print(file_list)

    total_event += reading_MCFiles(data_type, file_list)

    #muons_up_2.5 2.5x2.5 m^2 scoring plane; 10M IP1 pp collisions simulated
    data_type = 'muons_up_2.5 '
    file_path = '/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_up/scoring_2.5/sndLHC.Ntuple-TGeant4_150urad_1e7pr_dig.root'

    total_event += reading_MCFiles(data_type, file_path)


    #muons_down_2 2x2 m^2 scoring plane; 50M IP1 pp collisions simulated
    data_type = 'muons_down_2 '
    file_list = []
    path = '/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down'
    file_ending = 'dig.root'

    for file_name in os.listdir(path):
        if file_name.endswith(file_ending):
            file_list.append(os.path.join(path, file_name))

    total_event += reading_MCFiles(data_type, file_list)

    # muons_down_2_newGeo are MC set of 2x2 m^2 scoring plane; 50M IP1 pp collisions simulated
    data_type = 'muons_down_2_newGeo '
    file_path = '/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_2_newGeo/sndLHC.Ntuple-TGeant4_digCPP.root'

    total_event += reading_MCFiles(data_type, file_path)
    # muons_down_2.5 2.5x2.5 m^2 scoring plane; 10M IP1 pp collisions simulated
    data_type = 'muons_down_2.5 '
    file_path = '/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_2.5/sndLHC.Ntuple-TGeant4_-150urad_1e7pr_dig.root'

    total_event += reading_MCFiles(data_type, file_path)

    # muon_down_1.8 1.8x1.8 m^2 scoring plane; 50M IP1 pp collisions simulated, 
    #crossing angle matches precisely the LHC Run3 2022 conf and is -160urad; 
    #extended B filed map including the field outside the beam pipes, in the yoke of dipole 2 and quadruples 4-7
    data_type = 'muons_down_1.8 '
    file_path = '/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_1.8_Bfield/sndLHC.Ntuple-TGeant4-160urad_magfield_2022TCL6_muons_rock_5e7pr_digCPP.root'

    total_event += reading_MCFiles(data_type, file_path)

    #muon_down_1.8_4xstat
    data_type = 'muons_down_1.8_4xstat '
    file_path = '/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_1.8_Bfield_4xstat/sndLHC.Ntuple-TGeant4-160urad_magfield_2022TCL6_muons_rock_2e8pr.root'

    total_event += reading_MCFiles(data_type, file_path)

    #muon_down_1.8_4xstat_G4tridentBoost
    data_type = 'muons_down_1.8_4xstat_G4tridentBoost '
    file_path = '/eos/experiment/sndlhc/MonteCarlo/MuonBackground/muons_down/scoring_1.8_Bfield_4xstat_G4tridentBoost/sndLHC.Ntuple-TGeant4_boost1000.0-160urad_magfield_2022TCL6_muons_rock_2e8pr.root'

    total_event += reading_MCFiles(data_type, file_path)
    print("Total muon MC events:", total_event)

# Kaon
def Kaon_MCData():
    print('reading Kaon data...')
    tchain = ROOT.TChain("cbmsim")
    total_event = 0

    # Kaon on target
    data_type = 'Kaon_tgtarea'
    path = '/eos/experiment/sndlhc/users/marssnd/PGsim/Kaons'
    file_list = []
    subfolder_string='tgtarea'
    file_ending = '20240126_digCPP.root'

    break_flag = False
    for root, dirs, files in os.walk(path):
        if subfolder_string in root:   
            for file_name in os.listdir(root):
                if file_name.endswith(file_ending):
                    file_list.append(os.path.join(root, file_name))
#                    if(len(file_list)>100):
#                        break_flag = True   
#                if(break_flag):
#                    break         
#        if(break_flag):
#            break

    print("number of files:",len(file_list))

    total_evt = 0
    for file in file_list:
        f = ROOT.TFile(file, 'read')
        tree = f.Get('cbmsim')
        total_evt += tree.GetEntries()

    print("Total Kaon MC events:", total_evt)

# Neutron
def Neutron_MCData():
    print('reading Neutron data...')
    tchain = ROOT.TChain("cbmsim")
    total_event = 0

    # Kaon on target
    data_type = 'Neutron_tgtarea'
    path = '/eos/experiment/sndlhc/users/marssnd/PGsim/neutrons'
    file_list = []
    subfolder_string='tgtarea'
    file_ending = '20240126_digCPP.root'

    break_flag=False
    for root, dirs, files in os.walk(path):
        if subfolder_string in root:   
            for file_name in os.listdir(root):
                if file_name.endswith(file_ending):
                    file_list.append(os.path.join(root, file_name))    
#                    if(len(file_list)>100):
#                        break_flag = True   
#                if(break_flag):
#                    break         
#        if(break_flag):
#                    break   

    print("number of files:",len(file_list))

    total_evt = 0
    for file in file_list:
        if (file == '/eos/experiment/sndlhc/users/marssnd/PGsim/neutrons/neu_10_20_tgtarea/Ntuples/135/sndLHC.PG_2112-TGeant4_20240126_digCPP.root'):
            continue
        #print(file)
        f = ROOT.TFile(file, 'read')
        tree = f.Get('cbmsim')
        total_evt += tree.GetEntries()
    print("Total Neutron MC events:", total_evt)

def reading_MCFiles(data_type, file_list):
    print("data type:", data_type)
    rdf = ROOT.RDataFrame('cbmsim', file_list)
    original_events = rdf.Count().GetValue()
    print("    Original events:", original_events)
    
    cut = '    Digi_ScifiHits.GetEntries()  > 0 || Digi_MuFilterHits.GetEntries()  > 0'
    target_events = rdf.Filter(cut)
    selected_events = target_events.Count().GetValue()
    print("    Event has at least 1 hit:", selected_events)

    percentage_selected = (selected_events / original_events) * 100
    print("    {}%".format(percentage_selected))
    
    return selected_events

def real_data():



    # Path to the folder with ROOT files
    folder_path = "/eos/experiment/sndlhc/convertedData/physics/2023_reprocess"

    # List to hold all root files
    root_files = []


    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".root") and not(file_name.startswith("geo")):
                #print(file_name)
                root_files.append(os.path.join(root, file_name))

    # Check if there are any ROOT files in the folder
    if len(root_files) == 0:
        print("No ROOT files found in the folder.")
        exit(1)

    print('debug')

    # Create RDataFrame from all ROOT files
    rdf = ROOT.RDataFrame("cbmsim", root_files)

    # Count the total number of events in all files
    total_events = rdf.Count().GetValue()

    print(f"Total number of events: {total_events}")



    

if __name__ == '__main__':
    #Muon_MCData()
    Neutrino_MCData()
    #Kaon_MCData()
    #Neutron_MCData()
    #real_data()


