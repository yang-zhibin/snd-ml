
import os
import fnmatch
import ROOT

def print_tree_structure(file_path, tree_name):
    # Open the ROOT file
    root_file = ROOT.TFile(file_path, "READ")
    
    # Attempt to get the tree from the file
    tree = root_file.Get(tree_name)
    
    if tree:
        # Print the structure of the tree
        tree.Print()
    else:
        # Inform the user if the tree is not found
        print(f"Tree '{tree_name}' not found in the file '{file_path}'.")
    
    # Close the file to free resources
    root_file.Close()

def find_files(root_path, pattern):
    matches = []
    for root, dirs, files in os.walk(root_path):
        for filename in fnmatch.filter(files, pattern):
            if filename.endswith('.root'):
                matches.append(os.path.join(root, filename))
    return matches

def cal_eff(root_path):
    
    pattern = '*converted*'
    files_list = find_files(root_path, pattern)

    #print_tree_structure(files_list[0], "cbmsim")
    
    chain = ROOT.TChain("cbmsim")

    for file in files_list:
        chain.Add(file)

    # Create RDataFrame from TChain
    df = ROOT.RDataFrame(chain)
    
    print("-------",os.path.basename(os.path.normpath(root_path)),"------")

    # Example operation: count the entries
    total_count = df.Count().GetValue()
    print("Number of entries {:.2e}".format(total_count))

    # Filter the DataFrame to select events where stage1 is True
    stage1 = df.Filter("stage1")
    stage2 = df.Filter("stage2")
    count1 = stage1.Count().GetValue()
    count2 = stage2.Count().GetValue()

    eff1 = count1 / total_count
    eff2 = count2/total_count
    # Print the number of events where stage1 is True
    print("stage1, count:{}, eff:{:.4e}".format(count1, eff1) )
    print("stage2, count:{}, eff:{:.4e}".format(count2, eff2) )

    #runId = df.Take[int]("runId").GetValue()
    #uni_runid = set(runId)
    #print(uni_runid)

def read_cut_entry(file_path, n_cut, cuts, stage):
    #print(file_path, n_cut)
    file = ROOT.TFile(file_path, "READ")
    
    #print(file)
    for i in range(-1, n_cut-1):
        #print(i)
        cut_name = "numuCC_{}_Enu".format(i)
        #print(cut_names)
        hist = file.Get(cut_name)
        entry = hist.GetEntries()
        cuts[i+1] += entry
    
    file.Close()

    return entry




def cuts_eff(root_path):
    print('processing')

    total = 0
    stage1_count =0
    stage2_count =0
    s2 = 0
    stage1_cuts = [0]*9
    stage2_cuts = [0]*8
    count = 0
    #for partition in os.listdir(root_path):
    for partition in range(1,400):
        #print(partition)
        partition = str(partition)
        for file in os.listdir(os.path.join(root_path, partition)):
            #print(file)
            if "converted" in file:
                #print(file)
                f = ROOT.TFile(os.path.join(root_path, partition,file), 'read')
                tree = f.Get('cbmsim')
                total += tree.GetEntries()
                f.Close()
                
                df = ROOT.RDataFrame("cbmsim", os.path.join(root_path, partition,file))
                #print("display columns:",df.GetColumnNames())
                stage1_count += df.Sum("stage1").GetValue()
                stage2_count += df.Sum("stage2").GetValue()

            elif file.endswith('stage1.root'):
                read_cut_entry(os.path.join(root_path, partition,file), 9, stage1_cuts, 1)

            elif file.endswith('stage2.root'):
                s2 += read_cut_entry(os.path.join(root_path, partition,file), 8, stage2_cuts, 2)
            #print(file)
            #print("s2 count",stage2_count, s2) 

            
        count+=1
        #if (count>30):
        #    break
    print(total)
    print(stage1_count)
    print(stage2_count)
    print(stage1_cuts)
    print(stage2_cuts)

    cuts_eff = []
    cuts = stage1_cuts + stage2_cuts
    for i in range(len(cuts)-1):
        eff = cuts[i+1] / cuts[i]
        cuts_eff.append(eff)

    for i in range (len(cuts_eff)):
        cut_name = chr(i+65)
        print(cut_name, "eff:{:.3}".format(cuts_eff[i]))

    count1 = stage1_cuts[6]
    count2 = stage2_cuts[-1]
    eff1 = count1/cuts[0]
    eff2 = count2/cuts[0]
    print("A-F, pass cut:{}, eff:{:.4e}".format(count1, eff1) )
    print("A-P, pass cut:{}, eff:{:.4e}".format(count2, eff2) )
    print("boolean count{}, eff:{:.4e}".format(stage2_count, stage2_count/cuts[0]), )





def main():
    kaons_path = "/eos/user/z/zhibin/sndData/converted/kaons/neu_5_10_tgtarea/"
    neutron_path = "/eos/user/z/zhibin/sndData/converted/neutrons/neu_5_10_tgtarea/"
    nertrino_path = "/eos/user/z/zhibin/sndData/converted/Neutrinos/" 

    kaons_80_90 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_80_90_tgtarea"
    neutron_80_90 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_90_100_tgtarea"
    

    #kaon_

    #cal_eff(kaons_path)
    #cal_eff(neutron_path)
    #cal_eff(kaons_80_90)
    #cal_eff(neutron_80_90)
    #cal_eff(nertrino_path)
    path_list = [
    # "/eos/user/z/zhibin/sndData/converted/kaons/neu_5_10_tgtarea/",
    # "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_80_90_tgtarea",
    # "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_90_100_tgtarea",
    # "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_FTFP_BERT_80_90_tgtarea",
    # "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_FTFP_BERT_90_100_tgtarea",
    "/eos/user/z/zhibin/sndData/converted/neutrons/neu_5_10_tgtarea/",
    "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_90_100_tgtarea",
    "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_FTFP_BERT_80_90_tgtarea",
    "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_FTFP_BERT_90_100_tgtarea"
    "/eos/user/z/zhibin/sndData/converted/Neutrinos/",
    ]

    # for path in path_list:
    #     last_dir = os.path.basename(os.path.normpath(path))
    #     #print(last_dir)
    #     cal_eff(path)

    print(nertrino_path)

    cuts_eff(nertrino_path)

if __name__ == "__main__":
    main()

