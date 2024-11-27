
import os
import fnmatch
import ROOT
from argparse import ArgumentParser

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
    #print(file_path)
    file = ROOT.TFile(file_path, "READ")
    
    #print(file)
    for i in range(-1, n_cut-1):
        #print(i)
        cut_name = "numuCC_{}_Enu".format(i)
        #print(cut_name)
        hist = file.Get(cut_name)
        try:
            entry = hist.GetEntries()
        except AttributeError:
            print("No attribute GetEntries file", file_path)
            file = open("/afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/condor_failed_list.txt", "a")
            file.write("{}\n".format(file_path))
            return 0

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
    for partition in os.listdir(root_path):
    #for partition in range(30,2800):
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
    print("total",total)
    print("stage1_count",stage1_count)
    print("stage2_count",stage2_count)
    print("stage1_cuts",stage1_cuts)
    print("stage2_cuts",stage2_cuts)

    cuts_eff = []
    cuts = stage1_cuts + stage2_cuts
    for i in range(len(cuts)-1):
        if cuts[i] == 0:
            cuts_eff.append(float('inf'))
        else:
            eff = cuts[i+1] / cuts[i]
            cuts_eff.append(eff)

    for i in range (len(cuts_eff)):
        cut_name = chr(i+65)
        print(cut_name, "eff:{:.3}".format(cuts_eff[i]))

    count1 = stage1_cuts[6]
    count2 = stage2_cuts[-1]
    if (cuts[0]==0):
        eff1 = float('inf')
        eff2 = float('inf')
    else:
        eff1 = count1/cuts[0]
        eff2 = count2/cuts[0]
    bool_eff = stage2_count/total
    print("A-F, pass cut:{}, eff:{:.4e}".format(count1, eff1) )
    print("A-P, pass cut:{}, eff:{:.4e}".format(count2, eff2) )
    print("boolean count{}, eff:{:.4e}".format(stage2_count, bool_eff), )


def process():
    nertrino_path = "/eos/user/z/zhibin/sndData/converted/Neutrinos/" 

    kaons_5_10 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_5_10_tgtarea"
    kaons_5_10_highstat = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_5_10_tgtarea_highstat"
    kaons_10_20 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_10_20_tgtarea"
    kaons_20_30 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_20_30_tgtarea"
    kaons_30_40 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_30_40_tgtarea"
    kaons_40_50 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_40_50_tgtarea"
    kaons_50_60 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_50_60_tgtarea"
    kaons_60_70 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_60_70_tgtarea"
    kaons_70_80 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_70_80_tgtarea"
    kaons_80_90 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_80_90_tgtarea"
    kaons_90_100 = "/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_kaons_90_100_tgtarea"

    neutrons_5_10 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_5_10_tgtarea"
    neutrons_5_10_highstat = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_5_10_tgtarea_highstat"
    neutrons_10_20 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_10_20_tgtarea"
    neutrons_20_30 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_20_30_tgtarea"
    neutrons_30_40 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_30_40_tgtarea"
    neutrons_40_50 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_40_50_tgtarea"
    neutrons_50_60 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_50_60_tgtarea"
    neutrons_60_70 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_60_70_tgtarea"
    neutrons_70_80 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_70_80_tgtarea"
    neutrons_80_90 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_80_90_tgtarea"
    neutrons_90_100 = "/eos/user/z/zhibin/sndData/converted/neutrons/Filterv4_neutrons_90_100_tgtarea"



    cuts_eff(kaons_5_10_highstat)
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

    #print(nertrino_path)

    #cuts_eff(nertrino_path)

def cal_from_csv():
    csv_file = '/eos/user/z/zhibin/sndData/converted/val_files.csv'
    df = pd.read_csv(file,header=None,names=['path', 'n_event'] )

    for file_path in df['path']:
        df = ROOT.RDataFrame("cbmsim", file_path)
        #print("display columns:",df.GetColumnNames())
        stage1_count += df.Sum("stage1").GetValue()
        stage2_count += df.Sum("stage2").GetValue()

def main(args):
    cuts_eff(args.root)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--root", dest="root", required=True)
    args = parser.parse_args()
    
    main(args)

