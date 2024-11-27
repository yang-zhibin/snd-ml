import ROOT
import os
from array import array
import pandas as pd
from argparse import ArgumentParser

def gnerate_condor_list():
    file_path="/afs/cern.ch/user/z/zhibin/work/snd-ml/data/data_examination/file_paths.csv"
    df = pd.read_csv(file_path)

    df['run_number'] = df['path'].str.extract(r'run_(\d{6})').astype(int)
    grouped = df.groupby('partition')['run_number'].agg(['min', 'max']).reset_index()

    # Printing the results for each group
    for _, row in grouped.iterrows():
        print(f"{row['min']} {row['max']}")


def check_root_files():
    file_path="/eos/user/z/zhibin/sndData/converted/real_muon/2023_reprocess/2023_reprocess.csv"
    df = pd.read_csv(file_path)

    total_count_sum = df['count'].sum()
    print(f'total_count_sum:{total_count_sum:.3e}')

    # Define the threshold for each partition
    threshold = 4e6

    # Initialize cumulative sum and partition counter
    cumulative_sum = 0
    partition = 1
    partitions = []
    

    # Loop over rows and assign partitions
    for count in df['count']:
        if cumulative_sum + count > threshold:
            partition += 1  # Start a new partition
            cumulative_sum = 0
        cumulative_sum += count
        partitions.append(partition)

    # Assign partitions to DataFrame
    df['partition'] = partitions
    df['run_number'] = df['path'].str.extract(r'run_(\d{6})').astype(int)

    df.to_csv(file_path, index=False)

    # Display the DataFrame with partitions
    print(df)


def read_root(folder_path):
    # Lists to hold file paths and entry counts
    root_files = []
    root_entries = []
    
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".root") and not file_name.startswith("geo"):
                root_file = os.path.join(root, file_name)
                rdf = ROOT.RDataFrame("cbmsim", root_file)
                entries = rdf.Count().GetValue()
                root_files.append(root_file)
                root_entries.append(entries)

        # Progress check for every 100 files
        if len(root_files) % 100 == 0:
            print(f"Processed {len(root_files)} files...")

    # Check if there are any ROOT files in the folder
    if len(root_files) == 0:
        print("No ROOT files found in the folder.")
        return  # Use return to exit the function gracefully

    # Creating a DataFrame with columns 'path' and 'count'
    df = pd.DataFrame({"path": root_files, "count": root_entries})

    # Save to a CSV file
    df.to_csv("file_paths.csv", index=False)
    print("List of ROOT files saved in file_paths.csv")
    # Create RDataFrame from all ROOT files
    #rdf = ROOT.RDataFrame("cbmsim", root_files)

    # Count the total number of events in all files
    #total_events = rdf.Count().GetValue()

    #print(f"Total number of events: {total_events}")

    #return rdf


def simple_selection(rdf,file_path):
    print("selecting muon like data")
    #select 1 veto 1 scif, 1 muon hits 
    #print(dir(rdf))
    #print(rdf.GetColumnNames())
    total_events = rdf.Count().GetValue()

    rdf = rdf.Filter('Digi_MuFilterHits.GetEntries()>1')
    n_filter_events = rdf.Count().GetValue()
    print(f'filter with 2 muFilter hits: {n_filter_events}, ratio: {n_filter_events/total_events:.3e}')
    
    
    rdf.Snapshot('cbmsim', file_path)

    return file_path
    

def count_station_hit(file_path, output_file_path):
    print('counting station hit')
    input_file = ROOT.TFile(file_path, "READ")
    tree = input_file.Get("cbmsim")  # Replace with the actual tree name

    # Create a new ROOT file to save the updated tree

    output_file = ROOT.TFile(output_file_path, "CREATE")

    # Clone the tree structure from the original file to keep existing branches
    new_tree = tree.CloneTree(0)  # Set 0 to create an empty cloned tree

    # Initialize vectors for each new feature
    veto1_feature = array('i', [0])
    veto2_feature = array('i', [0])
    ds_features = [array('i', [0]) for _ in range(4)]  # ds1 to ds4
    us_features = [array('i', [0]) for _ in range(5)]  # us1 to us5
    scifi_features = [array('i', [0]) for _ in range(5)]  # scifi1 to scifi5

    scifi_avg_ver_feature = array('f', [-1])
    scifi_avg_hor_feature = array('f', [-1])
    DS_avg_ver_feature = array('f', [-1])
    DS_avg_hor_feature = array('f', [-1])
    

    # Create branches for each new feature
    veto1_branch = new_tree.Branch("veto1", veto1_feature, "veto1/I")
    veto2_branch = new_tree.Branch("veto2", veto2_feature, "veto2/I")
    ds_branches = [new_tree.Branch(f"ds{i+1}", ds_features[i], f'ds{i+1}/I') for i in range(4)]
    us_branches = [new_tree.Branch(f"us{i+1}", us_features[i], f'us{i+1}/I') for i in range(5)]
    scifi_branches = [new_tree.Branch(f"scifi{i+1}", scifi_features[i], f'scifi{i+1}/I') for i in range(5)]
    scifi_avg_hor_branches = new_tree.Branch("scifi_avg_hor", scifi_avg_ver_feature, "scifi_avg_hor/F")
    scifi_avg_ver_branches = new_tree.Branch("scifi_avg_ver", scifi_avg_hor_feature, "scifi_avg_ver/F")
    DS_avg_hor_branches = new_tree.Branch("DS_avg_hor", DS_avg_ver_feature, "DS_avg_hor/F")
    DS_avg_ver_branches = new_tree.Branch("DS_avg_ver", DS_avg_hor_feature, "DS_avg_ver/F")


    unique_stations = set()

    A, B = ROOT.TVector3(), ROOT.TVector3()

    # Loop over the events and fill the new branches
    for event in tree:
        # Initialize counters
        scifi_counts = [0] * 5  # scifi1 to scifi5
        veto1 = veto2 = 0
        ds_counts = [0] * 4  # ds1 to ds4
        us_counts = [0] * 5  # us1 to us5

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
            detID = aHit.GetDetectorID()
            station = detID // 1000000

            if 1 <= station <= 5:
                scifi_counts[station - 1] += 1
                #unique_stations.add(station)
            
            # Calculate average positions
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
            detID = aHit.GetDetectorID()
            detType = aHit.GetSystem()
            station = (detID // 1000) % 10

            if detType == 1 and station == 0:
                veto1 += 1
            elif detType == 1 and station == 1:
                veto2 += 1
                
            elif detType == 3 and 0 <= station <= 3:
                ds_counts[station ] += 1
            elif detType == 2 and 0 <= station <= 4:
                us_counts[station ] += 1

            # DS hit averaging, considering only system '3' which is downstream
            if aHit.GetSystem() == 3:
                x = detID % 1000
                if aHit.isVertical():
                    DS_avg_ver += x
                    DS_n_ver += 1
                else:
                    DS_avg_hor += x
                    DS_n_hor += 1

        # Update vector values for each feature and fill branches
        veto1_feature[0] = veto1
        veto2_feature[0] = veto2

        for i in range(5):
            scifi_features[i][0] = scifi_counts[i]
            us_features[i][0] = us_counts[i]
            

        for i in range(4):
            ds_features[i][0] = ds_counts[i]

        # Fill the tree with the current event data
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

        scifi_avg_ver_feature[0] = scifi_avg_ver
        scifi_avg_hor_feature[0] = scifi_avg_hor
        DS_avg_ver_feature[0] = DS_avg_ver
        DS_avg_hor_feature[0] = DS_avg_hor
        #print(scifi_avg_ver, scifi_avg_hor, DS_avg_ver, DS_avg_hor)
        new_tree.Fill()


    # Display unique station values
    print("Unique station values:", unique_stations)

    # Write the new tree to the output file and close both files
    new_tree.Write()
    output_file.Close()
    input_file.Close()


    return output_file_path

def select_muon_like(rdf):
    
    total_events = rdf.Count().GetValue()
    print(f'total: {total_events:.3e}')
    #print(rdf)
    data_dict = rdf.AsNumpy(columns=["veto2"])
    print(data_dict["veto2"])  

    veto_and_us1 = rdf.Filter("(veto1 > 0 && veto2 > 0) && us1 > 0 && us1 < 2")
    veto_and_us12 = veto_and_us1.Filter("(us2 > 0 && us2 < 2)")
    veto_and_us13 = veto_and_us12.Filter("(us3 > 0 && us3 < 2)")
    veto_and_us14 = veto_and_us13.Filter("(us4 > 0 && us4 < 2)")
    veto_and_us15 = veto_and_us14.Filter("(us5 > 0 && us5 < 2)")
    veto_and_us15_ds1 = veto_and_us15.Filter("(ds1 > 0 && ds1 < 4)")
    veto_and_us15_ds12 = veto_and_us15_ds1.Filter("(ds2 > 0 && ds2 < 4)")
    veto_and_us15_ds13 = veto_and_us15_ds12.Filter("(ds3 > 0 && ds3 < 4)")
    veto_and_us15_ds14 = veto_and_us15_ds13.Filter("(ds4 > 0 && ds4 < 4)")

    veto_and_us1_ds4 = rdf.Filter("(veto1 > 0 || veto2 > 0) && us1 > 0 && us1 < 2 && (ds4 > 0 && ds4 < 2)")
    veto_and_us1_ds43 = veto_and_us1_ds4.Filter("ds3 > 0 && ds3 < 2")
    veto_and_us1_ds42 = veto_and_us1_ds43.Filter("ds2 > 0 && ds2 < 2")
    veto_and_us1_ds41 = veto_and_us1_ds42.Filter("ds1 > 0 && ds2 < 2")

    veto_and_us15_ds14_scfifi35 = veto_and_us15_ds14.Filter("(scifi5 > 0 && scifi5 < 5 && scifi4 > 0 && scifi4 < 5 && scifi3 > 0 && scifi3 < 5 )")

    


    v_us1 = veto_and_us1.Count().GetValue()
    v_us12 = veto_and_us12.Count().GetValue()
    v_us13 = veto_and_us13.Count().GetValue()
    v_us14 = veto_and_us14.Count().GetValue()
    v_us15 = veto_and_us15.Count().GetValue()
    v_us15_ds1 = veto_and_us15_ds1.Count().GetValue()
    v_us15_ds12 = veto_and_us15_ds12.Count().GetValue()
    v_us15_ds13 = veto_and_us15_ds13.Count().GetValue()
    v_us15_ds14 = veto_and_us15_ds14.Count().GetValue()
    v_us1_ds4 = veto_and_us1_ds4.Count().GetValue()
    v_us1_ds43 = veto_and_us1_ds43.Count().GetValue()
    v_us1_ds42 = veto_and_us1_ds42.Count().GetValue()
    v_us1_ds41 = veto_and_us1_ds41.Count().GetValue()
    v_us15_ds14_scifi35 = veto_and_us15_ds14_scfifi35.Count().GetValue()

    # Print all counts together
    print(
        f'veto_and_us1 count: {v_us1:.3e}, ratio: {v_us1/total_events:.3e}\n'
        f'veto_and_us12 count: {v_us12:.3e}, ratio: {v_us12/total_events:.3e}\n'
        f'veto_and_us13 count: {v_us13:.3e}, ratio: {v_us13/total_events:.3e}\n'
        f'veto_and_us14 count: {v_us14:.3e}, ratio: {v_us14/total_events:.3e}\n'
        f'veto_and_us15 count: {v_us15:.3e}, ratio: {v_us15/total_events:.3e}\n'
        f'veto_and_us15_ds1 count: {v_us15_ds1:.3e}, ratio: {v_us15_ds1/total_events:.3e}\n'
        f'veto_and_us15_ds12 count: {v_us15_ds12:.3e}, ratio: {v_us15_ds12/total_events:.3e}\n'
        f'veto_and_us15_ds13 count: {v_us15_ds13:.3e}, ratio: {v_us15_ds13/total_events:.3e}\n'
        f'veto_and_us15_ds14 count: {v_us15_ds14:.3e}, ratio: {v_us15_ds14/total_events:.3e}\n'
        f'veto_and_us1_ds4 count: {v_us1_ds4:.3e}, ratio: {v_us1_ds4/total_events:.3e}\n'
        f'veto_and_us1_ds43 count: {v_us1_ds43:.3e}, ratio: {v_us1_ds43/total_events:.3e}\n'
        f'veto_and_us1_ds42 count: {v_us1_ds42:.3e}, ratio: {v_us1_ds42/total_events:.3e}\n'
        f'veto_and_us1_ds41 count: {v_us1_ds41:.3e}, ratio: {v_us1_ds41/total_events:.3e}\n'
        f'v_us15_ds14_scifi35 count: {v_us15_ds14_scifi35:.3e}, ratio: {v_us15_ds14_scifi35/total_events:.3e}\n'
    )
    veto_and_us15_ds14_scfifi23 = veto_and_us15_ds14.Filter("(scifi5 > 0 && scifi5 < 5 && scifi4 > 0 && scifi4 < 5 && scifi3 > 0 && scifi3 < 5 )")

    veto_and_us15_ds14.Snapshot('cbmsim', 'data/veto_and_us15_ds14.root')
    scifi1 = veto_and_us15_ds14.Filter('scifi1<1')
    scifi1_count = scifi1.Count().GetValue()

    scifi1.Snapshot('cbmsim', 'data/scifi1.root')

    print(f'scifi1_count: {scifi1_count}, ratio: {scifi1_count/v_us15_ds14:.3e}')

    scifi2 = veto_and_us15_ds14.Filter('scifi2<1')
    scifi2_count = scifi2.Count().GetValue()
    scifi2.Snapshot('cbmsim', 'data/scifi2.root')

    print(f'scifi2_count: {scifi2_count}, ratio: {scifi2_count/v_us15_ds14:.3e}')

def select_muon_like_condor(rdf, out_dir,partition):
    
    total_events = rdf.Count().GetValue()
    print(f'total: {total_events:.3e}')
    if(total_events == 0):
        print("Total events of simple selection is 0. Exiting.")
        raise SystemExit("No events to process after simple selection.")
    #print(rdf)
    #data_dict = rdf.AsNumpy(columns=["veto2"])
    #print(data_dict["veto2"])  

    #fiducial selection
    #fiducial = rdf.Filter()


    veto_and_us15_ds14 = rdf.Filter(
        """
        (veto1 > 0 || veto2 > 0) && 
        us1 > 0 && us1 < 2 && 
        us2 > 0 && us2 < 2 &&
        us3 > 0 && us3 < 2 &&
        us4 > 0 && us4 < 2 &&
        us5 > 0 && us5 < 2 &&
        ds1 > 0 && ds1 < 4 &&
        ds2 > 0 && ds2 < 4 &&
        ds3 > 0 && ds3 < 4 &&
        ds4 > 0 && ds4 < 4
        """
    )
    veto_and_us15_ds14_scfifi1345 = veto_and_us15_ds14.Filter("(scifi5 > 0 && scifi5 < 5 && scifi4 > 0 && scifi4 < 5 && scifi3 > 0 && scifi3 < 5 && scifi1 > 0 && scifi1 < 5)")
    veto_and_us15_ds14_scfifi2345 = veto_and_us15_ds14.Filter("(scifi5 > 0 && scifi5 < 5 && scifi4 > 0 && scifi4 < 5 && scifi3 > 0 && scifi3 < 5 && scifi2 > 0 && scifi2 < 5)")

    v_us15_ds14 = veto_and_us15_ds14.Count().GetValue()
    v_us15_ds14_scifi1345 = veto_and_us15_ds14_scfifi1345.Count().GetValue()
    v_us15_ds14_scifi2345 = veto_and_us15_ds14_scfifi2345.Count().GetValue()

    # Print all counts together
    v_us15_ds14_scifi1345_file = f'{out_dir}/veto_and_us15_ds14_scfifi1345_partition{partition}.root'
    veto_and_us15_ds14_scfifi1345.Snapshot('cbmsim', v_us15_ds14_scifi1345_file)



    v_us15_ds14_scifi2345_file = f'{out_dir}/veto_and_us15_ds14_scfifi2345_partition{partition}.root'
    veto_and_us15_ds14_scfifi2345.Snapshot('cbmsim', v_us15_ds14_scifi2345_file)
    
    
    veto_and_us15_ds14_file = f'{out_dir}/veto_and_us15_ds14_partition{partition}.root'
    veto_and_us15_ds14.Snapshot('cbmsim', veto_and_us15_ds14_file)

    scifi_plane1 = veto_and_us15_ds14_scfifi2345.Filter('scifi1<1')
    scifi1_count = scifi_plane1.Count().GetValue()

    scifi_plane1_file = f'{out_dir}/scifi_plane1_partition{partition}.root'
    scifi_plane1.Snapshot('cbmsim', scifi_plane1_file)

    

    scifi_plane2 = veto_and_us15_ds14_scfifi1345.Filter('scifi2<1')
    scifi2_count = scifi_plane2.Count().GetValue()
    scifi_plane2_file = f'{out_dir}/scifi_plane2_partition{partition}.root'
    scifi_plane2.Snapshot('cbmsim', scifi_plane2_file)

    print(f'veto_and_us15_ds14 count: {v_us15_ds14:.3e}, ratio: {v_us15_ds14/total_events:.3e}\n'
          f'v_us15_ds14_scifi1345 count: {v_us15_ds14_scifi1345:.3e}, ratio: {v_us15_ds14_scifi1345/total_events:.3e}\n'
          f'v_us15_ds14_scifi2345 count: {v_us15_ds14_scifi2345:.3e}, ratio: {v_us15_ds14_scifi2345/total_events:.3e}\n')
    print(f'scifi1_count: {scifi1_count}, ratio: {scifi1_count/v_us15_ds14_scifi2345:.3e}')
    print(f'scifi2_count: {scifi2_count}, ratio: {scifi2_count/v_us15_ds14_scifi1345:.3e}')
    

def main():
    folder_path = "/eos/experiment/sndlhc/convertedData/physics/2023_reprocess"
    eos_path ='/eos/user/z/zhibin/sndData/converted/veto_ineff/'
    output_file_path = f'{eos_path}/hit_counted.root'
    file_path = f'{eos_path}/simple_filtered.root'
    rdf = read_root(folder_path)
    
    simple_selection(rdf, file_path)
    count_station_hit(file_path,output_file_path)
    #rdf = ROOT.RDataFrame("cbmsim", output_file_path)
    #select_muon_like(rdf)

def condor_main():

    parser = ArgumentParser()
    
    #parser.add_argument("-m", "--recoMuon_path", dest="recoMuon_path", help="reco muon data path", required=True)
    parser.add_argument("-p", "--partition", dest="partition",type=int, help="partition", required=True)
    parser.add_argument("-o", "--outDir", dest="out_dir", help="output directory", required=True)

    args = parser.parse_args()

    df = pd.read_csv("/afs/cern.ch/user/z/zhibin/work/snd-ml/data/data_examination/file_paths.csv")

    # Convert the DataFrame to a list
    file_partition = df[df['partition'] == args.partition]

    root_files = file_partition["path"].tolist()

    filter_file = f'{args.out_dir}/simple_filtered_{args.partition}.root'
    hit_counted_file = f'{args.out_dir}/hit_counted_{args.partition}.root'

    rdf = ROOT.RDataFrame("cbmsim", root_files)
    total_events = rdf.Count().GetValue()
    print(f"Total number of events: {total_events:.3e}")

    simple_selection(rdf, filter_file)
    count_station_hit(filter_file,hit_counted_file)

    rdf = ROOT.RDataFrame("cbmsim", hit_counted_file)
    select_muon_like_condor(rdf, args.out_dir, args.partition)

def check_scifi_ineff():
    folder_path = '/eos/user/z/zhibin/sndData/converted/veto_ineff/'

    # Initialize an empty list to store matching file names
    hit_count_files = []

    # Loop through the folder
    for file_name in os.listdir(folder_path):
        # Check if the file name starts with 'hit_count'
        if file_name.startswith('hit_count'):
            # Add the file to the list
 
            hit_count_files.append(os.path.join(folder_path, file_name))

    # Print the list of matching files
    print(hit_count_files)

    hit_rdf = ROOT.RDataFrame("cbmsim", hit_count_files)

    select_muon_like(hit_rdf)


if __name__ == '__main__':

    #folder_path = "/eos/experiment/sndlhc/convertedData/physics/2023_reprocess/"
    #read_root(folder_path)
    check_root_files()
    #gnerate_condor_list()
    #main()
   # condor_main()
    # out_dir = '/eos/user/z/zhibin/sndData/converted/veto_ineff/'
    # partition = 20
    # rdf  = ROOT.RDataFrame("cbmsim", '/eos/user/z/zhibin/sndData/converted/veto_ineff/hit_counted_20.root')
    # select_muon_like_condor(rdf, out_dir, partition)

    #check_scifi_ineff()