import uproot
import os

def partition_list(file_list, ratios):
    #print(len(file_list))
    total = sum(ratios)
    lengths = [len(file_list) * r // total for r in ratios]
    # Adjust the last partition to account for any rounding errors.
    lengths[-1] += len(file_list) - sum(lengths)
    from itertools import accumulate
    start_indexes = [0] + list(accumulate(lengths[:-1]))
    return [file_list[start:start + length] for start, length in zip(start_indexes, lengths)]


def GetEventList(paths, partition, out_path):
    print("prepare event list for training")
    print(paths, partition) #partition = [8,1,1]

    train_Flist, validation_Flist, test_Flist = [], [], []

    for root_path in paths:
        print(f"Processing files in: {root_path}")
        file_list = []
        event_list = []

        for root, dirs, files in os.walk(root_path):
            for filename in files:
                if filename.endswith('.root') and "converted" in filename:
                    file_path = os.path.join(root, filename)
                    try:
                        with uproot.open(file_path) as Rfile:
                            tree = Rfile["cbmsim"]
                            n_evt = tree.num_entries
                        event_list.append(n_evt)
                        file_list.append(file_path)
                    except KeyError:
                        print(f"Error: 'cbmsim' tree not found in {filename}.")
                    except Exception as e:
                        print(f"Error opening {filename}: {e}")

        # Check if lists have files and events, then partition them
        if file_list:
            train_files, validation_files, test_files = partition_list(file_list, partition)
            train_evts, validation_evts, test_evts = partition_list(event_list, partition)
            train_Flist.extend(zip(train_files, train_evts))
            validation_Flist.extend(zip(validation_files, validation_evts))
            test_Flist.extend(zip(test_files, test_evts))

            print(f"Partition sizes - Train: {len(train_files)}, Validation: {len(validation_files)}, Test: {len(test_files)}")
            print(f"Event counts - Train: {len(train_evts)}, Validation: {len(validation_evts)}, Test: {len(test_evts)}")
        else:
            print("No valid files found in this directory.")        
            
    # Save the results to text files for each partition
    for category, data in zip(["train", "validation", "test"], [train_Flist, validation_Flist, test_Flist]):
        filename = f"{out_path}/{category}_files.csv"
        with open(filename, 'w') as file:
            for file_path, events in data:
                file.write(f"{file_path}, {events}\n")
        print(f"Saved {category} partition data to {filename}")
        
        #print(train_Elist, validation_Elist, test_Elist)


    # partition_names = ['train', 'validation', 'test']
    # for partition_name, files in zip(partition_names, [train_list, validation_list, test_list]):
    #     # Construct the file path for the output text file.
    #     file_name = f"{out_path}/{partition_name}_list.txt"
    #     # Write the list of files to the text file.
    #     with open(file_name, 'w') as file:
    #         for file_path in files:
    #             file.write(file_path + '\n')
    #     print(f"Saved {partition_name} files to {file_name}")
    # #print(file_list)
    

if __name__ == "__main__":
    pass
    #main()