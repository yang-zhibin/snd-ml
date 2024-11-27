import os

# Specify the directory containing the files
directory = '/afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/condor_scripts/output'
def process(data):
    error_entries = []
    error_patterns = ["no attribute"]
    for line in data.split("\n"):
        if any(pattern in line.lower() for pattern in error_patterns):
            error_entries.append(line)
    return error_entries
def read_failed_list():

    # Filter and process the files
    error_list = []
    for filename in os.listdir(directory):
        if filename.startswith('vm_eff') and filename.endswith('.log'):
            filepath = os.path.join(directory, filename)
            # Here you can add your processing code
            print(f'Processing file: {filename}')
            # For example, open the file and read its contents
            with open(filepath, 'r') as file:
                data = file.read()
                error_list.extend(process(data))

    error_list.sort()
   #print(error_list)
    with open('failed_list.txt', 'w') as f:
        for line in error_list:
            f.write(f"{line}\n")

def extract_path_fragment(line):
    start = line.find("converted") + len("converted/")
    end = line.find("/filtered_MC")
    if start > -1 and end > -1:
        path =  line[start:end]
    else:
        print("cant not extract info")

    parts = path.split('/')
    particle = parts[0]
    folder_part = parts[1].split('_')
    if "highstat" in folder_part:  
        folder = f"{folder_part[-4]}_{folder_part[-3]}_{folder_part[-2]}_{folder_part[-1]}"

    else:
        folder = f"{folder_part[-3]}_{folder_part[-2]}_{folder_part[-1]}"
        
    number = parts[2]
    return particle, folder, number
def pack_condor_line(particle,folder, partition):
    if (particle == 'kaons'):
        p_folder = 'Kaons'
        p_prefix = "K"
    else:
        p_folder = 'neutrons'
        p_prefix = "neu"
    
    input_dir = '/eos/experiment/sndlhc/users/marssnd/PGsim/'
    out_dir = '/eos/user/z/zhibin/sndData/converted/kaons/Filterv4_{}_'.format(particle)

    input_path = "{}/{}/{}_{}/Ntuples/".format(input_dir, p_folder,p_prefix, folder)
    out_path = '{}{}/'.format(out_dir, folder)

    path = '{}   {}   {}   {} {}'.format(particle, input_path, out_path, partition,partition)
    return path
def generate_condor_list():
    file_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/condor_scripts/failed_list.txt'
    failed_list = open(file_path, "r")
    data = failed_list.read()
    for line in data.split("\n"):
        #print(line)
        particle, folder, partition = extract_path_fragment(line)
        print(particle, folder, partition)
        condor_line = pack_condor_line(particle,folder, partition)
        file = open("/afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/condor_scripts/convert_failed.list", "a")
        file.write("{}\n".format(condor_line))



def main():
    generate_condor_list()
    
if __name__ == "__main__":
    main()