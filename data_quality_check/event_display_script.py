import os

#event_dict = {}
event_list = []

geo_file = "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V4_2023.root"

#with open("selected_events", "r") as f :
with open("followingEventList", "r") as f :
    lines = f.readlines()
    for line in lines :
        line = line.split()
        run = int(line[2])
        event = int(line[3])
        file_number = event//1000000
        local_event_number = event%1000000

        event_list.append((run, file_number, local_event_number))
        
for event in event_list :
    file_name = "run_{:06d}/sndsw_raw-{:04d}.root".format(event[0], event[1])

    p = "/eos/experiment/sndlhc/convertedData/physics/2022/"
    p_raw = "/eos/experiment/sndlhc/raw_data/physics/2022/"
    if not os.path.isfile(p+file_name) :
        p = "/eos/experiment/sndlhc/convertedData/physics/2023_reprocess/"
        p_raw = "/eos/experiment/sndlhc/raw_data/physics/2023/"
    if not os.path.isfile(p+file_name) :
        print("Couldn't find run {}, file {}".format(run_file_number[0], run_file_number[1]))
        continue

#    loopEventsLine  = "python -i $SNDSW_ROOT/shipLHC/scripts/2dEventDisplay.py -g {} -f {} -p {} -praw {} <<EOF\n".format(geo_file, file_name, p, p_raw)
    loopEventsLine  = "python -i  /afs/cern.ch/work/c/cvilela/public/SND_Nov_2023/cheatgit/sndsw/shipLHC/scripts/2dEventDisplay.py -g {} -f {} -p {} -praw {} <<EOF\n".format(geo_file, file_name, p, p_raw)
    loopEventsLine += "loopEvents(save = True, start = {}, hitColour = \"q\")\n".format(event[2])
    loopEventsLine += "EOF"
    
    print(loopEventsLine)
    os.system(loopEventsLine)