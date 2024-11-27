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

def check_with_TFile():

    file_path = "/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/0/sndLHC.Genie-TGeant4_digCPP.root"

    f = ROOT.TFile(file_path, 'read')
    tree = f.Get('cbmsim')

    #print_tree_structure(file_path, 'cbmsim')

    for event in tree:
        hit2MC =event.Digi_ScifiHits2MCPoints
        scifi_hits =event.Digi_ScifiHits
        MCTrack = event.MCTrack

        #print(dir(MCTrack))
        len_track = MCTrack.GetEntries()
        print("length of scifiHits",len(scifi_hits))
        print("length of MCTrack",len(MCTrack))
        #Print(type(MCTrack))
        #print(dir(MCTrack))
        for track in MCTrack:
            #print(track)
            #print(type(track))
            #print(dir(track))
            break

        count=0
        for aHit in scifi_hits:
            #print(count, "-------------")
            detID = aHit.GetDetectorID()
            #print("detId", detID)
            #print("hit to mc")
            #print(hit2MC[0].wList(detID))
            MCParticleID = hit2MC[0].wList(detID)
            #print(type(MCParticleID))
            #print(len(MCParticleID))

            for p in MCParticleID:
                #print(p)
                #print(type(p))
                trackId = p[0]
                ratio = p[1]

                #print(pid, ratio)
                #trcak = MCTrack[pid]
                #print(track)
            if trackId>len_track:
                print(count, pid, len_track)
            count+=1
            #if(count>0):
            #    break
        #print("length of h2mc",len(hit2MC))
        #print("length of hits", len(scifi_hits))

        #print("methods in Digi_ScifiHits2MCPoints: ")
        #print(dir(hit2MC))
        #print(dir(hit2MC))

        # print("Dumping of Digi_ScifiHits2MCPoints")
        # hit2MC.Dump()

        # print(type(hit2MC))
        # print(type(hit2MC[0]))
        
        break

        
        

if __name__ == "__main__":
    print("checking hits2MCPoints")
    check_with_TFile()

    