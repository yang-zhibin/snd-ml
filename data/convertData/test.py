import ROOT
import random

def creat_root():

    # Create file and tree
    file = ROOT.TFile("example.root", "RECREATE")
    tree = ROOT.TTree("tree", "A simple test tree")

    #C structure to hold tree variables
    #cmd = "struct Id {Int_t runId, evtId;};"
    #ROOT.gROOT.ProcessLine( cmd )

    ROOT.gROOT.ProcessLine(".L EventClasses.h+")

    n_evt = 5
    n_hit = 20

    #instance of the C structure
    id = ROOT.Id()
    hits = ROOT.TClonesArray("Hit")
    label = ROOT.Label()


    #pointing to variables in the C structure
    tree.Branch("Id", id)

    tree.Branch("Hits", hits)
    tree.Branch("Label", label)



    #input loop
    for i in range(n_evt):
        hits.Clear()
        #columns for the line
        #assign branch values
        id.runId = 1
        id.eventId = i

        label.pdgCode = 11
        label.pz = 100
        label.iz = 1+i
        label.iy = 2+i
        label.iz = 3+i
        for j in range(n_hit):
            hit = hits.ConstructedAt(j)  # Correct way to create a new Hit
            hit.x = random.gauss(0, 1)
            hit.y = random.gauss(0, 1)
            hit.z = random.gauss(0, 1)

        #fill the tree
        tree.Fill()

    # Write and close
    tree.Write()
    file.Close()

def read_root():
    example_root = ROOT.TFile('/afs/cern.ch/user/z/zhibin/work/snd-ml/data/convertedData/example.root', 'read')
    tree = example_root.Get('tree')

    tree.Print()

def main():
    creat_root()
    #read_root()


if __name__ == "__main__":
    main()