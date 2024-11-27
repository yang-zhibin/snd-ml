import ROOT
import os
from argparse import ArgumentParser
import ROOT

ROOT.gROOT.ProcessLine(".L EventClasses.h+")

def pre_selection():
    in_file = '/eos/user/z/zhibin/sndData/converted/Neutrinos_v2/selection_tmp/scifi_area_margin5cm_selection.root'
    out_file = '/eos/user/z/zhibin/sndData/converted/Neutrinos_v2/selection_tmp/scifi_area_margin5cm_pre_selection.root'

    rdf = ROOT.RDataFrame("cbmsim", in_file)
    rdf = rdf.Define('filtered_hits', "Hits[Hits.detType != 1]")

    filtered.Snapshot('cbmsim', out_file)


def main(args):
    
    rdf = ROOT.RDataFrame("cbmsim", args.in_file)

    total_count = rdf.Count().GetValue()
    print(f'total count: {total_count}')
    if (args.muon_like):
        rdf = rdf.Filter('Label.veto1 > 0 || Label.veto2 > 0')
        rdf = rdf.Filter("Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1 && Label.us4 == 1 && (Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && (Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)")
        muon_like_count = rdf.Count().GetValue()
        print(f'muon_like count: {muon_like_count}')
    # scifi position x = (-45.9, -6.9), y = (18.8, 57.8)
    filtered = rdf.Filter("Label.DS_avg_x_pos > -45.9 && Label.DS_avg_x_pos < -6.9 && Label.DS_avg_y_pos > 18.8 && Label.DS_avg_y_pos < 57.8")

    scifi_area_count = filtered.Count().GetValue()

    filtered.Snapshot('cbmsim', args.out_file)


    print(f'scifi area count:{scifi_area_count}')





if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("-i", "--inFile", dest="in_file", help="input file path", required=True)
    # parser.add_argument("-o", "--outFile", dest="out_file", help="output file path", required=True)
    # args = parser.parse_args()
    
    # main(args)
    pre_selection()