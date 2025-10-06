import os
import csv
import ROOT
import pandas as pd
import glob
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import difflib


f = ROOT.TFile.Open("/eos/experiment/sndlhc/convertedData/commissioning/testbeam_24/run_100894/sndsw_raw-0000.root")
# f.ls()

tree = f.Get("cbmsim")
# for br in tree.GetListOfBranches():
#     print(br.GetName())
tree.Print()