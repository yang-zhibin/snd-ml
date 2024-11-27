#!/bin/bash
input_file="/eos/experiment/sndlhc/users/marssnd/PGsim/neutrons/neu_5_10_tgtarea/Ntuples/0/20240126_digCPP.root"

filename=$(basename "$input_file")
base_name="${filename%.root}"
muon_out_file="${base_name}__muonReco.root"
echo $muon_out_file