
# no energy cut
- python $SNDSW_ROOT/shipLHC/run_simSND.py -n 10 --eMin 300 --PG --pID 2112 --EVx -20 --EVy 40 --EVz 270
- python  $SNDSW_ROOT/shipLHC/run_simSND.py --MuDIS -n 1000 -f  /eos/experiment/sndlhc/MonteCarlo/Pythia6/MuonDIS/muonDis_1001.root  --eMin 300
- python $SNDSW_ROOT/shipLHC/run_simSND.py --PG --Estart 100 --Eend    5000 --EVz -7100 --EVx -30 --EVy 40 --pID 13 -n 10000 --FastMuon
python $SNDSW_ROOT/shipLHC/run_simSND.py --PG --Estart 10 --Eend 500 --EVz 340 --EVx -30 --EVy 40 --pID 130 -n 100
