sndEventFilter \
  --input /eos/experiment/sndlhc/convertedData/physics/2024/run_241/run_008285/sndsw_raw-0000.root \
  --output /eos/experiment/sndlhc/users/zhibin/real_data/run_241/run_008285/nueAnalysisFilter_real_data_run_241_run_008285_sndsw_raw-0000.root \
  --geofile /eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root \
  --pipeline /afs/cern.ch/user/z/zhibin/work/public/SndBuild/sndsw/analysis/analyses/snd_analysis_2024_0mu/pipelines/nueFilterMoriondOrder_withEventLevelOutput.h
  
  sndEventFilter \
  --input /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2024/nu12/volume_volTarget/1/sndLHC.Genie-TGeant4_dig.root \
  --output /eos/experiment/sndlhc/users/zhibin/MC_neutrino/2024_ve/1/nueAnalysisFilter_MC_neutrino_2024_ve_1.root \
  --geofile  /eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2024/nu12/volume_volTarget/1/geofile_full.Genie-TGeant4.root \
  --pipeline /afs/cern.ch/user/z/zhibin/work/public/SndBuild/sndsw/analysis/analyses/snd_analysis_2024_0mu/pipelines/nueFilterMoriondOrder_withEventLevelOutput.h