
def main():
    expected_muon_flux_fiducial = 1.69e4 #cm^-2/fb^-1
    measured_muon_flux_fiducial = 2.07e4 #cm^-2/fb^-1

    lumi_2022 = 36.8 #fb^-1
    first_period_lumi_2022 = 23.1 #fb^-1
    second_period_lumi_2022 = 13.7 #fb^-1

    fiducial_area = 25*26 #cm^2
    scifi_area = 40*40 #cm^2

    avg_ineff_veto = 4.5e-4 
    ineff_veto = 1.1e-4
    total_muon_fiducial = measured_muon_flux_fiducial * lumi_2022 * fiducial_area

    measured_muon_flux_scifi = measured_muon_flux_fiducial 
    total_muon_scifi = measured_muon_flux_scifi * lumi_2022 * scifi_area

    combined_ineff = avg_ineff_veto * ineff_veto * ineff_veto  #veto + 2 scifi
    expected_muon_fiducial = total_muon_fiducial * combined_ineff 
    
   
    expected_muon_scifi = total_muon_scifi * avg_ineff_veto 

    print(f'total muon expected, fiducial area:{total_muon_fiducial:.2e}, scifi area:{total_muon_scifi:.2e}')
    print(f'Using fiducial cut: ineff (veto+2*scifi): {combined_ineff:.2e}, expected muon:{expected_muon_fiducial:.2e}')
    print(f'Not using fiducial cut, ineff(veto):{avg_ineff_veto:.2e}, expected muon:{expected_muon_scifi:.2e}')







if __name__ == "__main__":
    main()