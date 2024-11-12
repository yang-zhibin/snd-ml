import ROOT
import os
from argparse import ArgumentParser
import ROOT

ROOT.gROOT.ProcessLine(".L EventClasses.h+")

def plot_2d_distribution(rdf, name, x_column, y_column, nbins_x=100, nbins_y=100):
    # Calculate the min and max values for both columns
    x_min = rdf.Min(x_column).GetValue()
    x_max = rdf.Max(x_column).GetValue()
    y_min = rdf.Min(y_column).GetValue()
    y_max = rdf.Max(y_column).GetValue()

    # Create a 2D histogram using the calculated limits
    hist2d = rdf.Histo2D(("hist_name", f"2D Distribution of {x_column} vs {y_column};{x_column};{y_column}",
                          nbins_x, x_min, x_max, nbins_y, y_min, y_max), x_column, y_column)

    # Draw the 2D histogram
    canvas = ROOT.TCanvas("canvas", "2D Distribution", 800, 600)
    hist2d.Draw("COLZ")  # "COLZ" draws the histogram with a color palette
    out_path = f"plots/{name}_{x_column}_vs_{y_column}.png"
    canvas.SaveAs(out_path)  # Save the histogram to a file if needed
    #print(f'save plot {out_path}')

def filter_scifi_area(veto_and_us_ds, total_count):
    # Apply scifi position filter x(-45.9, -6.9) y(18.8, 57.8) cm
    margin = 0.5
    x1 = -45.9 + margin
    x2 = -6.9 - margin
    y1 = 18.8 + margin
    y2 = 57.8 - margin
    scifi_area = veto_and_us_ds.Filter(
        f"Label.scifi_avg_x_pos > {x1} && Label.scifi_avg_x_pos < {x2} && "
       f"Label.scifi_avg_y_pos > {y1} && Label.scifi_avg_y_pos < {y2}"
    )

    scifi_area_count = scifi_area.Count().GetValue()

    plot_2d_distribution (fiducial, 'fiducial', 'Label.scifi_avg_x_pos', 'Label.scifi_avg_y_pos')
    plot_2d_distribution (fiducial, 'fiducial','Label.DS_avg_x_pos', 'Label.DS_avg_y_pos')
    plot_2d_distribution (fiducial, 'fiducial', 'Label.scifi_avg_ver', 'Label.scifi_avg_hor')
    plot_2d_distribution (fiducial, 'fiducial','Label.DS_avg_ver', 'Label.DS_avg_hor')

    # Apply various scifi filters
    no_scifi1 = scifi_area.Filter('Label.scifi1 == 0')
    no_scifi2 = scifi_area.Filter('Label.scifi2 == 0')
    no_scifi3 = scifi_area.Filter('Label.scifi3 == 0')
    no_scifi4 = scifi_area.Filter('Label.scifi4 == 0')
    no_scifi5 = scifi_area.Filter('Label.scifi5 == 0')
    no_scifi12 = scifi_area.Filter('Label.scifi1 == 0 && Label.scifi2 == 0')


    print(f"acceptance area: scifi area - {margin} cm")

    print(f'scifi area muon_like count: {scifi_area_count:2e}, ratio: {(scifi_area_count/total_count if total_count != 0 else 0):2e}')
    print(f'no scifi1 count: {no_scifi1.Count().GetValue():2e}, ratio: {(no_scifi1.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')
    print(f'no scifi2 count: {no_scifi2.Count().GetValue():2e}, ratio: {(no_scifi2.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')
    print(f'no scifi3 count: {no_scifi3.Count().GetValue():2e}, ratio: {(no_scifi3.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')
    print(f'no scifi4 count: {no_scifi4.Count().GetValue():2e}, ratio: {(no_scifi4.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')
    print(f'no scifi5 count: {no_scifi5.Count().GetValue():2e}, ratio: {(no_scifi5.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')
    print(f'no scifi1 and 2 count: {no_scifi12.Count().GetValue():2e}, ratio: {(no_scifi12.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')

def filter_fiducial(veto_and_us_ds, total_count):

    # Apply DS fiducial 
    #hor [10, 50], vert [70, 107]
    
    fiducial = veto_and_us_ds.Filter(f"Label.DS_avg_ver >= 70 && Label.DS_avg_ver <= 105 && Label.DS_avg_hor >= 10 && Label.DS_avg_hor <= 50 &&"
                                     "Label.scifi_avg_ver >=300 && Label.scifi_avg_ver <=1336 && Label.scifi_avg_hor >=200 && Label.scifi_avg_ver <=1200"   )
    fiducial_count = fiducial.Count().GetValue()
    no_scifi1_fiducial = fiducial.Filter('Label.scifi1 == 0')
    no_scifi2_fiducial = fiducial.Filter('Label.scifi2 == 0')
    no_scifi3_fiducial = fiducial.Filter('Label.scifi3 == 0')
    no_scifi4_fiducial = fiducial.Filter('Label.scifi4 == 0')
    no_scifi5_fiducial = fiducial.Filter('Label.scifi5 == 0')
    no_scifi12_fiducial = fiducial.Filter('Label.scifi1 == 0 && Label.scifi2 == 0')

    plot_2d_distribution (fiducial, 'fiducial', 'Label.scifi_avg_x_pos', 'Label.scifi_avg_y_pos')
    plot_2d_distribution (fiducial, 'fiducial','Label.DS_avg_x_pos', 'Label.DS_avg_y_pos')
    plot_2d_distribution (fiducial, 'fiducial', 'Label.scifi_avg_ver', 'Label.scifi_avg_hor')
    plot_2d_distribution (fiducial, 'fiducial','Label.DS_avg_ver', 'Label.DS_avg_hor')

    print()
    print(f'fiducial count: {fiducial_count:2e}, ratio: {(fiducial_count/total_count if total_count != 0 else 0):2e}')
    print(f'no scifi1 count: {no_scifi1_fiducial.Count().GetValue():2e}, ratio: {(no_scifi1_fiducial.Count().GetValue()/fiducial_count if fiducial_count != 0 else 0):2e}')
    print(f'no scifi2 count: {no_scifi2_fiducial.Count().GetValue():2e}, ratio: {(no_scifi2_fiducial.Count().GetValue()/fiducial_count if fiducial_count != 0 else 0):2e}')
    print(f'no scifi3 count: {no_scifi3_fiducial.Count().GetValue():2e}, ratio: {(no_scifi3_fiducial.Count().GetValue()/fiducial_count if fiducial_count != 0 else 0):2e}')
    print(f'no scifi4 count: {no_scifi4_fiducial.Count().GetValue():2e}, ratio: {(no_scifi4_fiducial.Count().GetValue()/fiducial_count if fiducial_count != 0 else 0):2e}')
    print(f'no scifi5 count: {no_scifi5_fiducial.Count().GetValue():2e}, ratio: {(no_scifi5_fiducial.Count().GetValue()/fiducial_count if fiducial_count != 0 else 0):2e}')
    print(f'no scifi1 and 2 count: {no_scifi12_fiducial.Count().GetValue():2e}, ratio: {(no_scifi12_fiducial.Count().GetValue()/fiducial_count if fiducial_count != 0 else 0):2e}')



def filter_scifi345(veto_and_us_ds,total_count):

    veto_and_us_ds_scifi345 = veto_and_us_ds.Filter('Label.scifi3 > 0 && Label.scifi3 < 7 && Label.scifi4 > 0 && Label.scifi4 < 7 && '
        'Label.scifi5 > 0 && Label.scifi5 < 7')

    veto_and_us_ds_scifi345 = veto_and_us_ds_scifi345.Filter(f"Label.DS_avg_ver >= 70 && Label.DS_avg_ver <= 105 && Label.DS_avg_hor >= 10 && Label.DS_avg_hor <= 50 &&"
                                     "Label.scifi_avg_ver >=300 && Label.scifi_avg_ver <=1336 && Label.scifi_avg_hor >=200 && Label.scifi_avg_ver <=1200"   )

    veto_and_us_ds_scifi345_count = veto_and_us_ds_scifi345.Count().GetValue()

    no_only_scifi1 = veto_and_us_ds_scifi345.Filter('Label.scifi1 == 0')
    no_only_scifi2 = veto_and_us_ds_scifi345.Filter('Label.scifi2 == 0')
    no_only_scifi12 = veto_and_us_ds_scifi345.Filter('Label.scifi2 == 0 && Label.scifi2 == 0')

    plot_2d_distribution (veto_and_us_ds_scifi345, 'veto_and_us_ds_scifi345', 'Label.scifi_avg_x_pos', 'Label.scifi_avg_y_pos')
    plot_2d_distribution (veto_and_us_ds_scifi345, 'veto_and_us_ds_scifi345','Label.DS_avg_x_pos', 'Label.DS_avg_y_pos')
    plot_2d_distribution (veto_and_us_ds_scifi345, 'veto_and_us_ds_scifi345', 'Label.scifi_avg_ver', 'Label.scifi_avg_hor')
    plot_2d_distribution (veto_and_us_ds_scifi345, 'veto_and_us_ds_scifi345','Label.DS_avg_ver', 'Label.DS_avg_hor')

    print()
    print(f'veto, us, ds scifi345 count, and fiducial: {veto_and_us_ds_scifi345_count:2e}, ratio: {(veto_and_us_ds_scifi345_count/total_count if total_count != 0 else 0):2e}')
    print(f'no only scifi1 count: {no_only_scifi1.Count().GetValue():2e}, ratio: {(no_only_scifi1.Count().GetValue()/veto_and_us_ds_scifi345_count if veto_and_us_ds_scifi345_count != 0 else 0):2e}')
    print(f'no only scifi2 count: {no_only_scifi2.Count().GetValue():2e}, ratio: {(no_only_scifi2.Count().GetValue()/veto_and_us_ds_scifi345_count if veto_and_us_ds_scifi345_count != 0 else 0):2e}')
    print(f'no only scifi12 count: {no_only_scifi12.Count().GetValue():2e}, ratio: {(no_only_scifi12.Count().GetValue()/veto_and_us_ds_scifi345_count if veto_and_us_ds_scifi345_count != 0 else 0):2e}')


    
   

def filter_no_us(veto_and_scifi):

    no_us1 = veto_and_scifi.Filter(
        "Label.us1 == 0 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1 && "
        "(Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)"
    )
    
    # Similar filters for other upstream and downstream conditions
    no_us2 = veto_and_scifi.Filter(
        "Label.us1 == 1 && Label.us2 == 0 && Label.us3 == 1 && Label.us4 == 1 && "
        "(Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)"
    )
    no_us3 = veto_and_scifi.Filter(
        "Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 0 && Label.us4 == 1 && "
        "(Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)"
    )
    no_us4 = veto_and_scifi.Filter(
        "Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 0 && "
        "(Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)"
    )
    no_us5 = veto_and_scifi.Filter(
        "Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1 && "
        "(Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)"
    )
    print(f'no us1 count: {no_us1.Count().GetValue():2e}, ratio: {(no_us1.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')
    print(f'no us2 count: {no_us2.Count().GetValue():2e}, ratio: {(no_us2.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')
    print(f'no us3 count: {no_us3.Count().GetValue():2e}, ratio: {(no_us3.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')
    print(f'no us4 count: {no_us4.Count().GetValue():2e}, ratio: {(no_us4.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')
    print(f'no us5 count: {no_us5.Count().GetValue():2e}, ratio: {(no_us5.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')

def filter_no_ds(veto_and_scifi):
    no_ds1 = veto_and_scifi.Filter(
        "Label.ds1 == 0 && Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1"
    )
    no_ds2 = veto_and_scifi.Filter(
        "Label.ds2 == 0 && Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1"
    )
    no_ds3 = veto_and_scifi.Filter(
        "Label.ds3 == 0 && Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1"
    )
    no_ds4 = veto_and_scifi.Filter(
        "Label.ds4 == 0 && Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1"
    )

    print(f'no ds1 count: {no_ds1.Count().GetValue():2e}, ratio: {(no_ds1.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')
    print(f'no ds2 count: {no_ds2.Count().GetValue():2e}, ratio: {(no_ds2.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')
    print(f'no ds3 count: {no_ds3.Count().GetValue():2e}, ratio: {(no_ds3.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')
    print(f'no ds4 count: {no_ds4.Count().GetValue():2e}, ratio: {(no_ds4.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')




def main():
    # Directory to search files in
    directory = "/eos/user/z/zhibin/sndData/converted/real_muon/muon_2023_reprocess_2"

    # List to store the matching files
    file_list = []

    # Walk through the directory, including subfolders
    for root, _, files in os.walk(directory):
        for filename in files:
            if "converted" in filename and filename.endswith('.root'):
                file_list.append(os.path.join(root, filename))
        if (len(file_list)>300):
            break
    
    rdf = ROOT.RDataFrame("cbmsim", file_list)

    total_count = rdf.Count().GetValue()

    # Apply veto filter
    veto = rdf.Filter('Label.veto1 > 0 && Label.veto2 > 0')

    # Filter events that satisfy the upstream and downstream conditions
    veto_and_us_ds = veto.Filter(
        "Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1 && Label.us5 == 1&& "
        "(Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)"
    )

    # Count muon-like events
    veto_and_us_ds_count = veto_and_us_ds.Count().GetValue()


    #filter_fiducial(veto_and_us_ds, total_count)

    

    # plot_2d_distribution (veto_and_us_ds, 'veto_and_us_ds', 'Label.scifi_avg_x_pos', 'Label.scifi_avg_y_pos')
    # plot_2d_distribution (veto_and_us_ds, 'veto_and_us_ds','Label.DS_avg_x_pos', 'Label.DS_avg_y_pos')
    # plot_2d_distribution (veto_and_us_ds, 'veto_and_us_ds', 'Label.scifi_avg_ver', 'Label.scifi_avg_hor')
    # plot_2d_distribution (veto_and_us_ds, 'veto_and_us_ds','Label.DS_avg_ver', 'Label.DS_avg_hor')
   
    # Apply veto and scifi filters
    veto_and_scifi = veto.Filter(
        'Label.scifi1 > 0 && Label.scifi1 < 7 && Label.scifi2 > 0 && Label.scifi2 < 7 && '
        'Label.scifi3 > 0 && Label.scifi3 < 7 && Label.scifi4 > 0 && Label.scifi4 < 7 && '
        'Label.scifi5 > 0 && Label.scifi5 < 7'
    )
    # Print the results and calculate ratios
    print(f'total count: {total_count:2e}')
    print(f'veto and us and ds count: {veto_and_us_ds_count:2e}, ratio: {veto_and_us_ds_count/total_count:2e}')
    veto_and_scifi_count = veto_and_scifi.Count().GetValue()
    print(f'veto and scifi count: {veto_and_scifi_count:2e}, ratio: {veto_and_scifi_count/total_count:2e}')

    filter_scifi345(veto_and_us_ds,total_count)


    
   



if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("-i", "--inFile", dest="in_file", help="input file path", required=True)
    # parser.add_argument("-o", "--outFile", dest="out_file", help="output file path", required=True)
    # parser.add_argument("-m", "--muonLike", dest="muon_like", help="selection for moun-like real data", type=bool, default=False)
    # args = parser.parse_args()
    
    # main(args)
    main()
