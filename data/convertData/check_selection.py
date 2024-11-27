import ROOT
import os
from argparse import ArgumentParser
import ROOT
import numpy as np

ROOT.gROOT.ProcessLine(".L EventClasses.h+")
# Function to fit each event and return slope and intercept, switching x and z
ROOT.gInterpreter.Declare("""
    std::pair<float, float> LinearFitPerEvent(const ROOT::VecOps::RVec<float>& z, const ROOT::VecOps::RVec<float>& x) {
        if (x.size() == 0 || z.size() == 0) {
            return std::make_pair(0.0, 0.0);  // Return default values if there is no data
        }
        std::vector<float> ex(z.size(), 1.0f);  // x-errors initialized to 1.0
        std::vector<float> ey(z.size(), 1.0f);

        // Print out the contents of z and x for debugging
        // std::cout << "z: ";
        // for (const auto& val : z) {
        //     std::cout << val << " ";
        // }
        // std::cout << std::endl;
    
        // std::cout << "x: ";
        // for (const auto& val : x) {
        //     std::cout << val << " ";
        // }
        // std::cout << std::endl;
        
        TGraphErrors graph(z.size(), z.data(), x.data(), ex.data(), ey.data());
        TF1 linear_func("linear_func", "[0]*x + [1]", 0, 600);
        graph.Fit(&linear_func, "Q");  // Suppress fit output

        float slope = linear_func.GetParameter(0);
        float intercept = linear_func.GetParameter(1);

        return std::make_pair(slope, intercept);
    }
    """)

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

def filter_scifi_area(veto_and_us_ds, total_count, margin = 0.5, if_save=False):
    # Apply scifi position filter x(-45.9, -6.9) y(18.8, 57.8) cm
    #margin = 5
    x1 = -45.9 + margin
    x2 = -6.9 - margin
    y1 = 18.8 + margin
    y2 = 57.8 - margin

    scifi_area = veto_and_us_ds.Filter(
        f"Label.DS_avg_x_pos > {x1} && Label.DS_avg_x_pos < {x2} && "
       f"Label.DS_avg_y_pos > {y1} && Label.DS_avg_y_pos < {y2}"
    )

    if (if_save):
        dir = '/eos/user/z/zhibin/sndData/converted/real_muon/selection_tmp/'
        out_file_path = f'{dir}/scifi_area_margin{margin}cm_selection.root'
        print(f'saving scifi_area selection to {out_file_path}')
        scifi_area.Snapshot('cbmsim', out_file_path)


    scifi_area_count = scifi_area.Count().GetValue()


    # Apply various scifi filters
    no_scifi1 = scifi_area.Filter('Label.scifi1 == 0')
    no_scifi2 = scifi_area.Filter('Label.scifi2 == 0')
    no_scifi3 = scifi_area.Filter('Label.scifi3 == 0')
    no_scifi4 = scifi_area.Filter('Label.scifi4 == 0')
    no_scifi5 = scifi_area.Filter('Label.scifi5 == 0')
    no_scifi12 = scifi_area.Filter('Label.scifi1 == 0 && Label.scifi2 == 0')

    if (if_save):
        dir = '/eos/user/z/zhibin/sndData/converted/real_muon/selection_tmp/'
        out_file_path = f'{dir}/scifi_area_margin{margin}cm_no_scifi12_selection.root'
        print(f'saving scifi_area selection to {out_file_path}')
        no_scifi12.Snapshot('cbmsim', out_file_path)


    print(f"acceptance area: scifi area - {margin} cm")
    
    print(f'scifi area muon_like count: {scifi_area_count:2e}, ratio: {(scifi_area_count/total_count if total_count != 0 else 0):2e}')
    print(f'no scifi1 count: {no_scifi1.Count().GetValue():2e}, ratio: {(no_scifi1.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')
    print(f'no scifi2 count: {no_scifi2.Count().GetValue():2e}, ratio: {(no_scifi2.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')
    print(f'no scifi3 count: {no_scifi3.Count().GetValue():2e}, ratio: {(no_scifi3.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')
    print(f'no scifi4 count: {no_scifi4.Count().GetValue():2e}, ratio: {(no_scifi4.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')
    print(f'no scifi5 count: {no_scifi5.Count().GetValue():2e}, ratio: {(no_scifi5.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')
    print(f'no scifi1 and 2 count: {no_scifi12.Count().GetValue():2e}, ratio: {(no_scifi12.Count().GetValue()/scifi_area_count if scifi_area_count != 0 else 0):2e}')

def filter_scifi_area_with_channel(veto_and_us_ds, total_count, scifi_margin = 10, ds_margin = 2):
    # Apply scifi position filter x(-45.9, -6.9) y(18.8, 57.8) cm
    #margin = 0.5
    ds_ver_min = 0
    ds_ver_max = 0
    ds_hor_min = 0
    ds_hor_max = 0
    scifif_ver_min = 0
    scifif_ver_max = 0
    scifif_hor_min = 0
    scifif_hor_max = 0

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
    no_only_scifi12 = veto_and_us_ds_scifi345.Filter('Label.scifi1 == 0 && Label.scifi2 == 0')

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
        "Label.ds1 == 0 && Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1 &&"
        "(Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)"
    )
    no_ds2 = veto_and_scifi.Filter(
        "Label.ds2 == 0 && Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1 &&"
        "(Label.ds1 > 0 && Label.ds1 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)"
    )
    no_ds3 = veto_and_scifi.Filter(
        "Label.ds3 == 0 && Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1 &&"
        "(Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds4 > 0 && Label.ds4 < 4)"
    )
    no_ds4 = veto_and_scifi.Filter(
        "Label.ds4 == 0 && Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1 &&"
        "(Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) "
    )

    print(f'no ds1 count: {no_ds1.Count().GetValue():2e}, ratio: {(no_ds1.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')
    print(f'no ds2 count: {no_ds2.Count().GetValue():2e}, ratio: {(no_ds2.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')
    print(f'no ds3 count: {no_ds3.Count().GetValue():2e}, ratio: {(no_ds3.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')
    print(f'no ds4 count: {no_ds4.Count().GetValue():2e}, ratio: {(no_ds4.Count().GetValue()/veto_and_scifi_count if veto_and_scifi_count != 0 else 0):2e}')


def read_file_path(directory, n_file = 300):
    # List to store the matching files
    file_list = []

    # Walk through the directory, including subfolders
    for root, _, files in os.walk(directory):
        for filename in files:
            if "converted" in filename and filename.endswith('.root'):
                file_list.append(os.path.join(root, filename))
        if (len(file_list)>n_file):
            break

    return file_list

def linear_func(x, y):
    pass

def print_neutrino_count(counts, rdf):

    ve_count = rdf.Filter("Label.pdgCode == 12 || Label.pdgCode == -12").Count().GetValue()
    vm_count = rdf.Filter("Label.pdgCode == 14 || Label.pdgCode == -14").Count().GetValue()
    vt_count = rdf.Filter("Label.pdgCode == 16 || Label.pdgCode == -16").Count().GetValue()
    nc_count = rdf.Filter("Label.pdgCode == 112 || Label.pdgCode == -112 || "
                        "Label.pdgCode == 114 || Label.pdgCode == -114 || "
                        "Label.pdgCode == 116 || Label.pdgCode == -116").Count().GetValue()

    

    ve_ratio = ve_count/counts['ve_count'] if counts['ve_count'] > 0 else 0
    vm_ratio = vm_count/counts['vm_count'] if counts['vm_count'] > 0 else 0
    vt_ratio = vt_count/counts['vt_count'] if counts['vt_count'] > 0 else 0
    nc_ratio = nc_count/counts['nc_count'] if counts['nc_count'] > 0 else 0

    print(f"Electron neutrino (ve) count: {ve_count:.2e}, ratio: {ve_ratio:.2e}")
    print(f"Muon neutrino (vm) count:     {vm_count:.2e}, ratio: {vm_ratio:.2e}")
    print(f"Tau neutrino (vt) count:      {vt_count:.2e}, ratio: {vt_ratio:.2e}")
    print(f"Neutral current (nc) count:   {nc_count:.2e}, ratio: {nc_ratio:.2e}")

def check_scifi_plane_hits(counts, rdf):
    no_scifi1 = rdf.Filter('Label.scifi1 == 0')
    print("no scifi hit on plane 1")
    print_neutrino_count(counts, no_scifi1)
    no_scifi2 = rdf.Filter('Label.scifi2 == 0 && Label.scifi1 == 0')
    print("no scifi hit on plane 12")
    print_neutrino_count(counts, no_scifi2)
    no_scifi3 = rdf.Filter('Label.scifi3 == 0 && Label.scifi2 == 0 && Label.scifi1 == 0')
    print("no scifi hit on plane 123")
    print_neutrino_count(counts, no_scifi3)
    no_scifi4 = rdf.Filter('Label.scifi4 == 0 && Label.scifi3 == 0 && Label.scifi2 == 0 && Label.scifi1 == 0')
    print("no scifi hit on plane 1234")
    print_neutrino_count(counts, no_scifi4)
    no_scifi5 = rdf.Filter('Label.scifi5 == 0 && Label.scifi4 == 0 && Label.scifi3 == 0 && Label.scifi2 == 0 && Label.scifi1 == 0')
    print("no scifi hit on plane 12345")
    print_neutrino_count(counts, no_scifi5)

def check_neutrino_selection():
     # Directory to search files in
    #directory = "/eos/user/z/zhibin/sndData/converted/real_muon/2023_reprocess"
    directory = "/eos/user/z/zhibin/sndData/converted/Neutrinos_v2"

    start = 188
    end = 400 
    file_list = []
    for i in range(start, end):
        dir_i = f"{directory}/{i}/"
        if os.path.isdir(dir_i):
            for filename in os.listdir(dir_i):
                if "converted" in filename and filename.endswith('.root'):
                    file_list.append(os.path.join(dir_i, filename))

    print(f"n files: {len(file_list)}")
    rdf = ROOT.RDataFrame("cbmsim", file_list)

    total_count = rdf.Count().GetValue()

    ve_count = rdf.Filter("Label.pdgCode == 12 || Label.pdgCode == -12").Count().GetValue()
    vm_count = rdf.Filter("Label.pdgCode == 14 || Label.pdgCode == -14").Count().GetValue()
    vt_count = rdf.Filter("Label.pdgCode == 16 || Label.pdgCode == -16").Count().GetValue()
    nc_count = rdf.Filter("Label.pdgCode == 112 || Label.pdgCode == -112 || "
                        "Label.pdgCode == 114 || Label.pdgCode == -114 || "
                        "Label.pdgCode == 116 || Label.pdgCode == -116").Count().GetValue()

    counts = {
    "ve_count": ve_count,
    "vm_count": vm_count,
    "vt_count": vt_count,
    "nc_count": nc_count
    }

    check_scifi_plane_hits(counts, rdf)
    

    margin = 5 #cm
    x1 = -45.9 + margin
    x2 = -6.9 - margin
    y1 = 18.8 + margin
    y2 = 57.8 - margin

    # Print the results and calculate ratios
    print(f'total count: {total_count:2e}')
    print_neutrino_count(counts, rdf)

    scifi_area = rdf.Filter(
        f"Label.DS_avg_x_pos > {x1} && Label.DS_avg_x_pos < {x2} && "
       f"Label.DS_avg_y_pos > {y1} && Label.DS_avg_y_pos < {y2}"
    )

    print('acceptance area cut')
    check_scifi_plane_hits(counts, scifi_area)
    print_neutrino_count(counts, scifi_area)

    scifi_area_ds_us = rdf.Filter(
        "Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1 && Label.us5 == 1&& "
        "(Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)"
    )

    print('us ds hits cut')
    print_neutrino_count(counts, scifi_area_ds_us)

    rdf = rdf.Define('ds_x', "Hits.x1[Hits.detType == 3 && Hits.orientation == 1]") \
                                        .Define('ds_z', "Hits.z1[Hits.detType == 3 && Hits.orientation == 1]")


    rdf = rdf.Define("fit_result", "LinearFitPerEvent(ds_z, ds_x)") \
                                        .Define("fit_slope", "fit_result.first") \
                                        .Define("fit_intercept", "fit_result.second")

    rdf = rdf.Define("scifi_intercept", "fit_slope*300+fit_intercept")                        

    rdf_fitted = rdf.Filter(f"scifi_intercept>{x1}&& scifi_intercept<{x2}")
    print('fitted track position cut')
    print_neutrino_count(counts, rdf_fitted)



    if_save = False

    if (if_save):
        dir = '/eos/user/z/zhibin/sndData/converted/Neutrinos_v2/selection_tmp/'
        out_file_path = f'{dir}/scifi_area_margin{margin}cm_selection.root'
        print(f'saving scifi_area selection to {out_file_path}')
        scifi_area.Snapshot('cbmsim', out_file_path)

    #filter_scifi345(veto_and_us_ds,total_count)
    scifi_area_count = scifi_area.Count().GetValue()
    #print(f'scifi area muon_like count: {scifi_area_count:2e}, ratio: {(scifi_area_count/total_count if total_count != 0 else 0):2e}')
    #filter_scifi_area(rdf, total_count, margin=margin, if_save=False)


def check_selection():
    # Directory to search files in
    directory = "/eos/user/z/zhibin/sndData/converted/real_muon/2023_reprocess"
    #directory = "/eos/user/z/zhibin/sndData/converted/Neutrinos_v2"

    file_list = read_file_path(directory, 400)
    print(f"n files: {len(file_list)}")
    rdf = ROOT.RDataFrame("cbmsim", file_list)

    total_count = rdf.Count().GetValue()

    margin = 5 #cm
    x1 = -45.9 + margin
    x2 = -6.9 - margin
    y1 = 18.8 + margin
    y2 = 57.8 - margin

    acceptance = rdf.Filter(
        f"Label.DS_avg_x_pos > {x1} && Label.DS_avg_x_pos < {x2} && "
        f"Label.DS_avg_y_pos > {y1} && Label.DS_avg_y_pos < {y2}"
    )
    acceptance_and_veto = acceptance.Filter('Label.veto1 > 0 || Label.veto2 > 0')
    acceptance_and_no_veto = acceptance.Filter('Label.veto1 == 0 and Label.veto2 == 0')

    if_save = True
    if (if_save):
        dir = '/eos/user/z/zhibin/sndData/converted/real_muon/selection_tmp/'
        out_file_path = f'{dir}/scifi_area_margin{margin}cm_veto.root'
        print(f'saving scifi_area with veto hits selection to {out_file_path}')
        acceptance_and_veto.Snapshot('cbmsim', out_file_path)
    if (if_save):
        dir = '/eos/user/z/zhibin/sndData/converted/real_muon/selection_tmp/'
        out_file_path = f'{dir}/scifi_area_margin{margin}cm_no_veto.root'
        print(f'saving scifi_area with no veto selection to {out_file_path}')
        acceptance_and_no_veto.Snapshot('cbmsim', out_file_path)

    # Apply veto filter
    veto = rdf.Filter('Label.veto1 > 0 && Label.veto2 > 0')


    # Filter events that satisfy the upstream and downstream conditions
    veto_and_us_ds = veto.Filter(
        "Label.us1 == 1 && Label.us2 == 1 && Label.us3 == 1 && Label.us4 == 1 && Label.us5 == 1&& "
        "(Label.ds1 > 0 && Label.ds1 < 4) && (Label.ds2 > 0 && Label.ds2 < 4) && "
        "(Label.ds3 > 0 && Label.ds3 < 4) && (Label.ds4 > 0 && Label.ds4 < 4)"
    )
    veto_and_us_ds_count = veto_and_us_ds.Count().GetValue()



    veto_and_us_ds_in_acceptance = veto_and_us_ds.Filter(f"!Any(Hits.detType == 1 && (Hits.y1 > {y2} || Hits.y1 < {y1}))")

    veto_and_us_ds_outside_acceptance = veto_and_us_ds.Filter(f"Any(Hits.detType == 1 && (Hits.y1 > {y2} || Hits.y1 < {y1}))")

    veto_and_us_ds_in_acceptance_count = veto_and_us_ds_in_acceptance.Count().GetValue()

    out_file_path = f'/eos/user/z/zhibin/sndData/converted/real_muon/selection_tmp//outside_scifi_area_selection.root'
    print(f'saving outside scifi_area selection to {out_file_path}')
    veto_and_us_ds_outside_acceptance.Snapshot('cbmsim', out_file_path)

    veto_and_us_ds_in_acceptance = veto_and_us_ds_in_acceptance.Define('ds_x', "Hits.x1[Hits.detType == 3 && Hits.orientation == 1]") \
                                                               .Define('ds_z', "Hits.z1[Hits.detType == 3 && Hits.orientation == 1]")



    # Define new columns in the dataframe for slope and intercept using the custom function
    print('fitting')
    veto_and_us_ds_in_acceptance = veto_and_us_ds_in_acceptance.Define("fit_result", "LinearFitPerEvent(ds_z, ds_x)") \
                                                            .Define("fit_slope", "fit_result.first") \
                                                            .Define("fit_intercept", "fit_result.second")

    veto_and_us_ds_in_acceptance = veto_and_us_ds_in_acceptance.Define("scifi_intercept", "fit_slope*300+fit_intercept")                        

    veto_and_us_ds_in_acceptance_fitted = veto_and_us_ds_in_acceptance.Filter(f"scifi_intercept>{x1}&& scifi_intercept<{x2}")


    veto_and_us_ds_in_acceptance_fitted_count = veto_and_us_ds_in_acceptance_fitted.Count().GetValue()


    # Apply veto and scifi filters
    veto_and_scifi = veto.Filter(
        'Label.scifi1 > 0 && Label.scifi1 < 7 && Label.scifi2 > 0 && Label.scifi2 < 7 && '
        'Label.scifi3 > 0 && Label.scifi3 < 7 && Label.scifi4 > 0 && Label.scifi4 < 7 && '
        'Label.scifi5 > 0 && Label.scifi5 < 7'
    )
    # Print the results and calculate ratios
    print(f'total count: {total_count:2e}')
    print(f'veto and us and ds count: {veto_and_us_ds_count:2e}, ratio: {veto_and_us_ds_count/total_count:2e}')
    print(f'veto and us and ds in acceptance area count: {veto_and_us_ds_in_acceptance_count:2e}, ratio: {veto_and_us_ds_in_acceptance_count/total_count:2e}')
    print(f'veto and us and ds, and fitted track in acceptance area count: {veto_and_us_ds_in_acceptance_fitted_count:2e}, ratio: {veto_and_us_ds_in_acceptance_fitted_count/total_count:2e}')



    veto_and_scifi_count = veto_and_scifi.Count().GetValue()
    print(f'veto and scifi count: {veto_and_scifi_count:2e}, ratio: {veto_and_scifi_count/total_count:2e}')

    #filter_scifi345(veto_and_us_ds,total_count)
    filter_scifi_area(veto_and_us_ds_in_acceptance_fitted, total_count, margin=margin, if_save=True)


def save_selection_sample():
    directory = "/eos/user/z/zhibin/sndData/converted/real_muon/muon_2023_reprocess_2"

    file_list = read_file_path(directory, 10)

    rdf = ROOT.RDataFrame("cbmsim", file_list)

    total_count = rdf.Count().GetValue()


   



if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("-i", "--inFile", dest="in_file", help="input file path", required=True)
    # parser.add_argument("-o", "--outFile", dest="out_file", help="output file path", required=True)
    # parser.add_argument("-m", "--muonLike", dest="muon_like", help="selection for moun-like real data", type=bool, default=False)
    # args = parser.parse_args()
    
    # main(args)
    #check_neutrino_selection()
    check_selection()


