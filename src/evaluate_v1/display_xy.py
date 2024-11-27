import ROOT

def plot_2d_distribution(file_path, tree_name, branch_x, branch_y, output_path, num_bins=100, 
                         x_range=None, y_range=None, x_title=None, y_title=None, box_coords=None):
    """
    Create and save a 2D distribution plot of two branches from a ROOT file,
    with optional manual or automatic adjustment for range and bins, axis titles, and a highlighted box.

    Parameters:
    - file_path: str, path to the ROOT file.
    - tree_name: str, name of the tree in the ROOT file.
    - branch_x: str, name of the branch to plot on the x-axis.
    - branch_y: str, name of the branch to plot on the y-axis.
    - output_path: str, path to save the output plot image.
    - num_bins: int, number of bins for histogramming (default 100).
    - x_range: tuple(float, float), range to use for the x-axis (min, max), if None, range is auto-detected.
    - y_range: tuple(float, float), range to use for the y-axis (min, max), if None, range is auto-detected.
    - x_title: str, title for the x-axis, if None, branch_x is used.
    - y_title: str, title for the y-axis, if None, branch_y is used.
    - box_coords: tuple(float, float, float, float), coordinates of the box (x1, y1, x2, y2), optional.
    """
    event_ids = []
    try:
        # Open the ROOT file
        file = ROOT.TFile(file_path, "READ")
        if file.IsZombie():
            print("File could not be opened!")
            return

        # Access the tree
        tree = file.Get(tree_name)
        if not tree:
            print(f"Tree {tree_name} not found!")
            return

        # Determine the range for each branch using temporary histograms or provided ranges
        if x_range is None or y_range is None:
            c_temp = ROOT.TCanvas("c_temp", "Temporary Canvas", 800, 600)
            if x_range is None:
                tree.Draw(branch_x + ">>hx(" + str(num_bins) + ")")
                hx = ROOT.gDirectory.Get("hx")
                x_range = (hx.GetXaxis().GetXmin(), hx.GetXaxis().GetXmax())
            if y_range is None:
                tree.Draw(branch_y + ">>hy(" + str(num_bins) + ")")
                hy = ROOT.gDirectory.Get("hy")
                y_range = (hy.GetXaxis().GetXmin(), hy.GetXaxis().GetXmax())

        # Create a 2D histogram with appropriate ranges and bins
        hist2d = ROOT.TH2F("hist2d", f"2D Distribution of {branch_x} and {branch_y}",
                           num_bins, x_range[0], x_range[1], num_bins, y_range[0], y_range[1])

        # Set axis titles
        hist2d.GetXaxis().SetTitle(x_title if x_title else branch_x)
        hist2d.GetYaxis().SetTitle(y_title if y_title else branch_y)

        # Fill the histogram
        i_event = 0
        for event in tree:
            x_value = getattr(event, branch_x)
            y_value = getattr(event, branch_y)
            hist2d.Fill(x_value, y_value)
            if box_coords and (box_coords[0] <= x_value <= box_coords[2]) and (box_coords[1] <= y_value <= box_coords[3]):
                event_ids.append(i_event)
            i_event += 1
        # Create a canvas to draw the histogram
        canvas = ROOT.TCanvas("canvas", "Canvas for plotting", 800, 600)
        hist2d.Draw("COLZ")

        # If box coordinates are given, draw the box
        if box_coords:
            box = ROOT.TBox(box_coords[0], box_coords[1], box_coords[2], box_coords[3])
            box.SetFillStyle(0)  # Transparent fill
            box.SetLineColor(ROOT.kRed)
            box.SetLineWidth(3)  # White line color
            box.Draw()

        # Save the canvas as an image file
        canvas.SaveAs(output_path)

    except Exception as e:
        print("An error occurred:", e)
    finally:
        # Close the ROOT file
        file.Close()
        if 'c_temp' in locals():  # Close the temporary canvas if it was used
            c_temp.Close()

    return event_ids, i_event

if __name__ == "__main__":
    model = "vm_multiClass_weight_recoMuon_classWeight_10"
    input_file = f"/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/merge_output/{model}_merged_file.root"
    tree_name = "merged_tree"
    x = "DS_avg_hor"
    y = "DS_avg_ver"
    x_range = (0, 60)
    y_range = (60, 120)
    out_name = f"/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/plot/2D_Distribution_{model}_{x}_{y}.png"
    box_coords = (10, 70, 50, 105)  # Example coordinates for the box
    # Example usage of the function with axis titles and box
    scifi_filter_events, event_count = plot_2d_distribution(input_file, tree_name, x, y, out_name,x_title=x, y_title=y, box_coords=box_coords, x_range=x_range, y_range=y_range)

    x = "scifi_avg_ver"
    y = "scifi_avg_hor"
    x_range = (0, 1600)
    y_range = (0, 1600)
    out_name = f"/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/plot/2D_Distribution_{model}_{x}_{y}.png"
    box_coords = (200, 300, 1200, 1336)  # Example coordinates for the box
    # Example usage of the function with axis titles and box
    DS_filter_events,event_count = plot_2d_distribution(input_file, tree_name, x, y, out_name,x_title=x, y_title=y, box_coords=box_coords, x_range=x_range, y_range=y_range)

    common_events = set(scifi_filter_events) & set(DS_filter_events)

    scifi_eff = len(scifi_filter_events)/event_count
    DS_eff = len(DS_filter_events)/event_count
    fiducial_eff = len(common_events)/event_count

    print("scifi avg channel cut eff:", scifi_eff)
    print("DS avg channel cut eff:", DS_eff)
    print("Fiducial cut eff (scifi + DS):", fiducial_eff)

