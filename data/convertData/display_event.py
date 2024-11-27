import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

def linear_fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)

    x_across_det = [280, 575]
    y_fit = [slope * xi + intercept for xi in x_across_det]

    return x_across_det, y_fit


def draw_event(x1, y1, z1, detType, orientation,fit_slope, fit_intercept,  event_index):
    # Create a figure with two subplots: one for vertical and one for horizontal fibers
    fig, (ax_vertical, ax_horizontal) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Define a color map for different detType values
    colors = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y'}
    labels = {0: 'scifi', 1: 'veto', 2: 'us', 3: 'ds'}

    x_across_det = [280, 575]
    xz_fit = [fit_slope * xi + fit_intercept for xi in x_across_det]
    
    # Plot for vertical fibers (orientation == 0) in the zy plane
    vertical_mask = orientation == 0
    for dtype in np.unique(detType[vertical_mask]):
        det_mask = (detType == dtype) & vertical_mask
        
        ax_vertical.scatter(z1[det_mask], y1[det_mask], c=colors.get(dtype, 'k'), 
                            label=labels.get(dtype, f'detType {dtype}'), s=10, alpha=0.7)
        if (dtype == 3):
            x_across_det, y_fit = linear_fit(z1[det_mask], y1[det_mask])
            ax_vertical.plot(x_across_det, y_fit, color="red", label="DS hits fitted line")
    
    # Draw transparent boxes for SciFi and muon system areas in the vertical (ZY) plot
    scifi_box_vertical = Rectangle((299, 18.8), 353 - 299, 57.8 - 18.8, linewidth=1, edgecolor='b', facecolor='b', alpha=0.1, label='SciFi Area')
    muon_box_vertical = Rectangle((379, 13.9), 554 - 379, 74 - 13.9, linewidth=1, edgecolor='r', facecolor='r', alpha=0.1, label='Muon System Area')
    ax_vertical.add_patch(scifi_box_vertical)
    ax_vertical.add_patch(muon_box_vertical)

    ax_vertical.set_xlabel('Z axis')
    ax_vertical.set_ylabel('Y axis')
    ax_vertical.set_title(f'Event {event_index} - Horizontal Fibers (ZY Plane)')
    ax_vertical.set_xlim(280, 575)
    ax_vertical.set_ylim(-5, 75)
    ax_vertical.legend(loc='best')

    # Plot for horizontal fibers (orientation == 1) in the zx plane
    horizontal_mask = orientation == 1
    for dtype in np.unique(detType[horizontal_mask]):
        det_mask = (detType == dtype) & horizontal_mask
        ax_horizontal.scatter(z1[det_mask], x1[det_mask], c=colors.get(dtype, 'k'), 
                              label=labels.get(dtype, f'detType {dtype}'), s=10, alpha=0.7)
        if (dtype == 3):
            x_across_det, y_fit = linear_fit(z1[det_mask], x1[det_mask])
            ax_horizontal.plot(x_across_det, y_fit, color="red", label="DS hits fitted line")
    
    # Draw transparent boxes for SciFi and muon system areas in the horizontal (ZX) plot
    scifi_box_horizontal = Rectangle((299, -45.9), 353 - 299, -6.9 + 45.9, linewidth=1, edgecolor='b', facecolor='b', alpha=0.1, label='SciFi Area')
    muon_box_horizontal = Rectangle((379, -78.9), 554 - 379, -3.6 + 78.9, linewidth=1, edgecolor='r', facecolor='r', alpha=0.1, label='Muon System Area')
    ax_horizontal.add_patch(scifi_box_horizontal)
    ax_horizontal.add_patch(muon_box_horizontal)

    #ax_horizontal.plot(x_across_det, xz_fit, color="blue", label="DS fitted with root")
    ax_horizontal.set_xlabel('Z axis')
    ax_horizontal.set_ylabel('X axis')
    ax_horizontal.set_title(f'Event {event_index} - Vertical Fibers (ZX Plane)')
    ax_horizontal.set_xlim(280, 575)
    ax_horizontal.set_ylim(-90, 30)
    ax_horizontal.legend(loc='best')




    fig.suptitle(f'Detector Event Visualization - Event {event_index}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate title
    return fig


# Main function to read data, call draw_event, and save to PDF
def read_file_and_draw(dir, file_name, event_number):
    
    file_path = f'{dir}/{file_name}.root'
    tree_name = 'cbmsim'
    
    # Open the ROOT file and retrieve data
    with uproot.open(file_path) as file:
        tree = file[tree_name]



    # Read the specified number of events
    event_data = tree.arrays(
        #["Hits.x1", "Hits.y1", "Hits.z1", "Hits.x2", "Hits.y2", "Hits.z2", "Hits.detType", "Hits.orientation", 'fit_slope', "fit_intercept"],
        ["Hits.x1", "Hits.y1", "Hits.z1", "Hits.x2", "Hits.y2", "Hits.z2", "Hits.detType", "Hits.orientation"],
        entry_start=0,
        entry_stop=event_number + 1,
        library="np"
    )

    pdf_path = f"plots/{file_name}_fitted_event_display_{event_number}.pdf"
    with PdfPages(pdf_path) as pdf:
        for i in range(len(event_data["Hits.x1"])):
            # Extract data for each event
            x1 = event_data["Hits.x1"][i]
            y1 = event_data["Hits.y1"][i]
            z1 = event_data["Hits.z1"][i]
            detType = event_data["Hits.detType"][i]
            orientation = event_data["Hits.orientation"][i]

            #fit_slope = event_data['fit_slope'][i]
            #fit_intercept = event_data['fit_intercept'][i]
            fit_slope = 0
            fit_intercept = 0
            

            # Create a figure for the event with both vertical and horizontal fibers
            fig = draw_event(x1, y1, z1, detType, orientation, fit_slope, fit_intercept, event_index=i)
            
            # Save the figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Plots saved to {pdf_path}")

# Execute main function
if __name__ == "__main__":
    dir = '/eos/user/z/zhibin/sndData/converted/real_muon/selection_tmp/'
    
    margin = 5
    #read_file_and_draw(dir, f'scifi_area_margin{margin}cm_selection', 50)
    #read_file_and_draw(dir, f'scifi_area_margin{margin}cm_no_scifi12_selection', 50)
    read_file_and_draw(dir, f'outside_scifi_area_selection', 50)
    