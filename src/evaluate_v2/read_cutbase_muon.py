import uproot
import os

# Open the ROOT file

# List all the objects in the file
# print(file.keys())

# # Access a histogram by its key (replace with the actual key name)
# hist = file["-1_AvgSFChan_0"]


#E. Average SciFi hit channel number must be within [200, 1200] (ver) and [300, max-200] (hor)
#F. Average DS hit bar number must be within [70, 105] (ver) and [10, 50] (hor)
#B. No veto hits
#C. No hits in first SciFi plane
#C. No hits in second SciFi plane
#D. Vertex not in 5th wall
#G. At least two consecutive SciFi planes hit
#H. If there is a downstream hit, require hits in all upstream stations.    
#J. Previous event more than 100 clock cycles away. To avoid deadtime issues.


def process(file_path, cuts_count, cuts):
    filtered_file = uproot.open(file_path)
    for i in range(-1,9):
        th1d_name = f'{i}_AvgSFChan_0'
        th1d = filtered_file[th1d_name]

        cut = cuts[i+1]
        th1d_count = th1d.member("fEntries")
        cuts_count[cut] += th1d_count

    




def main():
    cuts = ['Before_cut','A_avg_SF','B_avg_DS','C_no_Veto','D_no_1st_SF','E_no_2nd_SF','F_no_5th_Wall','G_2_consecutive_SF','H_DS_US','I_100_clock' ]
    cuts_count = {cut: 0 for cut in set(cuts)}
    filtered_dir = '/eos/user/z/zhibin/sndData/converted/real_muon/muon_2023_reprocess_2/'
    for root, dirs, files in os.walk(filtered_dir):
        for file in files:
            if file.startswith('filtered'):
                file_path = os.path.join(root, file)
                process(file_path, cuts_count,cuts)

    print(cuts_count)
    for i in range(len(cuts) - 1):
        current_cut = cuts[i]
        next_cut = cuts[i + 1]
        
        # Get their counts
        current_count = cuts_count[current_cut]
        next_count = cuts_count[next_cut]
        
        # Calculate the ratio
        ratio = next_count / current_count  if current_count != 0 else 0
        print(f"cut: {next_cut}, count: {next_count}, Ratio with previous cut ({current_cut}): {ratio:.2e}")


if __name__ == "__main__":
    main()