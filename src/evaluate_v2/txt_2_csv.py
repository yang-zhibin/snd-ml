import pandas as pd

# Function to parse each line
def parse_line(line):
    # Remove any leading/trailing whitespace
    line = line.strip()
    # Split by commas
    pairs = line.split(',')
    # Initialize dictionary to store key-value pairs
    data = {}
    for pair in pairs:
        # Split by the first colon to separate key and value
        key, value = pair.split(':', 1)
        # Add to dictionary
        data[key.strip()] = value.strip()
    #print(data)
    return data

# Read the file
file_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/plot/csv/baseline.txt'  # Replace with your file path
with open(file_path, 'r') as file:
    # Parse each line and store in a list
    data = [parse_line(line) for line in file]

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data)
df['model'] = 'baseline'

selected_columns = df[['model', 'bkg eff', 'signal eff', 'bkg yield', 'signal yield']]
selected_columns.columns = ['model', 'bkg_eff', 'signal_eff', 'bkg_yield', 'signal_yield']
out_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/csv/baseline_eff_yield_df.csv'
selected_columns.to_csv(out_path, index=False)


