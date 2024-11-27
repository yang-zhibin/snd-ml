import os
import pandas as pd
def main():
    directory = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/csv/'
    # Iterate over all the files in the directory
    for filename in os.listdir(directory):
        if filename.startswith('ve') and filename.endswith('.csv'):
            # Read each CSV file into a dataframe
            filepath = os.path.join(directory, filename)
            print(filepath)
            df = pd.read_csv(filepath)

            df['signal_yield_ve_1'] = df['signal_yield'] / 0.72 * 0.23 # ratio from vm observation analysis
            df['signal_yield_ve_2'] = df['signal_yield'] / 181185 * 55690 # ratio from MC, ve:55690, vm:181185, vt:3298,
            df.to_csv(filepath, index=False)            

if __name__ == "__main__":
    main()

'''
 <<< Combine >>>
 <<< v9.2.1 >>>
>>> Random number generator seed is 123456
>>> Method used is Significance

 -- Significance --
Significance: 10.4833
Done in 0.00 min (cpu), 0.00 min (real)
'''