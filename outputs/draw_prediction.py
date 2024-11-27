import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def main():
    pred_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/log/test_local/vm_predictions.csv'
    pred = pd.read_csv(pred_path)


    particle_mapping = {
        12:  've',
        -12: 've',
        14:  'vm',
        -14: 'vm',
        16:  'vt',
        -16: 'vt',
        2112:'neutron',
        2212:'neutron',
        130: 'kaon',
        230: 'kaon'
    }
    pred['particle'] = pred['PdgCode'].map(particle_mapping).fillna('NC')
    #print(pred)

    sns.set(style="whitegrid")
    df  = pred.sample(frac=0.01, random_state=1)
    print(df)
    # Create a histogram of 'prediction' for each 'particle_type'
    plt.figure(figsize=(10, 6))  # Set the figure size
    print("drawing")
    # sns.histplot(data=pred, x='Prediction', hue='particle' , bins=100)

    # plt.title('Distribution of Predictions by Particle Type')
    # plt.xlabel('Prediction Value')
    # plt.ylabel('Frequency')
    # plt.legend(title='Particle Type')

    # plot_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/outputs/plots/prediction_score_2.png'
    # plt.savefig(plot_path)
    # print("save plot at:", plot_path )

    for particle_type in df['particle'].unique():
        subset = df[df['particle'] == particle_type]
        plt.hist(subset['Prediction'], bins=30, alpha=0.5, label=str(particle_type))

    plt.title('Distribution of Predictions by Particle Type')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.legend(title='Particle Type')
    plt.grid(True)

# Show the plot
plt.show()

    # #print(pred)
    # unique_pdgCodes = pred['PdgCode'].unique()

    # # Print the unique pdgCodes
    # print(unique_pdgCodes)

if __name__ == "__main__":

    main()