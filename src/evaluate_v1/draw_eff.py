import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_data(data, particle_type, title, output_dir, filenames):
    plt.figure(figsize=(10, 5))

    for df, filename in zip(data, filenames):
        # Extracting a meaningful part of the filename for the label
        label = os.path.basename(filename).replace('.csv', '').replace('_', ' ').title()
        plt.plot(df['partition'], df['model_yield'], marker='o', linestyle='-', label=label)
    
    plt.title(title)
    plt.xlabel('Energy Range (GeV)')
    plt.ylabel('bkg Yield')
    #plt.yscale('log')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{particle_type}_yield.png')
    plt.savefig(save_path)


def draw_eff():
    print("start drawing")

    csv_dir = "/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/yield/"
    output_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/plot'
    csv_files = [
        f'{csv_dir}/eff_binary_vm_2.7.csv',
        f'{csv_dir}/eff_binary_vm_weight_2.7.csv',
        f'{csv_dir}/eff_binary_vm_weight_recoMuon_2.7.csv'
    ]

    kaon_dfs = []
    neutron_dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        kaon_data = df[df['particle'] == 'kaon']
        neutron_data = df[df['particle'] == 'neutron']
        kaon_dfs.append(kaon_data)
        neutron_dfs.append(neutron_data)

    plot_data(kaon_dfs, 'Kaons', 'Efficiency of Kaons across Different Energy Ranges from Multiple Files', output_dir, csv_files)
    plot_data(neutron_dfs, 'Neutrons', 'Efficiency of Neutrons across Different Energy Ranges from Multiple Files', output_dir, csv_files)

if __name__ == "__main__":
    draw_eff()
