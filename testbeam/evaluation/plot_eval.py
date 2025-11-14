import ROOT
import pandas as pd
import os
import matplotlib.pyplot as plt

particle_2_class = {
    'e-': 0,
    'pi': 1,
}


def plot_fiducial_cuts(df, norm_factor, output_path="plot/fiducial_cut_counts.png"):
    fiducial_list = [0, 1]

    ve_counts = []
    vm_counts = []
    vt_counts = []
    nc_counts = []
    score_labels = []

    for f in fiducial_list:

        ve = df.Filter(f'PredClass == 0 && fudicial_{f} == 1 && ParticleClass == 0').Count().GetValue() * norm_factor
        vm = df.Filter(f'PredClass == 0 && fudicial_{f} == 1 && ParticleClass == 1').Count().GetValue() * norm_factor
        vt = df.Filter(f'PredClass == 0 && fudicial_{f} == 1 && ParticleClass == 2').Count().GetValue() * norm_factor
        nc = df.Filter(f'PredClass == 0 && fudicial_{f} == 1 && ParticleClass == 3').Count().GetValue() * norm_factor

        ve_counts.append(ve)
        vm_counts.append(vm)
        vt_counts.append(vt)
        nc_counts.append(nc)
        score_labels.append(f"fudicial_{f}")

    # Plotting
    x = range(len(fiducial_list))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar([i - 1.5*width for i in x], ve_counts, width, label='ve')
    plt.bar([i - 0.5*width for i in x], vm_counts, width, label='vm')
    plt.bar([i + 0.5*width for i in x], vt_counts, width, label='vt')
    plt.bar([i + 1.5*width for i in x], nc_counts, width, label='NC')

    plt.xticks(x, score_labels, rotation=45)
    plt.xlabel("fudicial cut")
    plt.ylabel("Count (PredClass == 0)")
    plt.title("ParticleClass Counts at Different Fudicial cuts")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_hit_cuts(df, norm_factor, output_path="plot/hit_cut_counts2.png"):
    n_hits_list = [100, 300, 500, 700, 900]

    ve_counts = []
    vm_counts = []
    vt_counts = []
    nc_counts = []
    score_labels = []

    for n_hit in n_hits_list:
        #print(n_hit)
        ve = df.Filter(f'PredClass == 0 && scifi_gt_{str(n_hit)} == 1 && ParticleClass == 0').Count().GetValue() * norm_factor
        vm = df.Filter(f'PredClass == 0 && scifi_gt_{str(n_hit)} == 1 && ParticleClass == 1').Count().GetValue() * norm_factor
        vt = df.Filter(f'PredClass == 0 && scifi_gt_{str(n_hit)} == 1 && ParticleClass == 2').Count().GetValue() * norm_factor
        nc = df.Filter(f'PredClass == 0 && scifi_gt_{str(n_hit)} == 1 && ParticleClass == 3').Count().GetValue() * norm_factor

        ve_counts.append(ve)
        vm_counts.append(vm)
        vt_counts.append(vt)
        nc_counts.append(nc)
        score_labels.append(f"scifi_gt_{n_hit}")

    # Plotting
    x = range(len(n_hits_list))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar([i - 1.5*width for i in x], ve_counts, width, label='ve')
    plt.bar([i - 0.5*width for i in x], vm_counts, width, label='vm')
    plt.bar([i + 0.5*width for i in x], vt_counts, width, label='vt')
    plt.bar([i + 1.5*width for i in x], nc_counts, width, label='NC')

    plt.xticks(x, score_labels, rotation=45)
    plt.xlabel("number of scifi hit")
    plt.ylabel("Count (PredClass == 0)")
    plt.title("ParticleClass Counts at Different Number of Scifi Hit Cuts")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_score_cuts(df, norm_factor, output_path="plot/score_cut_counts.png"):
    score_df = pd.read_csv('/afs/cern.ch/user/z/zhibin/work/snd-ml/data_quality_check/ve_cut_scores.csv')
    score_list = score_df['score'].astype(float)

    ve_counts = []
    vm_counts = []
    vt_counts = []
    nc_counts = []
    score_labels = []

    for score in score_list:
        colname = f"score_{str(score).replace('.', '_')}"

        ve = df.Filter(f'PredClass == 0 && {colname} == 1 && ParticleClass == 0').Count().GetValue() * norm_factor
        vm = df.Filter(f'PredClass == 0 && {colname} == 1 && ParticleClass == 1').Count().GetValue() * norm_factor
        vt = df.Filter(f'PredClass == 0 && {colname} == 1 && ParticleClass == 2').Count().GetValue() * norm_factor
        nc = df.Filter(f'PredClass == 0 && {colname} == 1 && ParticleClass == 3').Count().GetValue() * norm_factor

        ve_counts.append(ve)
        vm_counts.append(vm)
        vt_counts.append(vt)
        nc_counts.append(nc)
        score_labels.append(f"{score:.5f}")

    # Plotting
    x = range(len(score_list))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar([i - 1.5*width for i in x], ve_counts, width, label='ve')
    plt.bar([i - 0.5*width for i in x], vm_counts, width, label='vm')
    plt.bar([i + 0.5*width for i in x], vt_counts, width, label='vt')
    plt.bar([i + 1.5*width for i in x], nc_counts, width, label='NC')

    plt.xticks(x, score_labels, rotation=45)
    plt.xlabel("Score Threshold")
    plt.ylabel("Count (PredClass == 0)")
    plt.title("ParticleClass Counts at Different Score Thresholds")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300)
    plt.close()


def read_exist_output(dir_data, metadata_data_df):
    def file_exists(row):
        file_path = row['eval_baseline_output_path']
        return os.path.isfile(file_path)

    metadata_data_df = metadata_data_df[metadata_data_df.apply(file_exists, axis=1)].reset_index(drop=True)

    return metadata_data_df

def main():
    metadata_mc_neutrino_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'  
    metadata_mc_neutrino_df = pd.read_csv(metadata_mc_neutrino_path)
    dir_MC = '/eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1'
    metadata_mc_neutrino_df = read_exist_output(dir_MC, metadata_mc_neutrino_df)    

    prediction_chain = ROOT.TChain("snddata")

    model_name = 'baseline'
    n_file_count = 0
    for index, row in metadata_mc_neutrino_df.iterrows():
        
        pred_path = row[f'eval_{model_name}_output_path']
        print(f'reading...{pred_path}')
        prediction_chain.Add(pred_path)
        n_file_count+=1

    rdf = ROOT.RDataFrame(prediction_chain)

    real_data_lumi = 1.871e+02
    mc_neutrino_lumi = 100*n_file_count
    norm_factor = real_data_lumi/mc_neutrino_lumi


    #plot_score_cuts(rdf, norm_factor)
    plot_hit_cuts(rdf, norm_factor)
    #plot_fiducial_cuts(rdf, norm_factor)
    

    #plot_vt_score(rdf)
    #plot_vt_score_by_pred_particle(rdf)
    #plot_high_vt_score(rdf)
    #plot_2d_vt_score(rdf)
    #plot_max_pred_score(rdf)




if __name__ == "__main__":
    main()