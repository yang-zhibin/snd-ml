import pandas as pd
import os
import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_plot(combined_data, save_path="plot/hits_3d_plots2.pdf", threshold=0.195):
    with PdfPages(save_path) as pdf:
        for i, event in enumerate(combined_data):
            pos = event["hits_pos"]
            label = event["hits_label"]
            pred = event["hits_prediction"]
            pdg = event["pdgCode"]
            run = event["runId"]
            event_id = event["eventId"]

            true_mask = label == 1
            pred_mask = pred > threshold

            true_count = np.sum(label == 1)
            all_count = len(label)
            print(f"Event {event_id} | PDG {pdg} | True hits: {true_count} / {all_count}")

            fig = plt.figure(figsize=(18, 5))

            # Plot 1: All hits
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="gray", s=1, alpha=0.1)
            ax1.set_title(f"All Hits\nEvent {event_id} | PDG {pdg}")

            # Plot 2: True hits
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            ax2.scatter(pos[true_mask, 0], pos[true_mask, 1], pos[true_mask, 2], c="green", s=1, alpha=0.1)
            ax2.set_title("True Hits (label=1)")

            # Plot 3: Predicted hits above threshold
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            ax3.scatter(pos[pred_mask, 0], pos[pred_mask, 1], pos[pred_mask, 2], c="red", s=1, alpha=0.1)
            ax3.set_title(f"Predicted Hits\n(score > {threshold})")

            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim(-50, 0)
                ax.set_ylim(10, 60)
                ax.set_zlim(275, 375)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved 3D plots to: {save_path}")

def plot_pred_score(combined_data, save_path="plot/3d_prediction_score_distribution2.png"):
    true_scores = []
    fake_scores = []

    for event in combined_data:
        preds = event["hits_prediction"]
        labels = event["hits_label"]

        true_scores.extend(preds[labels == 1])
        fake_scores.extend(preds[labels == 0])

    true_scores = np.array(true_scores)
    fake_scores = np.array(fake_scores)

    plt.figure(figsize=(8, 5))
    plt.hist(true_scores, bins=50, alpha=0.6, label="True hits (label=1)")
    plt.hist(fake_scores, bins=50, alpha=0.6, label="Fake hits (label=0)")
    plt.xlabel("Prediction score")
    plt.ylabel("Frequency")
    plt.title("Prediction score distribution by hit label")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

def read_exist_output(metadata_data_df):
    def file_exists(row):
        file_path = row['model_baseline_3d_output_path']
        return os.path.isfile(file_path)

    metadata_data_df = metadata_data_df[metadata_data_df.apply(file_exists, axis=1)].reset_index(drop=True)

    return metadata_data_df

def main():
    metadata_mc_neutrino_path = '/afs/cern.ch/user/z/zhibin/work/snd-ml/snakemake/metadata/updated/MC_neutrino_volTarget_100fb-1_metadata.csv'  
    metadata_mc_neutrino_df = pd.read_csv(metadata_mc_neutrino_path)

    # just keep rows where split_3d_1 == 'test'
    metadata_mc_neutrino_df = metadata_mc_neutrino_df[metadata_mc_neutrino_df["split_3d_1"] == "test"]

    metadata_mc_neutrino_df = read_exist_output(metadata_mc_neutrino_df)    

    print(metadata_mc_neutrino_df)
    
    for index, row in metadata_mc_neutrino_df.iterrows():
        hit_file = row['pkl_hit_path']
        pred_file = row ['model_baseline_3d_output_path']

        # load both files 
        with gzip.open(hit_file, 'rb') as f:
            hit_data = pickle.load(f)

        # Load hit data
        with gzip.open(hit_file, 'rb') as f:
            hit_data = pickle.load(f)
        
        # You can add similar loading code for pred_file if needed
        with gzip.open(pred_file, 'rb') as f:
            pred_data = pickle.load(f)

        

        hit_data = hit_data[:len(pred_data)]
        #print('hit_data:',(hit_data))
        #print('pred_data:',(pred_data))

        combined_data = []

        # Convert pred_data into a lookup dictionary for fast access
        pred_lookup = {
            (e["pdgCode"], e["runId"], e["eventId"]): e
            for e in pred_data
        }

        # Iterate through hit_data and find the matching prediction
        for hit_event in hit_data:
            key = (hit_event["pdgCode"], hit_event["runId"], hit_event["eventId"])
            
            pred_event = pred_lookup.get(key)
            if pred_event is None:
                print(f"Warning: No prediction found for {key}")
                continue

            # Combine the matching events
            combined_data.append({
                "pdgCode": hit_event["pdgCode"],
                "runId": hit_event["runId"],
                "eventId": hit_event["eventId"],
                "hits_pos": hit_event["hits_pos"],
                "hits_label": hit_event["hits_label"],
                "hits_prediction": pred_event["hits_prediction"]
            })
        #print(combined_data)

        #print(hit_data, pred_data)
        plot_pred_score(combined_data)
        plot_3d_plot(combined_data)

if __name__ == "__main__":
    main()