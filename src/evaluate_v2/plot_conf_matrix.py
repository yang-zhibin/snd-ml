import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
in_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate_v2/csv_matrix/'
infile = in_dir + 'baseline_muon_confusion_matrix_df.csv'
confusion_matrix = pd.read_csv(infile)
confusion_matrix.index = ['ve', 'vm', 'vt', 'NC', 'kaon', 'neutron', 'muon']

# Print the confusion matrix for inspection
print(confusion_matrix)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="0.2e", cmap="Blues", cbar=True, linewidths=.5)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Save the plot
save_path = in_dir + 'confusion_matrix_heatmap_baseline_muon.pdf'
plt.savefig(save_path)
