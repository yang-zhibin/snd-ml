import os

# Path to the directory you want to scan
root_dir = "/eos/experiment/sndlhc/users/zhibin/MC_neutrino/"

# Loop through all files in the directory
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # Check for the target pattern in filename
        if filename.startswith("model_baseline_output") and "_chunk_0" in filename:
            # New name with '_chunk_0' removed
            new_filename = filename.replace("_chunk_0", "")
            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirpath, new_filename)
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
