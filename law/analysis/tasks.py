import law
import csv
import os

from analysis.framework import HTCondorWorkflow, Task

law.contrib.load("wlcg")


class Digi2Hits(Task, HTCondorWorkflow, law.LocalWorkflow):

    def create_branch_map(self):
        branch_map = {}

        # Open the CSV file and read it
        csv_file = "/afs/cern.ch/user/z/zhibin/work/snd-ml/law/metadata/test_metadata.csv"  # Path to your CSV file

        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Use the partition as the branch index (or any other key you want to use)
                branch = int(row['partition'])

                # Store each row's data as a dictionary or in a custom format for the branch
                branch_map[branch] = {
                    'data_type': row['data_type'],
                    'subfolder': row['subfolder'],
                    'n_event': int(row['n_event']),
                    'digi_path': row['digi_path'],
                    'geo_path': row['geo_path'],
                    'hit_path': row['hit_path']
                }

        # Return the branch map where each branch has data corresponding to a CSV row
        return branch_map

    def output(self):
        # it’s best practice to encode the branch number into the output target
        hit_path = self.branch_data['hit_path']
        return self.local_target(hit_path)

    def run(self):
        # Access branch data (this will be the row from the CSV)
        branch_data = self.branch_data
        
        # Extract the digi_path, geo_path, and other parameters
        digi_path = branch_data['digi_path']
        geo_path = branch_data['geo_path']
        hit_path = branch_data['hit_path']
        data_type = branch_data['data_type']

        # Particle type, if any (optional)
        particle_type = "neutrino"  # Or another default if needed

        # Path to the environment script
        env_script_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/law/envs/env_sndsw.sh"

        # Define the path to the Python script
        script_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/convertData/digi_2_hits.py"

        # Step 1: Source the environment script
        source_command = f"source {env_script_path}"
        try:
            result = subprocess.run(
                source_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print(f"Environment setup completed: {result.stdout.decode()}")
        except subprocess.CalledProcessError as e:
            print(f"Error sourcing environment: {e.stderr.decode()}")
            return  # Exit early if sourcing the environment fails

        # Step 2: Run the Python script with the arguments
        python_command = f"python {script_path} -d {digi_path} -g {geo_path} -o {hit_path} -t {data_type} -p {particle_type}"
        try:
            result = subprocess.run(
                python_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print(f"Python script executed successfully: {result.stdout.decode()}")
        except subprocess.CalledProcessError as e:
            print(f"Error running Python script: {e.stderr.decode()}")
            return  # Exit early if the Python script execution fails
