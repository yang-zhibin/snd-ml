import os

def count_lines_in_python_files(directory):
    total_lines = 0
    total_files = 0
    ignore_dirs = ['/afs/cern.ch/user/z/zhibin/work/snd-ml/src/torchexplorer','/afs/cern.ch/user/z/zhibin/work/snd-ml/src/models']
    ignore_files = [
    "utils.py",
    "gnn_base.py",
    "exphormer.py",
    "fancyconv.py",
    "gravconv.py",
    "gravnet.py",
    "gravnetext.py",
    "multi_model.py"
]
    ignore_files = set(ignore_files)
    # Iterate through all the files in the directory
    for root, dirs, files in os.walk(directory):
        if any(os.path.abspath(root).startswith(ignore_dir) for ignore_dir in ignore_dirs):
            # Skip the current directory and its subdirectories
            dirs[:] = []
            continue
        for filename in files:
            if filename in ignore_files:
                continue

            # Check if the file is a Python file
            
            if filename.endswith(".py"):
                
                file_path = os.path.join(root, filename)
                print(filename)
                total_files +=1
                
                # Open and read the file
                with open(file_path, 'r', encoding='utf-8') as file:
                    # Count the number of lines in the file
                    lines = file.readlines()
                    total_lines += len(lines)
    
    return total_lines, total_files

# Specify the directory containing the Python files
directory_path = "/afs/cern.ch/user/z/zhibin/work/snd-ml/"

# Call the function and print the total number of lines
total_lines, total_files = count_lines_in_python_files(directory_path)
print(f"Total number of files of code in the directory: {total_files}")
print(f"Total number of lines of code in the directory: {total_lines}")