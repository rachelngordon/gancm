import os
import time
import tensorflow as tf

# Define the script to be executed on each worker node
script = '/media/aisec-102/DATA3/rachel/pcxgan/distrib_cyclegan.py'

# Define the configuration argument (replace with your actual configuration)
#config = '<your_config>'

# Define the flags to be passed to the script
flags = "--exp_name cycle_distrib_test --test_fold 5 --epochs 1000 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/no_eq_paired/norm_mask_neg1pos1_fold'"


# Read the list of worker nodes
with open(os.environ['PBS_NODEFILE'], 'r') as f:
    nodes = f.read().splitlines()

# Define the function to execute the script on each worker node
def execute_script(rank, num_nodes, node_file):
    # Construct the SSH command
    ssh_command = f'ssh {nodes[rank]} {script} {flags} {num_nodes} {rank} {nodes[0]} {node_file}'

    # Execute the SSH command
    os.system(ssh_command)

# Define the number of worker nodes
num_nodes = len(nodes)

# Execute the script on each worker node in parallel
for rank in range(num_nodes):
    execute_script(rank, num_nodes, os.environ['PBS_NODEFILE'])

# Wait for all worker nodes to generate their files
file_paths = [f'{os.environ["HOME"]}/nodefile_return/{os.environ["PBS_JOBID"]}_node{node}' for node in nodes]
while any(not os.path.isfile(file_path) for file_path in file_paths):
    time.sleep(60)

print("All files found!")