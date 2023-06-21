import os
import time
import tensorflow as tf
import argparse

# Define the script to be executed on each worker node
script = '/media/aisec-102/DATA3/rachel/pcxgan/distributed/distrib_pcxgan_mask.py'

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file')
args = parser.parse_args()

# Read the list of worker nodes
with open(os.environ['PBS_NODEFILE'], 'r') as f:
    nodes = f.read().splitlines()

# Define the function to execute the script on each worker node
def execute_script(rank, num_nodes, node_file):
    # Construct the SSH command
    ssh_command = f'ssh {nodes[rank]} {script} {args.config} {num_nodes} {rank} {nodes[0]} {node_file}'

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