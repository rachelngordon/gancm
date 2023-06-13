#!/usr/bin/env python3

import subprocess
import sys
import os
import time

# Clear
print("running on a single node")

script_path = '/grand/EVITA/ct-mri/pcxgan/polaris_exp/extra_p2p.py'
cfg = sys.argv[1]

# Execute the script using the Python interpreter
subprocess.call(['python3', script_path, cfg, '1', '0', os.environ['HOSTNAME'], os.environ['PBS_NODEFILE']])

# Remove existing files and directories
return_dir = os.path.expanduser('~/nodefile_return')
os.makedirs(return_dir, exist_ok=True)
file_path = f"{return_dir}/{os.environ['PBS_JOBID']}_node{os.environ['HOSTNAME']}"
if os.path.exists(file_path):
    os.remove(file_path)

# Staying alive...
while not os.path.isfile(file_path):
    time.sleep(60)  # Adjust the delay time as needed