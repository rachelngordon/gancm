#!/bin/bash

# Clear
echo running on a single node

SCRIPT='/grand/EVITA/ct-mri/pcxgan/polaris_exp/cycle_no_eq.py'
CFG='$1'

# Execute the script on the current node
$SCRIPT $1 1 0 $(hostname) $PBS_NODEFILE

# Remove existing files and directories
rm -rf ~/nodefile_return/*
mkdir -p ~/nodefile_return

# Staying alive...
file_path="$HOME/nodefile_return/${PBS_JOBID}_node$(hostname)"
while [ ! -f "$file_path" ]; do
    sleep 60  # Adjust the delay time as needed
done
