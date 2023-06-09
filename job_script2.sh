#!/bin/bash
#PBS -q preemptable    # Specify the queue you want to use
#PBS -l select=1:system=polaris   # Request one node
#PBS -l place=scatter
#PBS -l walltime=72:00:00 # Set the maximum wall time for your job
#PBS -l filesystems=home:grand  # Select filesystem
#PBS -N p2p_equalized        # Specify a name for your job
#PBS -o output_p2p_eq.log      # Redirect stdout to this file
#PBS -e error_p2p_eq.log       # Redirect stderr to this file

# Clear
echo running on a single node

SCRIPT='/grand/EVITA/ct-mri/pcxgan/polaris_exp/p2p_eq.py'
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