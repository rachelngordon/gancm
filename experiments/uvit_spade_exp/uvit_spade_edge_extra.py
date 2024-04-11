import subprocess
import multiprocessing

# Define the command lines and GPU IDs
command_lines = [
    ("python3 ~/gancm/train_uvit_spade.py --exp_name uvit_spade_bin_edge_eq_2345 --test_fold 1 --epochs 500 --batch_size 1 --data_path '/eagle/EVITA/ct-mri/data/CV/eq_bin_edge/norm_mask_neg1pos1_fold'", 0),
    ("python3 ~/gancm/train_uvit_spade.py --exp_name uvit_spade_bin_edge_avg_eq_2345 --test_fold 1 --epochs 500 --batch_size 1 --data_path '/eagle/EVITA/ct-mri/data/CV/avg_eq_bin_edge/norm_mask_neg1pos1_fold'", 2)
    ]

# Function to run a command on a GPU
def run_command(command, gpu_id):
    gpu_option = f"--gpu_ids {gpu_id}"
    full_command = command + " " + gpu_option
    subprocess.run(full_command, shell=True)

# Run the commands in parallel
processes = []
for cmd, gpu_id in command_lines:
    process = multiprocessing.Process(target=run_command, args=(cmd, gpu_id))
    processes.append(process)
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()