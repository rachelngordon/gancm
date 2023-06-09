import subprocess
import multiprocessing

# Define the command lines and GPU IDs
command_lines = [
    ("python3 train_pix2pix.py --exp_name p2p_fold2345_eq --test_fold 1 --epochs 1000 --batch_size 1 --equalized True", 0),
    ("python3 train_pix2pix.py --exp_name p2p_fold2345_no_eq --test_fold 1 --epochs 1000 --batch_size 1 --equalized False", 1), 
    ("python3 train_pcxgan_ct.py --exp_name pcx_fold2345_eq --test_fold 1 --epochs 1000 --batch_size 32 --equalized True", 2),
    ("python3 train_pcxgan_ct.py --exp_name pcx_fold2345_no_eq --test_fold 1 --epochs 1000 --batch_size 32 --equalized False", 3)
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