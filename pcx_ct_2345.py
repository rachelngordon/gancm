import subprocess
import multiprocessing

# Define the command lines and GPU IDs
command_lines = [
    ("python3 ~/pcxgan/train_pcxgan_no_mask.py --exp_name pcx_ct_no_aug_8_2345 --test_fold 1 --epochs 500 --batch_size 8 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_paired/norm_neg1pos1_fold'", 0),
    ("python3 ~/pcxgan/train_pcxgan_no_mask.py --exp_name pcx_ct_no_aug_16_2345 --test_fold 1 --epochs 500 --batch_size 16 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_paired/norm_neg1pos1_fold'", 1), 
    ("python3 ~/pcxgan/train_pcxgan_no_mask.py --exp_name pcx_ct_no_aug_32_2345 --test_fold 1 --epochs 500 --batch_size 32 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_paired/norm_neg1pos1_fold'", 2),
    ("python3 ~/pcxgan/train_pcxgan_no_mask.py --exp_name pcx_ct_no_aug_48_2345 --test_fold 1 --epochs 500 --batch_size 48 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_paired/norm_neg1pos1_fold'", 3)
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