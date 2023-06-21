import subprocess
import multiprocessing

# Define the command lines and GPU IDs
command_lines = [
    ("python3 /grand/EVITA/ct-mri/pcxgan/train_pcxgan_ct.py --exp_name pcx_fold1234_no_eq --test_fold 5 --epochs 1000 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/no_eq_mask/norm_mask_neg1pos1_fold'", 0),
    ("python3 /grand/EVITA/ct-mri/pcxgan/train_pcxgan_ct.py --exp_name pcx_fold1235_no_eq --test_fold 4 --epochs 1000 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/no_eq_mask/norm_mask_neg1pos1_fold'", 1), 
    ("python3 /grand/EVITA/ct-mri/pcxgan/train_pcxgan_ct.py --exp_name pcx_fold1245_no_eq --test_fold 3 --epochs 1000 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/no_eq_mask/norm_mask_neg1pos1_fold'", 2),
    ("python3 /grand/EVITA/ct-mri/pcxgan/train_pcxgan_ct.py --exp_name pcx_fold1345_no_eq --test_fold 2 --epochs 1000 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/no_eq_mask/norm_mask_neg1pos1_fold'", 3)
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