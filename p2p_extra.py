import subprocess
import multiprocessing

# Define the command lines and GPU IDs
command_lines = [
    ("python3 ~/pcxgan/train_pix2pix.py --exp_name p2p_eq_1234 --test_fold 5 --epochs 500 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/eq_paired/norm_neg1pos1_fold'", 0),
    ("python3 ~/pcxgan/train_pix2pix.py --exp_name p2p_no_eq_1234 --test_fold 5 --epochs 500 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/no_eq_paired/norm_neg1pos1_fold'", 1),
    ("python3 ~/pcxgan/train_pcxgan_mask.py --exp_name pcx_seg_eq_2345 --test_fold 1 --epochs 500 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/eq_mask/norm_mask_neg1pos1_fold'", 2),
    ("python3 ~/pcxgan/train_pcxgan_mask.py --exp_name pcx_seg_no_eq_2345 --test_fold 1 --epochs 500 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/no_eq_mask/norm_mask_neg1pos1_fold'", 3)
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