import subprocess
import multiprocessing

# Define the command lines and GPU IDs
command_lines = [
    ("python3 ~/gancm/train_gancm_512_aug.py --exp_name gancm_512_aug_500 --epochs 500 --batch_size 8 --remove_bad_images True --crop_size 512 --load_size 512", 0, 1),
    ("python3 ~/gancm/train_gancm_512_aug.py --exp_name gancm_512_aug_250 --epochs 250 --batch_size 8 --remove_bad_images True --crop_size 512 --load_size 512", 2, 3)
]

# Function to run a command on two GPUs
def run_command(command, gpu_id1, gpu_id2):
    gpu_option1 = f"--gpu_ids {gpu_id1}"
    gpu_option2 = f"--gpu_ids {gpu_id2}"
    full_command1 = command + " " + gpu_option1
    full_command2 = command + " " + gpu_option2
    subprocess.run(full_command1, shell=True)
    subprocess.run(full_command2, shell=True)

# Run the commands in parallel
processes = []
for cmd, gpu_id1, gpu_id2 in command_lines:
    process = multiprocessing.Process(target=run_command, args=(cmd, gpu_id1, gpu_id2))
    processes.append(process)
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()
