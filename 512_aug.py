import subprocess
import multiprocessing

# Define the command lines and GPU IDs
command_lines = [
    ("python3 ~/gancm/train_gancm_512_aug.py --exp_name gancm_512_aug_1000 --epochs 1000 --batch_size 1 --remove_bad_images True --crop_size 512 --load_size 512", 0),
    ("python3 ~/gancm/train_gancm_512_aug.py --exp_name gancm_512_aug_500 --epochs 500 --batch_size 1 --remove_bad_images True --crop_size 512 --load_size 512", 1), 
    ("python3 ~/gancm/train_gancm_512_aug.py --exp_name gancm_512_aug_250 --epochs 250 --batch_size 1 --remove_bad_images True --crop_size 512 --load_size 512", 2)
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