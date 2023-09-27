import subprocess
import multiprocessing

# Define the command lines and GPU IDs
command_lines = [
    ("python3 ~/pcxgan/train_pcxgan_ckpt.py --exp_name pcx_seg_ct_no_aug_32_2345_4000 --test_fold 1 --epochs 2000 --epoch_interval 100 --batch_size 32 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_mask/norm_mask_neg1pos1_fold' --disc_path '/grand/EVITA/ct-mri/exp_results/models/aug_new/pcx_seg_ct_no_aug_32_2345_2000_disc' --dec_path '/grand/EVITA/ct-mri/exp_results/models/aug_new/pcx_seg_ct_no_aug_32_2345_2000_d' --enc_path '/grand/EVITA/ct-mri/exp_results/models/aug_new/pcx_seg_ct_no_aug_32_2345_2000_e'", 0)
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