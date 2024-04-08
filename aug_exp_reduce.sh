# !/bin/bash
module load conda
conda activate 
python3 ~/gancm/train_gancm_aug_mask.py --exp_name gancm_256_aug_no_resize_reduce_params --data_path '/eagle/EVITA/ct-mri/data/CV/avg_eq_mask/norm_mask_neg1pos1_fold' --epochs 20 --epoch_interval 5 --batch_size 32 --rand_flip_h 0.5 --rand_flip_v 0.25 --bright_val 0.4 --rot_val 0.4 --crop_val 1