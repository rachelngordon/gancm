# !/bin/bash
module load conda
conda activate 
python3 ~/gancm/train_gancm_aug_mask.py --exp_name gancm_256_aug_no_central_crop --data_path '/eagle/EVITA/ct-mri/data/CV/avg_eq_mask/norm_mask_neg1pos1_fold' --epochs 20 --epoch_interval 5 --batch_size 32 --crop_val 1