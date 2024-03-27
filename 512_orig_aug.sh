# !/bin/bash
module load conda
conda activate 
python3 ~/gancm/train_gancm_512_aug.py --exp_name gancm_512_orig_aug --data_path '/eagle/EVITA/ct-mri/data/CV/avg_eq_mask/norm_mask_neg1pos1_fold' --epochs 20 --epoch_interval 5 --batch_size 8 --remove_bad_images True --crop_size 512 --load_size 512