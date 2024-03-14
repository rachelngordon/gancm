# !/bin/bash
module load conda
conda activate 
python3 ~/gancm/train_uvit_spade.py --exp_name uvit_spade_test --data_path '/eagle/EVITA/ct-mri/data/CV/avg_eq_mask/norm_mask_neg1pos1_fold' --epochs 100 --epoch_interval 5 --batch_size 1