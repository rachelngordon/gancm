# !/bin/bash
module load conda
conda activate 
python3 ~/gancm/train_uvit_spade.py --exp_name uvit_spade_test --data_path '/media/aisec-102/DATA3/rachel/data/CV/avg_eq_mask/norm_mask_neg1pos1_fold' --epochs 10 --epoch_interval 1 --batch_size 1