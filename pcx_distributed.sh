# !/bin/bash
module load conda
conda activate 
python3 ~/pcxgan/train_pcx_distrib.py --exp_name pcx_distributed_test_aug --test_fold 1 --epochs 10 --epoch_interval 1 --batch_size 64 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_mask/norm_mask_neg1pos1_fold'