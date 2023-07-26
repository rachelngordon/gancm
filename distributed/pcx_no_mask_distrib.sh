# !/bin/bash
module load conda
conda activate 
python3 ~/pcxgan/pcx_no_mask_distrib.py --exp_name pcx_distributed_test --test_fold 1 --epochs 10 --epoch_interval 1 --batch_size 64 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_paired/norm_neg1pos1_fold'