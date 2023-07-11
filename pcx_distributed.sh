# !/bin/bash
module load conda
conda activate 
python3 ~/pcxgan/train_pcx_distrib.py --exp_name pcx_distributed_test_500 --test_fold 1 --epochs 500 --epoch_interval 10 --batch_size 8 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_paired/norm_neg1pos1_fold'