# !/bin/bash
module load conda
conda activate 
python3 ~/pcxgan/train_pcx_aug_distrib.py --exp_name pcx_aug_32_1345_5000_dist --test_fold 2 --epochs 5000 --epoch_interval 250 --batch_size 32 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_paired/norm_neg1pos1_fold'
