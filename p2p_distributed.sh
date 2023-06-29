# !/bin/bash
module load conda
conda activate 
python3 ~/pcxgan/train_p2p_distrib.py --exp_name p2p_distributed_test --test_fold 1 --epochs 100 --epoch_interval 10 --batch_size 8 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_paired/norm_neg1pos1_fold'