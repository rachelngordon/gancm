# !/bin/bash
module load conda
conda activate 
python3 ~/pcxgan/p2p_distributed.py --exp_name p2p_distributed_test --test_fold 1 --epochs 10 --epoch_interval 2 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/avg_eq_paired/norm_neg1pos1_fold'