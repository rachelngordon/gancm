#!/bin/bash
module load conda
conda activate
python3 /grand/EVITA/ct-mri/pcxgan/distributed/distrib_pcxgan.py --exp_name distrib_pcxgan_test --test_fold 5 --epochs 1000 --batch_size 1 --data_path '/grand/EVITA/ct-mri/data/CV/no_eq_mask/norm_mask_neg1pos1_fold'