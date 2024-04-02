from flags import Flags
import data_loader
import evaluate

flags = Flags().parse()

# pass test fold to flags


# # get pix2pix generated images
# test_data_path = "/media/aisec-102/DATA3/rachel/data/CV/no_eq_paired/norm_neg1pos1_fold"

# test_dataset = data_loader.DataGenerator_PairedReady(flags, test_data_path, if_train=False).load()

# model_path = "/media/aisec-102/DATA3/rachel/experiments/models/p2p/p2p_no_eq_1234"

# for ct, mri in test_dataset:
#     evaluate.predict_p2p(model_path, "p2p_no_eq_1234", ct, mri)

# print("pix2pix complete")


# get pcxgan generated images
test_data_path = "/media/aisec-102/DATA3/rachel/data/anonymized_data/test2/gancm_processed_data/avg_eq_seg.npz"

test_dataset = data_loader.DataGenerator_Ready(flags, test_data_path, if_train=False).load()

model_path = "/media/aisec-102/DATA3/rachel/experiments/models/aug_exp_pixel/gancm_256_orig_aug_d"

for ct, mri, label in test_dataset:
    evaluate.predict_gancm_both(flags, model_path, ct, mri, label)


print("pcxgan complete")