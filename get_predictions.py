from flags import Flags
import data_loader
import evaluate

flags = Flags().parse()


# get pix2pix generated images

#test_data_path = "/grand/EVITA/ct-mri/data/norm_test/norm_test_data_pat1.npz"

test_dataset = data_loader.DataGenerator_PairedReady(flags, flags.test_data_path).load()

model_path = "/grand/EVITA/ct-mri/exp_results/models/p2p_avg_eq_1234"

for ct, mri in test_dataset:
    evaluate.predict_p2p(model_path, "p2p_avg_eq_1234", ct)
